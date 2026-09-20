//! [CACHE-SIG-SOURCE] OBSERVABILITY ONLY. No request mutation.
//!
//! Request-scoped, out-of-band attribution of the FINAL effective signature
//! source for each `functionCall` in a serialized upstream Gemini request.
//!
//! Two identities feed this module:
//! 1. During the synchronous request build a thread-local [`SigPlan`]
//!    accumulates builder attributions keyed by raw tool id (raw id is never
//!    logged).
//! 2. At the end of the builder the plan is registered in a global registry
//!    keyed by the final body's top-level `requestId`.
//!
//! `cache_diagnostics::observe_gated` later looks the plan up by
//! `body["requestId"]`, scans the FINAL body `request.contents[i].parts[j]`
//! for functionCall ids, and emits `[CACHE-SIG-SOURCE]` lines joined to the
//! same trajectory key and observation sequence as `[CACHE-DIAG]`.
//!
//! Last writer wins: the builder records first; a ThinkingStore restore
//! overwrites the entry to `PER_TURN_STORE` when it stamps a real signature.

use dashmap::DashMap;
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crate::proxy::cache_diagnostics::short_hash;

const MAX_RETAINED_PLANS: usize = 256;

/// Which kind of tool part an attribution belongs to. The internal plan key is
/// namespaced by part kind + raw tool id, so a `functionCall` and a
/// `functionResponse` carrying the same raw tool id cannot collide.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PartKind {
    FunctionCall,
    FunctionResponse,
    Thought,
}

impl PartKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            PartKind::FunctionCall => "functionCall",
            PartKind::FunctionResponse => "functionResponse",
            PartKind::Thought => "thought",
        }
    }
}

/// Internal plan key. Raw tool id stays process-memory only and is never logged.
fn plan_key(kind: PartKind, tool_id: &str) -> String {
    format!("{}|{}", kind.as_str(), tool_id)
}

/// Where the signature that ended up on the wire came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SigSource {
    PerTurnStore,
    ToolCache,
    SessionFallback,
    TrustedClient,
    Sentinel,
    None,
    FinalizeSiblingReal,
}

impl SigSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            SigSource::PerTurnStore => "PER_TURN_STORE",
            SigSource::ToolCache => "TOOL_CACHE",
            SigSource::SessionFallback => "SESSION_FALLBACK",
            SigSource::TrustedClient => "TRUSTED_CLIENT",
            SigSource::Sentinel => "SENTINEL",
            SigSource::None => "NONE",
            SigSource::FinalizeSiblingReal => "FINALIZE_SIBLING_REAL",
        }
    }
}

/// Which ThinkingStore restore phase matched the record, if any.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RestorePhase {
    P1ToolId,
    P2Fingerprint,
    P3Text,
    P4Tail,
    None,
}

impl RestorePhase {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            RestorePhase::P1ToolId => "P1_TOOL_ID",
            RestorePhase::P2Fingerprint => "P2_FINGERPRINT",
            RestorePhase::P3Text => "P3_TEXT",
            RestorePhase::P4Tail => "P4_TAIL",
            RestorePhase::None => "NONE",
        }
    }

    pub(crate) fn from_tag(tag: &str) -> Self {
        match tag {
            "P1_TOOL_ID" => RestorePhase::P1ToolId,
            "P2_FINGERPRINT" => RestorePhase::P2Fingerprint,
            "P3_TEXT" => RestorePhase::P3Text,
            "P4_TAIL" => RestorePhase::P4Tail,
            _ => RestorePhase::None,
        }
    }
}

/// [V2B] Action taken by the historical session-fallback stability policy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum V2bAction {
    None,
    StabilizedToSentinel,
}

impl V2bAction {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            V2bAction::None => "none",
            V2bAction::StabilizedToSentinel => "STABILIZED_TO_SENTINEL",
        }
    }
}

/// Per-tool-part attribution. Contains no raw request content.
#[derive(Clone, Debug)]
pub(crate) struct SigSourceEntry {
    pub source: SigSource,
    pub restore_phase: RestorePhase,
    pub store_record_hash: Option<String>,
    pub fallback_marked: bool,
    /// Which part kind this entry describes.
    pub part_kind: PartKind,
    /// "LATEST" for SessionFallback else "none".
    pub session_lookup: &'static str,
    /// None => print "none" (unknown; the API cannot expose the selected slot
    /// without a behavior change).
    pub session_message_count: Option<usize>,
    /// [V2B] Action applied by the historical session-fallback stability policy.
    pub v2b_action: V2bAction,
    /// [D4] Salted hash of the thought fingerprint (store fingerprint for a
    /// restore, else a once-hashed thought-text correlation). `None` for
    /// functionCall/functionResponse entries. Raw text/signature is never kept.
    pub thought_fingerprint_hash: Option<String>,
}

impl SigSourceEntry {
    fn builder(kind: PartKind, source: SigSource, fallback_marked: bool) -> Self {
        SigSourceEntry {
            source,
            restore_phase: RestorePhase::None,
            store_record_hash: None,
            fallback_marked,
            part_kind: kind,
            session_lookup: if source == SigSource::SessionFallback {
                "LATEST"
            } else {
                "none"
            },
            session_message_count: None,
            v2b_action: V2bAction::None,
            thought_fingerprint_hash: None,
        }
    }
}

/// Thread-local accumulation of builder + restore attributions for one request
/// build.
///
/// [D4] Thought parts have no tool id, so they are tracked separately by the
/// ORDINAL POSITION at which the builder pushed them. `thought_slots` holds the
/// builder attributions and `restore_thought_slots` holds the ThinkingStore
/// restore attributions; the final-body scan overlays the restore entries on
/// top of the builder entries by ordinal (restore wins), because restore runs
/// after the builder and only the final scan knows the serialized layout.
#[derive(Default)]
pub(crate) struct SigPlan {
    entries: HashMap<String, SigSourceEntry>,
    thought_slots: Vec<SigSourceEntry>,
    restore_thought_slots: Vec<SigSourceEntry>,
    /// [D4] Final-result source overrides keyed by the thought ordinal in the
    /// FINAL contents ordering observed by the scan (finalize / stabilizer).
    finalize_thought_sources: HashMap<usize, SigSourceEntry>,
}

/// Placeholder entry used to grow `thought_slots` when an earlier ordinal has
/// not been recorded (never emitted as a stale unrelated attribution).
fn thought_placeholder() -> SigSourceEntry {
    SigSourceEntry::builder(PartKind::Thought, SigSource::None, false)
}

/// Ensure `slots` has an entry at `idx`, pushing placeholders as needed.
fn ensure_slot(slots: &mut Vec<SigSourceEntry>, idx: usize) {
    while slots.len() <= idx {
        slots.push(thought_placeholder());
    }
}

impl SigPlan {
    fn record_builder(
        &mut self,
        kind: PartKind,
        tool_id: &str,
        source: SigSource,
        fallback_marked: bool,
    ) {
        if tool_id.is_empty() {
            return;
        }
        let key = plan_key(kind, tool_id);
        // Never let a builder record clobber an authoritative per-turn store
        // record (last writer wins in favor of the restore path).
        if let Some(existing) = self.entries.get(&key) {
            if existing.source == SigSource::PerTurnStore {
                return;
            }
        }
        self.entries
            .insert(key, SigSourceEntry::builder(kind, source, fallback_marked));
    }

    fn record_restore(
        &mut self,
        tool_id: &str,
        phase: RestorePhase,
        rec_hash: &str,
        fallback_marked: bool,
    ) {
        if tool_id.is_empty() {
            return;
        }
        // ThinkingStore restoration only ever stamps functionCall signatures.
        self.entries.insert(
            plan_key(PartKind::FunctionCall, tool_id),
            SigSourceEntry {
                source: SigSource::PerTurnStore,
                restore_phase: phase,
                store_record_hash: Some(rec_hash.to_string()),
                fallback_marked,
                part_kind: PartKind::FunctionCall,
                session_lookup: "none",
                session_message_count: None,
                v2b_action: V2bAction::None,
                thought_fingerprint_hash: None,
            },
        );
    }

    fn record_v2b_action(&mut self, kind: PartKind, tool_id: &str, action: V2bAction) {
        if tool_id.is_empty() {
            return;
        }
        if let Some(entry) = self.entries.get_mut(&plan_key(kind, tool_id)) {
            entry.v2b_action = action;
        }
    }

    /// [D4] Record the builder-selected source for a thought part identified by
    /// its builder ordinal slot. Never clobbers an authoritative per-turn store
    /// restore entry.
    fn record_builder_thought(
        &mut self,
        slot_index: usize,
        source: SigSource,
        fallback_marked: bool,
        fingerprint_hash: Option<String>,
    ) {
        ensure_slot(&mut self.thought_slots, slot_index);
        let entry = &mut self.thought_slots[slot_index];
        if entry.source == SigSource::PerTurnStore {
            return;
        }
        *entry = SigSourceEntry::builder(PartKind::Thought, source, fallback_marked);
        entry.thought_fingerprint_hash = fingerprint_hash;
    }

    /// [D4] Record a ThinkingStore restore attribution for a thought part
    /// (last writer wins). Stored in a parallel sidecar so the final scan can
    /// overlay it by ordinal.
    fn record_restore_thought(
        &mut self,
        slot_index: usize,
        phase: RestorePhase,
        rec_hash: &str,
        fallback_marked: bool,
        fingerprint_hash: Option<String>,
    ) {
        ensure_slot(&mut self.restore_thought_slots, slot_index);
        self.restore_thought_slots[slot_index] = SigSourceEntry {
            source: SigSource::PerTurnStore,
            restore_phase: phase,
            store_record_hash: Some(rec_hash.to_string()),
            fallback_marked,
            part_kind: PartKind::Thought,
            session_lookup: "none",
            session_message_count: None,
            v2b_action: V2bAction::None,
            thought_fingerprint_hash: fingerprint_hash,
        };
    }

    /// [D4] In-place source override for a finalize-assigned sibling REAL.
    ///
    /// A finalize sibling-REAL copy does not erase a meaningful builder
    /// provenance: if the builder already recorded where the value came from
    /// (session fallback, trusted client, tool cache, per-turn store), that
    /// attribution stays truthful because the sibling carries the same value.
    /// The override applies only when the slot has no builder attribution or a
    /// non-authoritative `SENTINEL`/`NONE` one, in which case the only honest
    /// source is `FINALIZE_SIBLING_REAL`.
    fn record_finalize_thought_source(&mut self, slot_index: usize, source: SigSource) {
        let existing = self.thought_slots.get(slot_index);
        if let Some(e) = existing {
            if !matches!(e.source, SigSource::Sentinel | SigSource::None) {
                // A meaningful builder provenance already describes the value
                // that the sibling REAL merely copied. Do not overwrite it.
                return;
            }
        }
        let mut entry = existing.cloned().unwrap_or_else(thought_placeholder);
        entry.source = source;
        entry.part_kind = PartKind::Thought;
        self.finalize_thought_sources.insert(slot_index, entry);
    }

    /// [D4] Attach the session message count to a builder thought slot.
    fn set_thought_session_message_count(&mut self, slot_index: usize, count: usize) {
        ensure_slot(&mut self.thought_slots, slot_index);
        self.thought_slots[slot_index].session_message_count = Some(count);
    }

    /// [D4] Record a V2B action on a thought slot.
    fn record_v2b_action_thought(&mut self, slot_index: usize, action: V2bAction) {
        if let Some(entry) = self.thought_slots.get_mut(slot_index) {
            entry.v2b_action = action;
        }
        if let Some(entry) = self.finalize_thought_sources.get_mut(&slot_index) {
            entry.v2b_action = action;
        }
    }

    /// [D4] A plan with any attribution at all must be retained by `finish_plan`
    /// so the final scan can reconcile thought ordinals.
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
            && self.thought_slots.is_empty()
            && self.restore_thought_slots.is_empty()
            && self.finalize_thought_sources.is_empty()
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<SigPlan>> = const { RefCell::new(None) };
}

/// RAII guard: owns the thread-local active plan for the duration of a request
/// build. Dropping it clears the slot so no plan leaks to a later request on
/// the same thread.
pub(crate) struct SigPlanGuard {
    armed: bool,
}

impl Drop for SigPlanGuard {
    fn drop(&mut self) {
        if self.armed {
            ACTIVE.with(|c| *c.borrow_mut() = None);
        }
    }
}

/// Arms a fresh thread-local plan when diagnostics are enabled. When disabled
/// this does nothing and the returned guard is inert.
pub(crate) fn begin_plan() -> SigPlanGuard {
    if is_enabled() {
        ACTIVE.with(|c| *c.borrow_mut() = Some(SigPlan::default()));
        SigPlanGuard { armed: true }
    } else {
        SigPlanGuard { armed: false }
    }
}

/// Record the builder-selected source for one functionCall. Cheap no-op when
/// diagnostics are disabled or no plan is active.
pub(crate) fn record_builder_function_call(tool_id: &str, source: SigSource, fallback_marked: bool) {
    if !is_enabled() {
        return;
    }
    ACTIVE.with(|c| {
        if let Some(plan) = c.borrow_mut().as_mut() {
            plan.record_builder(PartKind::FunctionCall, tool_id, source, fallback_marked);
        }
    });
}

/// Record the builder-selected source for one functionResponse. Cheap no-op
/// when diagnostics are disabled or no plan is active.
pub(crate) fn record_builder_function_response(
    tool_id: &str,
    source: SigSource,
    fallback_marked: bool,
) {
    if !is_enabled() {
        return;
    }
    ACTIVE.with(|c| {
        if let Some(plan) = c.borrow_mut().as_mut() {
            plan.record_builder(PartKind::FunctionResponse, tool_id, source, fallback_marked);
        }
    });
}

/// [D4] Salted hash of a thought fingerprint (store fingerprint for a restore,
/// else a once-hashed thought-text correlation). Never log the raw input.
pub(crate) fn thought_fingerprint_hash(fp: &str) -> String {
    short_hash(&format!("thoughtfp|{fp}"))
}

/// [D4] Record the builder-selected source for one thought part by its builder
/// ordinal slot. Cheap no-op when diagnostics are disabled or no plan is active.
pub(crate) fn record_builder_thought(
    slot_index: usize,
    source: SigSource,
    fallback_marked: bool,
    fingerprint_hash: Option<String>,
) {
    if !is_enabled() {
        return;
    }
    ACTIVE.with(|c| {
        if let Some(plan) = c.borrow_mut().as_mut() {
            plan.record_builder_thought(slot_index, source, fallback_marked, fingerprint_hash);
        }
    });
}

/// [D4] Record a ThinkingStore restore attribution for one thought part (last
/// writer wins). Cheap no-op when diagnostics are disabled or no plan is active.
pub(crate) fn record_restore_thought(
    slot_index: usize,
    phase_tag: &str,
    rec_hash: &str,
    fallback_marked: bool,
    fingerprint_hash: Option<String>,
) {
    if !is_enabled() {
        return;
    }
    let phase = RestorePhase::from_tag(phase_tag);
    ACTIVE.with(|c| {
        if let Some(plan) = c.borrow_mut().as_mut() {
            plan.record_restore_thought(
                slot_index,
                phase,
                rec_hash,
                fallback_marked,
                fingerprint_hash,
            );
        }
    });
}

/// [D4] In-place source override for a finalize-assigned sibling REAL. Cheap
/// no-op when diagnostics are disabled or no plan is active.
pub(crate) fn record_finalize_thought_source(slot_index: usize, source: SigSource) {
    if !is_enabled() {
        return;
    }
    ACTIVE.with(|c| {
        if let Some(plan) = c.borrow_mut().as_mut() {
            plan.record_finalize_thought_source(slot_index, source);
        }
    });
}

/// [D4] Attach the request message count to a builder thought slot. Cheap no-op
/// when diagnostics are disabled or no plan is active.
pub(crate) fn set_thought_session_message_count(slot_index: usize, count: usize) {
    if !is_enabled() {
        return;
    }
    ACTIVE.with(|c| {
        if let Some(plan) = c.borrow_mut().as_mut() {
            plan.set_thought_session_message_count(slot_index, count);
        }
    });
}

/// [D4] Attribute a V2B stability action to a thought slot. Cheap no-op when
/// diagnostics are disabled or no plan is active.
pub(crate) fn record_v2b_action_thought(slot_index: usize, action: V2bAction) {
    if !is_enabled() {
        return;
    }
    ACTIVE.with(|c| {
        if let Some(plan) = c.borrow_mut().as_mut() {
            plan.record_v2b_action_thought(slot_index, action);
        }
    });
}

/// Record a ThinkingStore restore attribution (last writer wins). Cheap no-op
/// when diagnostics are disabled or no plan is active.
pub(crate) fn record_restore_function_call(
    tool_id: &str,
    phase_tag: &str,
    rec_hash: &str,
    fallback_marked: bool,
) {
    if !is_enabled() {
        return;
    }
    let phase = RestorePhase::from_tag(phase_tag);
    ACTIVE.with(|c| {
        if let Some(plan) = c.borrow_mut().as_mut() {
            plan.record_restore(tool_id, phase, rec_hash, fallback_marked);
        }
    });
}

/// [V2B] Attribute a historical session-fallback stability action to a
/// functionCall or functionResponse. Cheap no-op when diagnostics are disabled,
/// no plan is active, or the slot has no builder attribution.
pub(crate) fn record_v2b_action(kind: PartKind, tool_id: &str, action: V2bAction) {
    if !is_enabled() {
        return;
    }
    ACTIVE.with(|c| {
        if let Some(plan) = c.borrow_mut().as_mut() {
            plan.record_v2b_action(kind, tool_id, action);
        }
    });
}

struct Registry {
    plans: DashMap<String, (u64, Arc<SigPlan>)>,
    seq: AtomicU64,
}

fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| Registry {
        plans: DashMap::new(),
        seq: AtomicU64::new(0),
    })
}

fn evict_oldest(reg: &Registry) {
    while reg.plans.len() > MAX_RETAINED_PLANS {
        let mut oldest: Option<(String, u64)> = None;
        for e in reg.plans.iter() {
            let seq = e.value().0;
            match &oldest {
                Some((_, sq)) if *sq <= seq => {}
                _ => oldest = Some((e.key().clone(), seq)),
            }
        }
        match oldest {
            Some((k, _)) => {
                if reg.plans.remove(&k).is_none() {
                    break;
                }
            }
            None => break,
        }
    }
}

/// Take the active plan and register it under the final body's top-level
/// `requestId`. No-op when disabled, when no plan was armed, when there are no
/// entries, or when the body carries no usable requestId.
pub(crate) fn finish_plan(body: &Value) {
    if !is_enabled() {
        return;
    }
    let plan = ACTIVE.with(|c| c.borrow_mut().take());
    let Some(plan) = plan else { return };
    if plan.is_empty() {
        return;
    }
    let Some(key) = body
        .get("requestId")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .filter(|s| !s.is_empty())
    else {
        return;
    };
    let reg = registry();
    let seq = reg.seq.fetch_add(1, Ordering::Relaxed);
    reg.plans.insert(key, (seq, Arc::new(plan)));
    evict_oldest(reg);
}

/// Salted hash of a ThinkingRecord fingerprint. Never log the raw fingerprint.
pub(crate) fn store_record_hash(fingerprint: &str) -> String {
    short_hash(&format!("rec|{fingerprint}"))
}

/// One attributed functionCall in the FINAL serialized layout.
#[derive(Clone, Debug)]
pub(crate) struct SigSourceObservation {
    pub content_index: usize,
    pub part_index: usize,
    /// Salted hash of the raw tool id. "none" for thought parts, which have no id.
    pub tool_id_hash: String,
    pub entry: SigSourceEntry,
    /// [D4] Salted thought fingerprint hash; `None` for functionCall /
    /// functionResponse observations.
    pub thought_fingerprint_hash: Option<String>,
}

/// Registry lookup by `body["requestId"]` plus a scan of the FINAL body layout.
/// Indices are the serialized final indices, so they remain correct even after
/// finalize reorders the thought part to index 0 or merges contents.
pub(crate) fn observations_for_body(body: &Value) -> Vec<SigSourceObservation> {
    let mut out = Vec::new();
    let Some(key) = body.get("requestId").and_then(|v| v.as_str()) else {
        return out;
    };
    if key.is_empty() {
        return out;
    }
    let reg = registry();
    let Some(plan) = reg.plans.get(key).map(|e| e.value().1.clone()) else {
        return out;
    };

    let inner = body.get("request").unwrap_or(body);
    if let Some(contents) = inner.get("contents").and_then(|c| c.as_array()) {
        // Pass 1: functionCall (existing behavior/order preserved).
        for (ci, content) in contents.iter().enumerate() {
            let Some(parts) = content.get("parts").and_then(|p| p.as_array()) else {
                continue;
            };
            for (pi, part) in parts.iter().enumerate() {
                let Some(id) = part
                    .get("functionCall")
                    .and_then(|f| f.get("id"))
                    .and_then(|v| v.as_str())
                else {
                    continue;
                };
                if let Some(entry) = plan.entries.get(&plan_key(PartKind::FunctionCall, id)) {
                    out.push(SigSourceObservation {
                        content_index: ci,
                        part_index: pi,
                        tool_id_hash: short_hash(&format!("toolid|{id}")),
                        entry: entry.clone(),
                        thought_fingerprint_hash: entry.thought_fingerprint_hash.clone(),
                    });
                }
            }
        }
        // Pass 2: functionResponse (V2C).
        for (ci, content) in contents.iter().enumerate() {
            let Some(parts) = content.get("parts").and_then(|p| p.as_array()) else {
                continue;
            };
            for (pi, part) in parts.iter().enumerate() {
                let Some(id) = part
                    .get("functionResponse")
                    .and_then(|f| f.get("id"))
                    .and_then(|v| v.as_str())
                else {
                    continue;
                };
                if let Some(entry) = plan.entries.get(&plan_key(PartKind::FunctionResponse, id)) {
                    out.push(SigSourceObservation {
                        content_index: ci,
                        part_index: pi,
                        tool_id_hash: short_hash(&format!("toolid|{id}")),
                        entry: entry.clone(),
                        thought_fingerprint_hash: entry.thought_fingerprint_hash.clone(),
                    });
                }
            }
        }

        // Pass 3: [D4] thought parts, reconciled by ordinal.
        //
        // Thought parts carry no tool id, so the builder recorded them by their
        // builder ordinal. `finalize` reorders thoughts to the front of a turn
        // and may synthesize one, so the mapping from builder ordinal to final
        // position is not identity. The Nth thought part in the FINAL
        // `contents` -> `parts` order maps to `thought_slots[N]` (and is
        // overlaid by `restore_thought_slots[N]`, which wins because the
        // ThinkingStore restore runs after the builder). A slot that has no
        // builder attribution emits a synthesized `part_kind=thought,
        // sig_source=NONE` entry rather than a stale unrelated one.
        //
        // LIMITATION: if the inbound pipeline demotes or drops a thought (e.g.
        // a non-first thought becomes plain text), the surviving thoughts shift
        // down and later ordinals can be attributed to the wrong logical turn.
        // The scan only ever emits for parts that are actually `thought==true`.
        let mut thought_ordinal: usize = 0;
        for (ci, content) in contents.iter().enumerate() {
            let Some(parts) = content.get("parts").and_then(|p| p.as_array()) else {
                continue;
            };
            for (pi, part) in parts.iter().enumerate() {
                let is_thought = part.get("thought").and_then(|t| t.as_bool()) == Some(true);
                if !is_thought {
                    continue;
                }
                let idx = thought_ordinal;
                thought_ordinal += 1;
                // Restore (authoritative per-turn) overlays the builder entry.
                let mut entry = if let Some(restore) = plan.restore_thought_slots.get(idx) {
                    restore.clone()
                } else if let Some(builder) = plan.thought_slots.get(idx) {
                    builder.clone()
                } else {
                    thought_placeholder()
                };
                // Finalize / stabilizer override wins over builder-only entries,
                // but never over an authoritative per-turn restore.
                if let Some(fin) = plan.finalize_thought_sources.get(&idx) {
                    if entry.source != SigSource::PerTurnStore {
                        entry.source = fin.source;
                    }
                    if fin.v2b_action != V2bAction::None {
                        entry.v2b_action = fin.v2b_action;
                    }
                    if entry.thought_fingerprint_hash.is_none() {
                        entry.thought_fingerprint_hash = fin.thought_fingerprint_hash.clone();
                    }
                }
                entry.part_kind = PartKind::Thought;
                let fp_hash = entry.thought_fingerprint_hash.clone();
                out.push(SigSourceObservation {
                    content_index: ci,
                    part_index: pi,
                    tool_id_hash: "none".to_string(),
                    thought_fingerprint_hash: fp_hash,
                    entry,
                });
            }
        }
    }
    out
}

/// EXACT emitted line format. Only salted hashes / enums / counts appear.
pub(crate) fn format_line(obs: &SigSourceObservation, trajectory: &str, seq: u64) -> String {
    format!(
        "[CACHE-SIG-SOURCE] trajectory={} obs_seq={} content_index={} part_index={} part_kind={} sig_source={} tool_id_hash={} store_record_hash={} restore_phase={} session_lookup={} session_message_count={} fallback_marked={} v2b_action={} thought_fingerprint_hash={}",
        trajectory,
        seq,
        obs.content_index,
        obs.part_index,
        obs.entry.part_kind.as_str(),
        obs.entry.source.as_str(),
        obs.tool_id_hash,
        obs.entry.store_record_hash.as_deref().unwrap_or("none"),
        obs.entry.restore_phase.as_str(),
        obs.entry.session_lookup,
        obs.entry
            .session_message_count
            .map(|v| v.to_string())
            .unwrap_or_else(|| "none".to_string()),
        obs.entry.fallback_marked,
        obs.entry.v2b_action.as_str(),
        obs.entry
            .thought_fingerprint_hash
            .as_deref()
            .unwrap_or("none"),
    )
}

/// Emit one `[CACHE-SIG-SOURCE]` line per attributed functionCall. No-op when
/// diagnostics are disabled.
pub(crate) fn emit_for_body(body: &Value, trajectory: &str, seq: u64) {
    if !is_enabled() {
        return;
    }
    for obs in observations_for_body(body) {
        tracing::info!("{}", format_line(&obs, trajectory, seq));
    }
}

/// Gated enablement. Reuses the existing `ABV_CACHE_DIAGNOSTICS` package flag.
pub(crate) fn is_enabled() -> bool {
    #[cfg(test)]
    {
        if let Some(v) = TEST_ENABLED_OVERRIDE.with(|c| c.get()) {
            return v;
        }
    }
    crate::proxy::cache_diagnostics::is_enabled()
}

#[cfg(test)]
thread_local! {
    static TEST_ENABLED_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_enabled_for_tests(v: bool) {
    TEST_ENABLED_OVERRIDE.with(|c| c.set(Some(v)));
}

#[cfg(test)]
pub(crate) fn clear_enabled_for_tests() {
    TEST_ENABLED_OVERRIDE.with(|c| c.set(None));
}

/// Serializes tests that touch the global registry so a `reset_for_tests` in
/// one test cannot clear another test's plan mid-assertion.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Clear the global registry and the thread-local active plan.
#[cfg(test)]
pub(crate) fn reset_for_tests() {
    let reg = registry();
    reg.plans.clear();
    reg.seq.store(0, Ordering::Relaxed);
    ACTIVE.with(|c| *c.borrow_mut() = None);
}

#[cfg(test)]
pub(crate) fn registry_len_for_tests() -> usize {
    registry().plans.len()
}

#[cfg(test)]
pub(crate) fn active_plan_present() -> bool {
    ACTIVE.with(|c| c.borrow().is_some())
}

#[cfg(test)]
pub(crate) fn has_plan_for_body(body: &Value) -> bool {
    let Some(key) = body.get("requestId").and_then(|v| v.as_str()) else {
        return false;
    };
    if key.is_empty() {
        return false;
    }
    registry().plans.contains_key(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obs(entry: SigSourceEntry) -> SigSourceObservation {
        SigSourceObservation {
            content_index: 1,
            part_index: 2,
            tool_id_hash: short_hash("toolid|call_x"),
            thought_fingerprint_hash: entry.thought_fingerprint_hash.clone(),
            entry,
        }
    }

    #[test]
    fn d3_unit_format_line_fields_present() {
        set_enabled_for_tests(true);
        let entry = SigSourceEntry::builder(PartKind::FunctionCall, SigSource::ToolCache, false);
        let line = format_line(&obs(entry), "traj123", 7);
        assert!(line.starts_with("[CACHE-SIG-SOURCE] "));
        assert!(line.contains("trajectory=traj123"));
        assert!(line.contains("obs_seq=7"));
        assert!(line.contains("content_index=1"));
        assert!(line.contains("part_index=2"));
        assert!(line.contains("part_kind=functionCall"));
        assert!(line.contains("sig_source=TOOL_CACHE"));
        assert!(line.contains("tool_id_hash="));
        assert!(line.contains("store_record_hash=none"));
        assert!(line.contains("restore_phase=NONE"));
        assert!(line.contains("session_lookup=none"));
        assert!(line.contains("session_message_count=none"));
        assert!(line.contains("fallback_marked=false"));
        clear_enabled_for_tests();
    }

    #[test]
    fn d3_unit_last_writer_wins_builder_then_restore() {
        set_enabled_for_tests(true);
        let fc_key = plan_key(PartKind::FunctionCall, "call_1");
        let mut plan = SigPlan::default();
        plan.record_builder(PartKind::FunctionCall, "call_1", SigSource::SessionFallback, true);
        assert_eq!(
            plan.entries.get(&fc_key).unwrap().source,
            SigSource::SessionFallback
        );
        plan.record_restore("call_1", RestorePhase::P1ToolId, "hash123", true);
        let e = plan.entries.get(&fc_key).unwrap();
        assert_eq!(e.source, SigSource::PerTurnStore);
        assert_eq!(e.restore_phase, RestorePhase::P1ToolId);
        assert_eq!(e.store_record_hash.as_deref(), Some("hash123"));
        // A later builder record must not clobber the restore.
        plan.record_builder(PartKind::FunctionCall, "call_1", SigSource::ToolCache, false);
        assert_eq!(
            plan.entries.get(&fc_key).unwrap().source,
            SigSource::PerTurnStore
        );
        clear_enabled_for_tests();
    }

    #[test]
    fn d3_unit_part_kind_namespaces_same_raw_id() {
        set_enabled_for_tests(true);
        let fc_key = plan_key(PartKind::FunctionCall, "call_1");
        let fr_key = plan_key(PartKind::FunctionResponse, "call_1");
        let mut plan = SigPlan::default();
        plan.record_builder(PartKind::FunctionCall, "call_1", SigSource::ToolCache, false);
        plan.record_builder(
            PartKind::FunctionResponse,
            "call_1",
            SigSource::SessionFallback,
            true,
        );
        assert_eq!(
            plan.entries.get(&fc_key).unwrap().source,
            SigSource::ToolCache
        );
        assert_eq!(
            plan.entries.get(&fc_key).unwrap().part_kind,
            PartKind::FunctionCall
        );
        assert_eq!(
            plan.entries.get(&fr_key).unwrap().source,
            SigSource::SessionFallback
        );
        assert_eq!(
            plan.entries.get(&fr_key).unwrap().part_kind,
            PartKind::FunctionResponse
        );
        plan.record_v2b_action(
            PartKind::FunctionResponse,
            "call_1",
            V2bAction::StabilizedToSentinel,
        );
        assert_eq!(
            plan.entries.get(&fr_key).unwrap().v2b_action,
            V2bAction::StabilizedToSentinel
        );
        assert_eq!(
            plan.entries.get(&fc_key).unwrap().v2b_action,
            V2bAction::None
        );
        clear_enabled_for_tests();
    }

    #[test]
    fn d3_unit_disabled_begin_plan_inert() {
        let _lock = test_lock();
        set_enabled_for_tests(false);
        reset_for_tests();
        let _guard = begin_plan();
        assert!(!active_plan_present());
        record_builder_function_call("call_1", SigSource::ToolCache, false);
        assert!(!active_plan_present());
        finish_plan(&json!({"requestId": "r1"}));
        assert_eq!(registry_len_for_tests(), 0);
        clear_enabled_for_tests();
    }

    // ==================================================================
    // [D4] thought-sidecar unit tests
    // ==================================================================

    #[test]
    fn d4_unit_thought_ordinal_reconciles_to_final_indices() {
        let _lock = test_lock();
        set_enabled_for_tests(true);
        reset_for_tests();
        // Plan with two builder thought slots; slot 0 gets a session fallback,
        // slot 1 a sentinel. No functionCall entries.
        {
            let _guard = begin_plan();
            assert!(active_plan_present(), "guard arms an empty plan");
            record_builder_thought(
                0,
                SigSource::SessionFallback,
                true,
                Some(thought_fingerprint_hash("fp0")),
            );
            record_builder_thought(1, SigSource::Sentinel, false, None);
            set_thought_session_message_count(0, 5);
            finish_plan(&json!({"requestId": "d4_unit_r1"}));
        }
        // The final body interleaves a non-thought part, so the Nth thought part
        // must map to thought slot N and carry the FINAL indices.
        let body = json!({
            "requestId": "d4_unit_r1",
            "request": {
                "contents": [
                    { "role": "user", "parts": [ { "text": "hi" } ] },
                    { "role": "model", "parts": [
                        { "text": "t0", "thought": true, "thoughtSignature": "x" },
                        { "text": "visible" },
                        { "text": "t1", "thought": true, "thoughtSignature": "y" }
                    ] }
                ]
            }
        });
        let obs = observations_for_body(&body);
        let thoughts: Vec<_> = obs
            .iter()
            .filter(|o| o.entry.part_kind == PartKind::Thought)
            .collect();
        assert_eq!(thoughts.len(), 2, "one observation per final thought part");
        assert_eq!((thoughts[0].content_index, thoughts[0].part_index), (1, 0));
        assert_eq!(thoughts[0].entry.source, SigSource::SessionFallback);
        assert!(thoughts[0].entry.fallback_marked);
        assert_eq!(thoughts[0].entry.session_message_count, Some(5));
        assert_eq!(
            thoughts[0].thought_fingerprint_hash.as_deref(),
            Some(thought_fingerprint_hash("fp0").as_str())
        );
        assert_eq!((thoughts[1].content_index, thoughts[1].part_index), (1, 2));
        assert_eq!(thoughts[1].entry.source, SigSource::Sentinel);
        assert!(!thoughts[1].entry.fallback_marked);
        assert_eq!(thoughts[1].entry.session_message_count, None);
        // tool_id_hash is "none" for thought observations.
        assert_eq!(thoughts[0].tool_id_hash, "none");
        clear_enabled_for_tests();
        reset_for_tests();
    }

    #[test]
    fn d4_unit_thought_fingerprint_hash_is_16_hex() {
        let h = thought_fingerprint_hash("some-store-fingerprint");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(h, thought_fingerprint_hash("some-store-fingerprint"));
        assert_ne!(h, thought_fingerprint_hash("other-fingerprint"));
        // Deterministic within the process (salted), but never contains the raw.
        assert!(!h.contains("some-store-fingerprint"));
    }

    #[test]
    fn d4_unit_restore_overlays_builder_by_ordinal() {
        let _lock = test_lock();
        set_enabled_for_tests(true);
        reset_for_tests();
        {
            let _guard = begin_plan();
            // Builder recorded a session fallback for slot 0 ...
            record_builder_thought(0, SigSource::SessionFallback, true, None);
            // ... but an authoritative ThinkingStore restore replaced it.
            record_restore_thought(0, "P1_TOOL_ID", "rechash", true, Some(thought_fingerprint_hash("recfp")));
            finish_plan(&json!({"requestId": "d4_unit_r2"}));
        }
        let body = json!({
            "requestId": "d4_unit_r2",
            "request": { "contents": [
                { "role": "model", "parts": [ { "text": "t", "thought": true } ] }
            ]}
        });
        let obs = observations_for_body(&body);
        let t = obs.iter().find(|o| o.entry.part_kind == PartKind::Thought).unwrap();
        assert_eq!(t.entry.source, SigSource::PerTurnStore, "restore wins over builder");
        assert_eq!(t.entry.restore_phase, RestorePhase::P1ToolId);
        assert_eq!(t.entry.store_record_hash.as_deref(), Some("rechash"));
        clear_enabled_for_tests();
        reset_for_tests();
    }

    #[test]
    fn d4_unit_missing_slot_synthesizes_none_not_stale() {
        let _lock = test_lock();
        set_enabled_for_tests(true);
        reset_for_tests();
        {
            let _guard = begin_plan();
            // Only slot 0 is known; the final body will have TWO thought parts.
            record_builder_thought(0, SigSource::ToolCache, false, None);
            finish_plan(&json!({"requestId": "d4_unit_r3"}));
        }
        let body = json!({
            "requestId": "d4_unit_r3",
            "request": { "contents": [
                { "role": "model", "parts": [
                    { "text": "a", "thought": true },
                    { "text": "b", "thought": true }
                ]}
            ]}
        });
        let obs = observations_for_body(&body);
        let thoughts: Vec<_> = obs
            .iter()
            .filter(|o| o.entry.part_kind == PartKind::Thought)
            .collect();
        assert_eq!(thoughts.len(), 2);
        assert_eq!(thoughts[0].entry.source, SigSource::ToolCache);
        assert_eq!(
            thoughts[1].entry.source,
            SigSource::None,
            "a missing slot must synthesize NONE, never a stale unrelated entry"
        );
        clear_enabled_for_tests();
        reset_for_tests();
    }

    #[test]
    fn d4_unit_disabled_thought_records_inert() {
        let _lock = test_lock();
        set_enabled_for_tests(false);
        reset_for_tests();
        {
            let _guard = begin_plan();
            assert!(!active_plan_present());
            record_builder_thought(0, SigSource::SessionFallback, true, None);
            record_restore_thought(0, "P1_TOOL_ID", "h", false, None);
            assert!(!active_plan_present());
            finish_plan(&json!({"requestId": "d4_unit_r4"}));
        }
        assert_eq!(registry_len_for_tests(), 0);
        clear_enabled_for_tests();
    }
}
