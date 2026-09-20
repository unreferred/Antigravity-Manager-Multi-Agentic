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

/// Where the signature that ended up on the wire came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SigSource {
    PerTurnStore,
    ToolCache,
    SessionFallback,
    TrustedClient,
    Sentinel,
    None,
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

/// Per-functionCall attribution. Contains no raw request content.
#[derive(Clone, Debug)]
pub(crate) struct SigSourceEntry {
    pub source: SigSource,
    pub restore_phase: RestorePhase,
    pub store_record_hash: Option<String>,
    pub fallback_marked: bool,
    /// "LATEST" for SessionFallback else "none".
    pub session_lookup: &'static str,
    /// None => print "none" (unknown; the API cannot expose the selected slot
    /// without a behavior change).
    pub session_message_count: Option<usize>,
    /// [V2B] Action applied by the historical session-fallback stability policy.
    pub v2b_action: V2bAction,
}

impl SigSourceEntry {
    fn builder(source: SigSource, fallback_marked: bool) -> Self {
        SigSourceEntry {
            source,
            restore_phase: RestorePhase::None,
            store_record_hash: None,
            fallback_marked,
            session_lookup: if source == SigSource::SessionFallback {
                "LATEST"
            } else {
                "none"
            },
            session_message_count: None,
            v2b_action: V2bAction::None,
        }
    }
}

/// Thread-local accumulation of builder + restore attributions for one request
/// build.
#[derive(Default)]
pub(crate) struct SigPlan {
    entries: HashMap<String, SigSourceEntry>,
}

impl SigPlan {
    fn record_builder(&mut self, tool_id: &str, source: SigSource, fallback_marked: bool) {
        if tool_id.is_empty() {
            return;
        }
        // Never let a builder record clobber an authoritative per-turn store
        // record (last writer wins in favor of the restore path).
        if let Some(existing) = self.entries.get(tool_id) {
            if existing.source == SigSource::PerTurnStore {
                return;
            }
        }
        self.entries
            .insert(tool_id.to_string(), SigSourceEntry::builder(source, fallback_marked));
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
        self.entries.insert(
            tool_id.to_string(),
            SigSourceEntry {
                source: SigSource::PerTurnStore,
                restore_phase: phase,
                store_record_hash: Some(rec_hash.to_string()),
                fallback_marked,
                session_lookup: "none",
                session_message_count: None,
                v2b_action: V2bAction::None,
            },
        );
    }

    fn record_v2b_action(&mut self, tool_id: &str, action: V2bAction) {
        if tool_id.is_empty() {
            return;
        }
        if let Some(entry) = self.entries.get_mut(tool_id) {
            entry.v2b_action = action;
        }
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
            plan.record_builder(tool_id, source, fallback_marked);
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
/// functionCall. Cheap no-op when diagnostics are disabled, no plan is active,
/// or the tool id has no builder attribution.
pub(crate) fn record_v2b_action(tool_id: &str, action: V2bAction) {
    if !is_enabled() {
        return;
    }
    ACTIVE.with(|c| {
        if let Some(plan) = c.borrow_mut().as_mut() {
            plan.record_v2b_action(tool_id, action);
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
    if plan.entries.is_empty() {
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
#[derive(Clone)]
pub(crate) struct SigSourceObservation {
    pub content_index: usize,
    pub part_index: usize,
    /// Salted hash of the raw tool id.
    pub tool_id_hash: String,
    pub entry: SigSourceEntry,
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
                if let Some(entry) = plan.entries.get(id) {
                    out.push(SigSourceObservation {
                        content_index: ci,
                        part_index: pi,
                        tool_id_hash: short_hash(&format!("toolid|{id}")),
                        entry: entry.clone(),
                    });
                }
            }
        }
    }
    out
}

/// EXACT emitted line format. Only salted hashes / enums / counts appear.
pub(crate) fn format_line(obs: &SigSourceObservation, trajectory: &str, seq: u64) -> String {
    format!(
        "[CACHE-SIG-SOURCE] trajectory={} obs_seq={} content_index={} part_index={} sig_source={} tool_id_hash={} store_record_hash={} restore_phase={} session_lookup={} session_message_count={} fallback_marked={} v2b_action={}",
        trajectory,
        seq,
        obs.content_index,
        obs.part_index,
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
            entry,
        }
    }

    #[test]
    fn d3_unit_format_line_fields_present() {
        set_enabled_for_tests(true);
        let entry = SigSourceEntry::builder(SigSource::ToolCache, false);
        let line = format_line(&obs(entry), "traj123", 7);
        assert!(line.starts_with("[CACHE-SIG-SOURCE] "));
        assert!(line.contains("trajectory=traj123"));
        assert!(line.contains("obs_seq=7"));
        assert!(line.contains("content_index=1"));
        assert!(line.contains("part_index=2"));
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
        let mut plan = SigPlan::default();
        plan.record_builder("call_1", SigSource::SessionFallback, true);
        assert_eq!(plan.entries.get("call_1").unwrap().source, SigSource::SessionFallback);
        plan.record_restore("call_1", RestorePhase::P1ToolId, "hash123", true);
        let e = plan.entries.get("call_1").unwrap();
        assert_eq!(e.source, SigSource::PerTurnStore);
        assert_eq!(e.restore_phase, RestorePhase::P1ToolId);
        assert_eq!(e.store_record_hash.as_deref(), Some("hash123"));
        // A later builder record must not clobber the restore.
        plan.record_builder("call_1", SigSource::ToolCache, false);
        assert_eq!(plan.entries.get("call_1").unwrap().source, SigSource::PerTurnStore);
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
}
