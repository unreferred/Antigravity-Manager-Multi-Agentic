// [CACHE-DIAG] OBSERVABILITY ONLY.
//
// This module performs read-only inspection of the exact bytes that are sent to
// the upstream Gemini endpoint. It NEVER mutates the request body or the
// serialized bytes, and it NEVER implements any cache optimization.
//
// ============================================================================
// Diagnostic trajectory / grouping design
// ============================================================================
// There are two distinct identities:
//
//   (A) trajectory identity — a stable, salted hash used as the store key so
//       that successive turns of the SAME conversation are compared with each
//       other. It is derived from the conversation content already present in
//       the request (first user text + system instruction + tool names), using
//       the repository's existing `SessionManager::extract_gemini_session_id`
//       fingerprint logic. It deliberately does NOT include the observed
//       account id or the upstream `sessionId`.
//   (B) observed account / session identity — the account id and the upstream
//       `request.sessionId`, each stored as a salted hash. Changes to these are
//       surfaced as `account_changed` / `session_changed` events rather than
//       creating a new stream.
//
// Because the trajectory is content-derived, an upstream `sessionId` bump
// (FIX session-1M) or an account rotation stays in the same trajectory, which
// makes `SESSION_ID_CHANGE` reachable and avoids splitting account rotations
// into fresh streams.
//
// If no user text anchor exists (e.g. image-only traffic) the trajectory falls
// back to an environment fingerprint (project/model/userAgent/system/tools).
// That is a *grouping heuristic*, explicitly labelled
// `trajectory_source=environment_fallback`; it is not a claimed conversation
// id. Missing `sessionId` therefore never collapses traffic into a single
// "none" stream.
//
// Concurrency: the store is a `DashMap`; the per-trajectory entry lock is held
// across compare-and-update, and the observation sequence is assigned while
// that lock is held. Ordering therefore means "serialization arrival order"
// (`ordering=arrival`), NOT causal conversation order. Parallel siblings of the
// same trajectory are serialized, so an N -> N+1 claim is never inferred from
// concurrent calls.
//
// ============================================================================
// Classification precedence
// ============================================================================
// When several semantic regions differ at once the classification is chosen by
// an explicit protocol-aware precedence list, NOT by serde map insertion order.
// Rationale for the order:
//   * `project` / `model` / `userAgent` are whole-cache-key selectors: a change
//     invalidates the entire upstream cache key, so they outrank anything
//     request-internal.
//   * `request.*` regions then follow their production serialization order
//     (systemInstruction -> tools -> toolConfig -> generationConfig ->
//     sessionId), which is the order the cached prefix is built in.
//   * Historical content instability (truncation, edits, signatures) follows,
//     because it invalidates the cached prefix part-way through.
//   * The volatile top-level `requestId` is last, and `UNKNOWN` is only reached
//     for a structural divergence no rule recognises.
// The precedence list (earliest cache-relevant instability first):
//
//   1. ACCOUNT_PROJECT_CHANGE          project
//   2. MODEL_CHANGE                    model
//   3. USER_AGENT_CHANGE               userAgent
//   4. SYSTEM_INSTRUCTION_CHANGE       request.systemInstruction
//   5. TOOL_DEFINITION_CHANGE          request.tools / toolConfig / tool_config
//   6. GENERATION_CONFIG_CHANGE        request.generationConfig / generation_config
//   7. SESSION_ID_CHANGE               request.sessionId / sessionId
//   8. HISTORICAL_TOOL_ARGS_TRUNCATION exact args sentinel transition
//   9. HISTORICAL_TOOL_OUTPUT_TRUNCATION exact output sentinel transition
//  10. HISTORICAL_TOOL_ARGS_CHANGE    non-sentinel historical args edit
//  11. HISTORICAL_TOOL_OUTPUT_CHANGE  non-sentinel historical output edit
//  12. THINKING_SIGNATURE_CHANGE      thoughtSignature / thought_signature
//  13. HISTORICAL_THOUGHT_CHANGE      historical thought text
//  14. REQUEST_ID_ONLY                only the volatile top-level requestId
//  15. UNKNOWN                        first structural divergence (sanitized)
//
// CONTENT_APPEND_ONLY is checked before the list: a pure trailing append of
// `request.contents` (tolerating the volatile top-level `requestId`) is the
// normal, expected case and is reported as such.
//
// Privacy: only lengths, hashes, offsets, sanitized JSON paths and
// classifications are emitted. Raw identifiers, prompts, tool outputs and
// source code are never logged.
//
// ============================================================================
// Memory bound
// ============================================================================
// Only the exact serialized bytes are retained per trajectory (the parsed
// `Value` is produced lazily from those bytes and only when a semantic
// classification is actually required; the current value is borrowed, never
// cloned). Worst-case retained serialized bytes is bounded by
// `MAX_TOTAL_RETAINED_BYTES` (128 MiB) and additionally by
// `MAX_SESSIONS * MAX_RETAINED_BYTES`.

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use rand::RngCore;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::proxy::session_manager::{sanitize_user_text_for_fingerprint, SessionManager};
use crate::proxy::thinking_store::SENTINEL_SIGNATURE;

/// Maximum number of trajectories retained in the previous-snapshot store.
/// On overflow the entry with the smallest sequence is evicted.
pub const MAX_SESSIONS: usize = 64;

/// Maximum serialized body size (bytes) that will be retained for diffing.
/// Requests larger than this are not stored (the current call is still
/// diagnosed against any existing previous snapshot).
pub const MAX_RETAINED_BYTES: usize = 4 * 1024 * 1024;

/// Global ceiling on retained serialized bytes across all trajectories.
/// Eviction is oldest-first until both this and `MAX_SESSIONS` hold.
pub const MAX_TOTAL_RETAINED_BYTES: usize = 128 * 1024 * 1024;

/// Approximate bytes-per-token divisor used to convert an LCP byte count into
/// an estimated token count. This is a coarse approximation only (English/code
/// text averages roughly 4 bytes/token); it is intentionally NOT a real
/// tokenizer.
pub const BYTES_PER_TOKEN_APPROX: usize = 4;

/// Minimum byte length production requires before treating a
/// `thoughtSignature` string as a real provider/cached signature.
///
/// Mirrors `thinking_store::is_real_signature` and `modules::proxy_db`
/// (`MIN_SIGNATURE_LENGTH` / `MIN_REAL_SIGNATURE`): a signature is "real" only
/// when it is at least this long and is not the sentinel. Diagnostic-only;
/// never used to mutate or restore signatures.
const MIN_REAL_SIGNATURE_LEN: usize = 50;

/// Exact production sentinel for truncated tool-call arguments
/// (`mappers/openai/request.rs`). The value is the string stored under
/// `_truncated` inside the `functionCall.args` object.
const ARGS_TRUNCATION_TEXT: &str = "Arguments truncated to save context window.";

/// Exact production prefix for truncated tool output
/// (`mappers/openai/request.rs`): `[Tool output truncated to save context.
/// Original length: N]`.
const TOOL_OUTPUT_TRUNCATION_PREFIX: &str =
    "[Tool output truncated to save context. Original length: ";

/// Known protocol field-name segments that are safe to echo in a JSON path.
/// Any other (user-controlled) key must be hashed before logging.
const KNOWN_PATH_SEGMENTS: &[&str] = &[
    "project",
    "request",
    "requestId",
    "sessionId",
    "session_id",
    "model",
    "userAgent",
    "requestType",
    "enabledCreditTypes",
    "systemInstruction",
    "system_instruction",
    "tools",
    "toolConfig",
    "tool_config",
    "functionDeclarations",
    "function_declarations",
    "generationConfig",
    "generation_config",
    "safetySettings",
    "safety_settings",
    "contents",
    "role",
    "parts",
    "text",
    "thought",
    "thoughtSignature",
    "thought_signature",
    "functionCall",
    "function_call",
    "functionResponse",
    "function_response",
    "name",
    "args",
    "id",
    "response",
    "result",
    "inlineData",
    "inline_data",
    "mimeType",
    "mime_type",
    "data",
    "cachedContent",
    "cached_content",
];

/// A retained previous request snapshot for one diagnostic trajectory.
///
/// Bounds: at most `MAX_SESSIONS` snapshots and at most
/// `MAX_TOTAL_RETAINED_BYTES` serialized bytes are retained globally; each
/// snapshot holds at most `MAX_RETAINED_BYTES`.
pub struct Snapshot {
    /// The exact serialized bytes observed on the previous call.
    pub bytes: Vec<u8>,
    /// SHA-256 hex of `bytes` (full 64 hex chars), used for dedupe.
    pub sha256_hex: String,
    /// Monotonic write sequence, used for oldest-first eviction.
    pub seq: u64,
    /// Salted hash of the account id observed on the previous call.
    pub account_key: String,
    /// Salted hash of the upstream `sessionId` observed on the previous call.
    pub session_key: Option<String>,
}

struct Store {
    sessions: DashMap<String, Snapshot>,
    seq: AtomicU64,
    total_bytes: AtomicUsize,
}

fn store() -> &'static Store {
    static STORE: OnceLock<Store> = OnceLock::new();
    STORE.get_or_init(|| Store {
        sessions: DashMap::new(),
        seq: AtomicU64::new(0),
        total_bytes: AtomicUsize::new(0),
    })
}

fn effective_total_cap() -> usize {
    #[cfg(test)]
    {
        let v = TEST_TOTAL_CAP_OVERRIDE.load(Ordering::Relaxed);
        if v > 0 {
            return v;
        }
    }
    MAX_TOTAL_RETAINED_BYTES
}

/// Test-only override for enablement so tests are deterministic regardless of
/// process-env mutation or test execution order/parallelism.
#[cfg(test)]
use std::sync::atomic::AtomicBool;
#[cfg(test)]
static TEST_ENABLED_OVERRIDE: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static TEST_TOTAL_CAP_OVERRIDE: AtomicUsize = AtomicUsize::new(0);

/// Returns true when `ABV_CACHE_DIAGNOSTICS` == "1".
///
/// The env read is cached in a `OnceLock` for cheap repeated calls. In test
/// builds the cached value is bypassed when the test override is active.
pub fn is_enabled() -> bool {
    #[cfg(test)]
    {
        if TEST_ENABLED_OVERRIDE.load(Ordering::Relaxed) {
            return true;
        }
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ABV_CACHE_DIAGNOSTICS").as_deref() == Ok("1"))
}

/// Test helper: force-enable/disable diagnostics for the current test process.
#[cfg(test)]
pub(crate) fn set_enabled_for_tests(enabled: bool) {
    TEST_ENABLED_OVERRIDE.store(enabled, Ordering::Relaxed);
}

/// Test helper: set a small global retained-bytes cap.
#[cfg(test)]
pub(crate) fn set_total_cap_for_tests(cap: usize) {
    TEST_TOTAL_CAP_OVERRIDE.store(cap, Ordering::Relaxed);
}

/// Number of retained trajectories. Test/diagnostic helper.
#[allow(dead_code)]
pub fn retained_session_count() -> usize {
    store().sessions.len()
}

/// Total retained serialized bytes. Test/diagnostic helper.
#[allow(dead_code)]
pub fn retained_bytes() -> usize {
    store().total_bytes.load(Ordering::Relaxed)
}

/// Clears all retained snapshots. Test-only.
#[cfg(test)]
pub(crate) fn reset_for_tests() {
    store().sessions.clear();
    store().seq.store(0, Ordering::Relaxed);
    store().total_bytes.store(0, Ordering::Relaxed);
}

/// Per-process random salt. Same identifier correlates within one process
/// lifetime; hashes are not stable across restarts.
fn process_salt() -> &'static [u8; 32] {
    static SALT: OnceLock<[u8; 32]> = OnceLock::new();
    SALT.get_or_init(|| {
        let mut salt = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut salt);
        salt
    })
}

/// First 16 hex chars of salted SHA-256. Used for hashed identifiers and
/// dynamic path segments only.
pub fn short_hash(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(process_salt());
    hasher.update(s.as_bytes());
    let hex = format!("{:x}", hasher.finalize());
    hex.chars().take(16).collect()
}

fn full_sha256_hex(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

/// Longest common prefix length in bytes (exact byte comparison). Operates on
/// raw bytes, so it may stop in the middle of a UTF-8 sequence.
pub fn longest_common_prefix(a: &[u8], b: &[u8]) -> usize {
    let mut i = 0;
    let max = a.len().min(b.len());
    while i < max && a[i] == b[i] {
        i += 1;
    }
    i
}

/// Approximate token count for an LCP byte count.
pub fn estimate_lcp_tokens(lcp_bytes: usize) -> usize {
    lcp_bytes / BYTES_PER_TOKEN_APPROX
}

/// LCP as a percentage of the shorter body, with 2 decimals. 0.0 when either
/// body is empty.
pub fn lcp_percent(lcp_bytes: usize, current_len: usize, previous_len: usize) -> f64 {
    let denom = current_len.min(previous_len);
    if denom == 0 {
        return 0.0;
    }
    let pct = (lcp_bytes as f64) * 100.0 / (denom as f64);
    (pct * 100.0).round() / 100.0
}

fn value_to_key_fragment(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

/// Salted hash of the observed account id.
fn account_key(account_id: Option<&str>) -> String {
    short_hash(&format!("acct={}", account_id.unwrap_or("none")))
}

/// Salted hash of the observed upstream session identity, if present.
///
/// `None` means "no sessionId in this request" and is reported explicitly; it
/// is never conflated with another request's missing identity.
fn observed_session_key(body: &Value) -> Option<String> {
    let inner = body.get("request").unwrap_or(body);
    let v = inner.get("sessionId").or_else(|| body.get("sessionId"))?;
    Some(short_hash(&format!("sess={}", value_to_key_fragment(v))))
}

/// The diagnostic trajectory identity plus how it was derived.
pub(crate) struct Trajectory {
    pub id: String,
    pub source: &'static str,
}

/// True when the request has a usable first-user-text anchor for a stable
/// content-derived conversation fingerprint.
fn has_conversation_anchor(inner: &Value) -> bool {
    inner
        .get("contents")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter().any(|content| {
                content.get("role").and_then(|r| r.as_str()) == Some("user")
                    && content
                        .get("parts")
                        .and_then(|p| p.as_array())
                        .map(|parts| {
                            parts.iter().any(|part| {
                                part.get("text")
                                    .and_then(|t| t.as_str())
                                    .map(|t| sanitize_user_text_for_fingerprint(t).len() >= 2)
                                    .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

/// Environment fallback fingerprint (no conversation anchor available).
/// Explicitly labelled; not a claimed conversation id.
fn env_fingerprint(body: &Value, inner: &Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(process_salt());
    hasher.update(b"env");
    for p in ["project", "model", "userAgent", "requestType"] {
        hasher.update(
            body.get(p)
                .map(|v| v.to_string())
                .unwrap_or_default()
                .as_bytes(),
        );
        hasher.update([0u8]);
    }
    for p in ["systemInstruction", "system_instruction", "tools"] {
        hasher.update(
            inner
                .get(p)
                .map(|v| v.to_string())
                .unwrap_or_default()
                .as_bytes(),
        );
        hasher.update([0u8]);
    }
    format!("{:x}", hasher.finalize())
}

/// Derives the diagnostic trajectory from the most stable available
/// conversation identity already present in the request.
fn derive_trajectory(body: &Value) -> Trajectory {
    let inner = body.get("request").unwrap_or(body);
    let anchored = has_conversation_anchor(inner);
    let seed = if anchored {
        let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("");
        SessionManager::extract_gemini_session_id(inner, model)
    } else {
        env_fingerprint(body, inner)
    };
    Trajectory {
        id: short_hash(&format!("traj|{seed}")),
        source: if anchored {
            "content_seed"
        } else {
            "environment_fallback"
        },
    }
}

/// Path string representation of a container step.
fn join_path(path: &str, step: &str) -> String {
    if path.is_empty() {
        step.to_string()
    } else {
        format!("{path}.{step}")
    }
}

fn join_index(path: &str, idx: usize) -> String {
    format!("{path}[{idx}]")
}

/// Walks two `Value`s and returns the first differing JSON path, or `None`
/// when the values are structurally identical.
///
/// Objects are walked using the union of keys: previous keys first, then any
/// extra keys present only in current. Arrays are compared index by index up to
/// the shorter length; a length mismatch reports the first missing index.
pub fn first_divergence_path(prev: &Value, curr: &Value) -> Option<String> {
    first_divergence_path_inner("", prev, curr)
}

fn first_divergence_path_inner(path: &str, prev: &Value, curr: &Value) -> Option<String> {
    match (prev, curr) {
        (Value::Object(p), Value::Object(c)) => {
            for (k, pv) in p.iter() {
                match c.get(k) {
                    Some(cv) => {
                        if let Some(p) = first_divergence_path_inner(&join_path(path, k), pv, cv) {
                            return Some(p);
                        }
                    }
                    None => return Some(join_path(path, k)),
                }
            }
            for k in c.keys() {
                if !p.contains_key(k) {
                    return Some(join_path(path, k));
                }
            }
            None
        }
        (Value::Array(p), Value::Array(c)) => {
            let min = p.len().min(c.len());
            for i in 0..min {
                if let Some(found) = first_divergence_path_inner(&join_index(path, i), &p[i], &c[i])
                {
                    return Some(found);
                }
            }
            if p.len() != c.len() {
                return Some(join_index(path, min));
            }
            None
        }
        (Value::Null, Value::Null) => None,
        (p, c) => {
            if p == c {
                None
            } else {
                Some(path.to_string())
            }
        }
    }
}

/// True when `path` is exactly an ignored path or lies beneath one.
fn path_is_ignored(path: &str, ignored: &[&str]) -> bool {
    ignored.iter().any(|ig| {
        *ig == path || path.starts_with(&format!("{ig}.")) || path.starts_with(&format!("{ig}["))
    })
}

/// Deep structural equality with an explicit set of ignored paths. Only the
/// listed paths (and their descendants) are ignored; a user payload key that
/// happens to be named `requestId` or `contents` is NOT ignored unless it sits
/// at exactly one of the listed protocol paths.
fn equal_except(a: &Value, b: &Value, path: &str, ignored: &[&str]) -> bool {
    if path_is_ignored(path, ignored) {
        return true;
    }
    match (a, b) {
        (Value::Object(am), Value::Object(bm)) => {
            for (k, va) in am.iter() {
                let p = join_path(path, k);
                if path_is_ignored(&p, ignored) {
                    continue;
                }
                match bm.get(k) {
                    Some(vb) => {
                        if !equal_except(va, vb, &p, ignored) {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
            for k in bm.keys() {
                let p = join_path(path, k);
                if path_is_ignored(&p, ignored) {
                    continue;
                }
                if !am.contains_key(k) {
                    return false;
                }
            }
            true
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return false;
            }
            x.iter()
                .zip(y.iter())
                .enumerate()
                .all(|(i, (u, v))| equal_except(u, v, &join_index(path, i), ignored))
        }
        _ => a == b,
    }
}

fn contents_arrays<'a>(prev: &'a Value, curr: &'a Value) -> (&'a [Value], &'a [Value]) {
    let p = prev
        .get("request")
        .and_then(|r| r.get("contents"))
        .and_then(|c| c.as_array())
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let c = curr
        .get("request")
        .and_then(|r| r.get("contents"))
        .and_then(|c| c.as_array())
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    (p, c)
}

/// Resolves a dot-separated path (no array indices).
fn dotted_get<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = v;
    for seg in path.split('.') {
        cur = cur.get(seg)?;
    }
    Some(cur)
}

/// Returns a precise divergence path within the region rooted at `path`, if the
/// region differs at all.
fn region_diff_path(prev: &Value, curr: &Value, path: &str) -> Option<String> {
    let a = dotted_get(prev, path);
    let b = dotted_get(curr, path);
    match (a, b) {
        (None, None) => None,
        (Some(x), Some(y)) => {
            if x == y {
                None
            } else {
                first_divergence_path_inner(path, x, y).or_else(|| Some(path.to_string()))
            }
        }
        _ => Some(path.to_string()),
    }
}

/// True for the exact production tool-argument truncation sentinel object.
fn is_args_truncation_sentinel(v: &Value) -> bool {
    v.as_object()
        .and_then(|m| m.get("_truncated"))
        .and_then(|t| t.as_str())
        .map(|s| s == ARGS_TRUNCATION_TEXT)
        .unwrap_or(false)
}

/// True for the exact production tool-output truncation sentinel string.
/// Validates structure/format, not a broad substring.
fn is_tool_output_truncation_sentinel(v: &Value) -> bool {
    let s = match v.as_str() {
        Some(s) => s,
        None => return false,
    };
    let rest = match s.strip_prefix(TOOL_OUTPUT_TRUNCATION_PREFIX) {
        Some(r) => r,
        None => return false,
    };
    let digits = match rest.strip_suffix(']') {
        Some(d) => d,
        None => return false,
    };
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

fn part_get<'a>(part: &'a Value, camel: &str, snake: &str) -> Option<&'a Value> {
    part.get(camel).or_else(|| part.get(snake))
}

struct ContentDiffs {
    args_trunc: Option<String>,
    args_change: Option<String>,
    output_trunc: Option<String>,
    output_change: Option<String>,
}

/// Scans historical `contents[i]` (i < last index of current contents) for
/// tool-argument and tool-output changes, distinguishing exact sentinel
/// transitions from arbitrary edits.
fn scan_historical_content_diffs(prev: &Value, curr: &Value) -> ContentDiffs {
    let (pv, cv) = contents_arrays(prev, curr);
    let last = cv.len().saturating_sub(1);
    let mut out = ContentDiffs {
        args_trunc: None,
        args_change: None,
        output_trunc: None,
        output_change: None,
    };

    let limit = last.min(pv.len());
    for i in 0..limit {
        let pp = pv[i].get("parts").and_then(|p| p.as_array());
        let cp = cv[i].get("parts").and_then(|p| p.as_array());
        let (pp, cp) = match (pp, cp) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let jmax = pp.len().min(cp.len());
        for j in 0..jmax {
            // functionCall.args
            let pa = part_get(&pp[j], "functionCall", "function_call").and_then(|f| f.get("args"));
            let ca = part_get(&cp[j], "functionCall", "function_call").and_then(|f| f.get("args"));
            if let (Some(pa), Some(ca)) = (pa, ca) {
                if pa != ca {
                    let base = format!("request.contents[{i}].parts[{j}].functionCall.args");
                    if is_args_truncation_sentinel(ca) && !is_args_truncation_sentinel(pa) {
                        out.args_trunc.get_or_insert(base);
                    } else {
                        let precise = first_divergence_path_inner(&base, pa, ca)
                            .unwrap_or_else(|| base.clone());
                        out.args_change.get_or_insert(precise);
                    }
                }
            }

            // functionResponse.response.result
            let pr = part_get(&pp[j], "functionResponse", "function_response")
                .and_then(|f| f.get("response"))
                .and_then(|r| r.get("result"));
            let cr = part_get(&cp[j], "functionResponse", "function_response")
                .and_then(|f| f.get("response"))
                .and_then(|r| r.get("result"));
            if let (Some(pr), Some(cr)) = (pr, cr) {
                if pr != cr {
                    let base = format!(
                        "request.contents[{i}].parts[{j}].functionResponse.response.result"
                    );
                    if is_tool_output_truncation_sentinel(cr)
                        && !is_tool_output_truncation_sentinel(pr)
                    {
                        out.output_trunc.get_or_insert(base);
                    } else {
                        let precise = first_divergence_path_inner(&base, pr, cr)
                            .unwrap_or_else(|| base.clone());
                        out.output_change.get_or_insert(precise);
                    }
                }
            }
        }
    }
    out
}

/// Diagnostic representation class of a single `thoughtSignature` slot.
///
/// Derived from the exact serialized snapshot bytes. No signature value is
/// ever retained or emitted; only the class is used, and it is consumed
/// immediately inside this module.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SignatureKind {
    /// Field absent or explicit JSON `null`.
    Missing,
    /// Exactly the production sentinel `skip_thought_signature_validator`.
    Sentinel,
    /// Non-sentinel value satisfying production's minimum-length predicate.
    Real,
    /// Present but not production-real and not the sentinel (e.g. a short
    /// non-sentinel string, or a non-string JSON value). Cannot be safely
    /// classified as MISSING or REAL.
    Unknown,
}

/// True when `s` is a real signature under production's own predicate:
/// non-sentinel and at least `MIN_REAL_SIGNATURE_LEN` bytes.
fn is_real_signature(s: &str) -> bool {
    s.len() >= MIN_REAL_SIGNATURE_LEN && s != SENTINEL_SIGNATURE
}

/// Classifies a `thoughtSignature` slot without retaining its value.
fn signature_kind(v: Option<&Value>) -> SignatureKind {
    match v {
        None | Some(Value::Null) => SignatureKind::Missing,
        Some(Value::String(s)) if s == SENTINEL_SIGNATURE => SignatureKind::Sentinel,
        Some(Value::String(s)) if is_real_signature(s) => SignatureKind::Real,
        _ => SignatureKind::Unknown,
    }
}

/// Maps a previous/current representation pair to the transition enum.
///
/// `SENTINEL_TO_SENTINEL_DIFFERENT` is deliberately absent: the sentinel is a
/// single exact value, so two sentinel slots are always byte-equal and can
/// never be a detected divergence. Every pair that is not a representable
/// transition resolves to `UNKNOWN_SIGNATURE_TRANSITION`.
fn signature_transition_enum(prev: SignatureKind, curr: SignatureKind) -> &'static str {
    use SignatureKind::*;
    match (prev, curr) {
        (Missing, Real) => "MISSING_TO_REAL",
        (Missing, Sentinel) => "MISSING_TO_SENTINEL",
        (Sentinel, Real) => "SENTINEL_TO_REAL",
        (Sentinel, Missing) => "SENTINEL_TO_MISSING",
        (Real, Real) => "REAL_TO_REAL_DIFFERENT",
        (Real, Sentinel) => "REAL_TO_SENTINEL",
        (Real, Missing) => "REAL_TO_MISSING",
        _ => "UNKNOWN_SIGNATURE_TRANSITION",
    }
}

/// First path where a `thoughtSignature` / `thought_signature` value differs,
/// including signatures nested under `functionCall`, together with the
/// representation transition observed at that exact path.
///
/// The transition is computed from the same previous/current values that
/// produced the divergence, i.e. the exact serialized snapshots, and never
/// from any cache state.
fn signature_divergence(prev: &Value, curr: &Value) -> Option<(String, &'static str)> {
    let (pv, cv) = contents_arrays(prev, curr);
    let limit = pv.len().min(cv.len());
    for i in 0..limit {
        let pp = pv[i].get("parts").and_then(|p| p.as_array());
        let cp = cv[i].get("parts").and_then(|p| p.as_array());
        let (pp, cp) = match (pp, cp) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let jmax = pp.len().min(cp.len());
        for j in 0..jmax {
            for key in ["thoughtSignature", "thought_signature"] {
                let a = pp[j].get(key);
                let b = cp[j].get(key);
                if a != b {
                    return Some((
                        format!("request.contents[{i}].parts[{j}].{key}"),
                        signature_transition_enum(signature_kind(a), signature_kind(b)),
                    ));
                }
            }
            let a = part_get(&pp[j], "functionCall", "function_call")
                .and_then(|f| f.get("thoughtSignature"));
            let b = part_get(&cp[j], "functionCall", "function_call")
                .and_then(|f| f.get("thoughtSignature"));
            if a != b {
                return Some((
                    format!("request.contents[{i}].parts[{j}].functionCall.thoughtSignature"),
                    signature_transition_enum(signature_kind(a), signature_kind(b)),
                ));
            }
        }
    }
    None
}

/// First `thoughtSignature` divergence path, if any.
fn signature_diff_path(prev: &Value, curr: &Value) -> Option<String> {
    signature_divergence(prev, curr).map(|(path, _)| path)
}

/// Transition enum for a signature divergence, or `none` when the two
/// snapshots have no divergent signature slot.
pub(crate) fn signature_transition(prev: &Value, curr: &Value) -> &'static str {
    signature_divergence(prev, curr)
        .map(|(_, t)| t)
        .unwrap_or("none")
}

/// First differing `text` in a historical content part flagged `thought: true`.
fn historical_thought_change_path(prev: &Value, curr: &Value) -> Option<String> {
    let (pv, cv) = contents_arrays(prev, curr);
    let last = cv.len().saturating_sub(1);
    let limit = last.min(pv.len());
    for i in 0..limit {
        let pp = pv[i].get("parts").and_then(|p| p.as_array());
        let cp = cv[i].get("parts").and_then(|p| p.as_array());
        let (pp, cp) = match (pp, cp) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let jmax = pp.len().min(cp.len());
        for j in 0..jmax {
            let thought = pp[j]
                .get("thought")
                .and_then(|t| t.as_bool())
                .unwrap_or(false)
                || cp[j]
                    .get("thought")
                    .and_then(|t| t.as_bool())
                    .unwrap_or(false);
            if thought && pp[j].get("text") != cp[j].get("text") {
                return Some(format!("request.contents[{i}].parts[{j}].text"));
            }
        }
    }
    None
}

/// True when `curr.request.contents` is `prev.request.contents` plus a trailing
/// append, tolerating only the volatile top-level `requestId`.
fn is_content_append_only(prev: &Value, curr: &Value) -> Option<String> {
    let (pc, cc) = contents_arrays(prev, curr);
    if pc.len() >= cc.len() {
        return None;
    }
    if pc != &cc[..pc.len()] {
        return None;
    }
    if !equal_except(prev, curr, "", &["requestId", "request.contents"]) {
        return None;
    }
    Some(format!("request.contents[{}]", pc.len()))
}

/// Classifies the semantic divergence between two request bodies.
///
/// Returns `(path, classification)`. `path` is empty and classification is
/// `IDENTICAL` when the two bodies are structurally equal. See the module
/// documentation for the explicit precedence order.
pub(crate) fn classify_diff(prev: &Value, curr: &Value) -> (String, &'static str) {
    if prev == curr {
        return (String::new(), "IDENTICAL");
    }

    // Normal expected case: pure trailing append (plus volatile requestId).
    if let Some(path) = is_content_append_only(prev, curr) {
        return (path, "CONTENT_APPEND_ONLY");
    }

    // Explicit protocol-aware precedence (earliest cache-relevant instability
    // first). Never rely on serde map insertion order here.
    const REGIONS: &[(&str, &str)] = &[
        ("project", "ACCOUNT_PROJECT_CHANGE"),
        ("model", "MODEL_CHANGE"),
        ("userAgent", "USER_AGENT_CHANGE"),
        ("request.systemInstruction", "SYSTEM_INSTRUCTION_CHANGE"),
        ("request.tools", "TOOL_DEFINITION_CHANGE"),
        ("request.toolConfig", "TOOL_DEFINITION_CHANGE"),
        ("request.tool_config", "TOOL_DEFINITION_CHANGE"),
        ("request.generationConfig", "GENERATION_CONFIG_CHANGE"),
        ("request.generation_config", "GENERATION_CONFIG_CHANGE"),
        ("request.sessionId", "SESSION_ID_CHANGE"),
        ("request.session_id", "SESSION_ID_CHANGE"),
        ("sessionId", "SESSION_ID_CHANGE"),
        ("session_id", "SESSION_ID_CHANGE"),
    ];
    for (path, class) in REGIONS {
        if let Some(found) = region_diff_path(prev, curr, path) {
            return (found, class);
        }
    }

    let diffs = scan_historical_content_diffs(prev, curr);
    if let Some(p) = diffs.args_trunc {
        return (p, "HISTORICAL_TOOL_ARGS_TRUNCATION");
    }
    if let Some(p) = diffs.output_trunc {
        return (p, "HISTORICAL_TOOL_OUTPUT_TRUNCATION");
    }
    if let Some(p) = diffs.args_change {
        return (p, "HISTORICAL_TOOL_ARGS_CHANGE");
    }
    if let Some(p) = diffs.output_change {
        return (p, "HISTORICAL_TOOL_OUTPUT_CHANGE");
    }
    if let Some(p) = signature_diff_path(prev, curr) {
        return (p, "THINKING_SIGNATURE_CHANGE");
    }
    if let Some(p) = historical_thought_change_path(prev, curr) {
        return (p, "HISTORICAL_THOUGHT_CHANGE");
    }

    // Volatile top-level requestId is the only remaining difference.
    if equal_except(prev, curr, "", &["requestId"]) {
        return ("requestId".to_string(), "REQUEST_ID_ONLY");
    }

    let path = first_divergence_path(prev, curr).unwrap_or_default();
    (path, "UNKNOWN")
}

/// Renders one path segment: numeric indices are echoed, known protocol field
/// names are echoed, everything else is replaced by a salted hash.
fn render_path_segment(seg: &str, in_bracket: bool) -> String {
    if seg.is_empty() {
        return String::new();
    }
    if in_bracket && seg.bytes().all(|b| b.is_ascii_digit()) {
        return seg.to_string();
    }
    if KNOWN_PATH_SEGMENTS.contains(&seg) {
        return seg.to_string();
    }
    format!("<key:{}>", short_hash(&format!("path|{seg}")))
}

/// Sanitizes a JSON path for logging: fixed protocol field names and array
/// indices are shown; arbitrary/user-controlled key segments are hashed.
fn sanitize_path(path: &str) -> String {
    let mut out = String::new();
    let mut token = String::new();
    let mut in_bracket = false;
    for ch in path.chars() {
        match ch {
            '.' => {
                if !token.is_empty() {
                    out.push_str(&render_path_segment(&token, false));
                    token.clear();
                }
                out.push('.');
            }
            '[' => {
                if !token.is_empty() {
                    out.push_str(&render_path_segment(&token, false));
                    token.clear();
                }
                in_bracket = true;
                out.push('[');
            }
            ']' => {
                if !token.is_empty() {
                    out.push_str(&render_path_segment(&token, in_bracket));
                    token.clear();
                }
                in_bracket = false;
                out.push(']');
            }
            _ => token.push(ch),
        }
    }
    if !token.is_empty() {
        out.push_str(&render_path_segment(&token, false));
    }
    out
}

/// Runs the diagnostic core with an explicit previous snapshot. Pure with
/// respect to global state; exposed for deterministic unit testing.
pub(crate) fn observe_inner(
    prev: Option<&Snapshot>,
    serialized: &[u8],
    body: &Value,
    method: &str,
    account_id: Option<&str>,
    trajectory: &Trajectory,
    observation_seq: u64,
) -> DiagnosticOutcome {
    let current_len = serialized.len();
    let current_sha = full_sha256_hex(serialized);
    let account = account_key(account_id);
    let session = observed_session_key(body);

    let mut outcome = DiagnosticOutcome {
        trajectory_key: trajectory.id.clone(),
        trajectory_source: trajectory.source,
        account_key: account.clone(),
        account_changed: false,
        session_key: session.clone(),
        session_changed: false,
        method: method.to_string(),
        observation_seq,
        current_bytes: current_len,
        current_sha256: current_sha.clone(),
        previous_bytes: None,
        lcp_bytes: None,
        lcp_percent: None,
        estimated_lcp_tokens: None,
        first_diff_offset: None,
        first_divergence: None,
        classification: "NEW_SESSION",
        signature_transition: "none",
        ordering: "arrival",
    };

    if let Some(p) = prev {
        outcome.previous_bytes = Some(p.bytes.len());
        outcome.account_changed = p.account_key != account;
        outcome.session_changed = p.session_key != session;

        if p.sha256_hex == current_sha {
            // Idempotent: repeated endpoint attempts with identical bytes.
            outcome.classification = "DEDUPED";
            outcome.lcp_bytes = Some(current_len);
            outcome.lcp_percent = Some(lcp_percent(current_len, current_len, p.bytes.len()));
            outcome.estimated_lcp_tokens = Some(estimate_lcp_tokens(current_len));
            outcome.first_diff_offset = Some(current_len);
            return outcome;
        }

        let lcp = longest_common_prefix(serialized, &p.bytes);
        // Parse the previous bytes lazily: only needed for semantic
        // classification, and never cloned from a retained `Value`.
        let (path, class) = match serde_json::from_slice::<Value>(&p.bytes) {
            Ok(prev_value) => {
                let (path, class) = classify_diff(&prev_value, body);
                // Only a signature-classified divergence gets a transition; a
                // signature slot that happens to differ alongside a
                // higher-precedence change is not surfaced as a signature event.
                if class == "THINKING_SIGNATURE_CHANGE" {
                    outcome.signature_transition = signature_transition(&prev_value, body);
                }
                (path, class)
            }
            Err(_) => (String::new(), "PREVIOUS_PARSE_ERROR"),
        };
        outcome.lcp_bytes = Some(lcp);
        outcome.lcp_percent = Some(lcp_percent(lcp, current_len, p.bytes.len()));
        outcome.estimated_lcp_tokens = Some(estimate_lcp_tokens(lcp));
        outcome.first_diff_offset = Some(lcp);
        outcome.first_divergence = Some(path);
        outcome.classification = class;
    }

    outcome
}

/// Structured diagnostic outcome. Contains no raw request content.
pub(crate) struct DiagnosticOutcome {
    pub trajectory_key: String,
    pub trajectory_source: &'static str,
    pub account_key: String,
    pub account_changed: bool,
    pub session_key: Option<String>,
    pub session_changed: bool,
    pub method: String,
    pub observation_seq: u64,
    pub current_bytes: usize,
    #[allow(dead_code)]
    pub current_sha256: String,
    pub previous_bytes: Option<usize>,
    pub lcp_bytes: Option<usize>,
    pub lcp_percent: Option<f64>,
    pub estimated_lcp_tokens: Option<usize>,
    pub first_diff_offset: Option<usize>,
    pub first_divergence: Option<String>,
    pub classification: &'static str,
    /// Representation transition for `THINKING_SIGNATURE_CHANGE`, or `none`
    /// for every other classification / snapshot state. Never value-derived.
    pub signature_transition: &'static str,
    pub ordering: &'static str,
}

impl DiagnosticOutcome {
    fn format_line(&self) -> String {
        let divergence = self
            .first_divergence
            .as_deref()
            .map(sanitize_path)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "none".to_string());
        format!(
            "[CACHE-DIAG] trajectory={} trajectory_source={} ordering={} account={} account_changed={} session={} session_changed={} method={} obs_seq={} current_bytes={} previous_bytes={} lcp_bytes={} lcp_percent={} estimated_lcp_tokens={} first_diff_offset={} first_divergence={} classification={} signature_transition={}",
            self.trajectory_key,
            self.trajectory_source,
            self.ordering,
            self.account_key,
            self.account_changed,
            self.session_key.as_deref().unwrap_or("none"),
            self.session_changed,
            self.method,
            self.observation_seq,
            self.current_bytes,
            self.previous_bytes
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.lcp_bytes
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.lcp_percent
                .map(|v| format!("{v:.2}"))
                .unwrap_or_else(|| "none".to_string()),
            self.estimated_lcp_tokens
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            self.first_diff_offset
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            divergence,
            self.classification,
            self.signature_transition,
        )
    }
}

/// Oldest-first eviction until both the session count and the retained-byte
/// ceiling hold. Called after releasing the per-trajectory entry lock.
fn evict_if_over_cap() {
    let st = store();
    loop {
        let over_count = st.sessions.len() > MAX_SESSIONS;
        let over_bytes = st.total_bytes.load(Ordering::Relaxed) > effective_total_cap();
        if !over_count && !over_bytes {
            break;
        }
        let mut oldest: Option<(String, u64, usize)> = None;
        for e in st.sessions.iter() {
            let s = e.value();
            match &oldest {
                Some((_, sq, _)) if *sq <= s.seq => {}
                _ => oldest = Some((e.key().clone(), s.seq, s.bytes.len())),
            }
        }
        match oldest {
            Some((k, _, len)) => {
                if st.sessions.remove(&k).is_some() {
                    st.total_bytes.fetch_sub(len, Ordering::Relaxed);
                }
            }
            None => break,
        }
    }
}

/// Read-only observation of the exact serialized request bytes.
///
/// When diagnostics are disabled this returns immediately and performs ZERO
/// work (no hashing, no allocation, no store access). It never mutates
/// `serialized` or `body`.
pub fn observe(serialized: &[u8], body: &Value, method: &str, account_id: Option<&str>) {
    if !is_enabled() {
        return;
    }
    let _ = observe_gated(serialized, body, method, account_id);
}

/// Core gated path, separated from the env gate so it is directly testable.
///
/// The per-trajectory entry lock is held across the compare-and-update so that
/// snapshot updates and observation-sequence assignment are atomic per
/// trajectory.
pub(crate) fn observe_gated(
    serialized: &[u8],
    body: &Value,
    method: &str,
    account_id: Option<&str>,
) -> DiagnosticOutcome {
    let trajectory = derive_trajectory(body);
    let current_sha = full_sha256_hex(serialized);
    let account = account_key(account_id);
    let session = observed_session_key(body);
    let st = store();

    let mut grew = false;
    let outcome;

    {
        let entry = st.sessions.entry(trajectory.id.clone());
        match entry {
            Entry::Vacant(v) => {
                let seq = st.seq.fetch_add(1, Ordering::Relaxed);
                outcome =
                    observe_inner(None, serialized, body, method, account_id, &trajectory, seq);
                if serialized.len() <= MAX_RETAINED_BYTES {
                    v.insert(Snapshot {
                        bytes: serialized.to_vec(),
                        sha256_hex: current_sha.clone(),
                        seq,
                        account_key: account.clone(),
                        session_key: session.clone(),
                    });
                    st.total_bytes
                        .fetch_add(serialized.len(), Ordering::Relaxed);
                    grew = true;
                }
            }
            Entry::Occupied(mut o) => {
                let seq = st.seq.fetch_add(1, Ordering::Relaxed);
                let (outcome_tmp, deduped, prev_len) = {
                    let prev = o.get();
                    (
                        observe_inner(
                            Some(prev),
                            serialized,
                            body,
                            method,
                            account_id,
                            &trajectory,
                            seq,
                        ),
                        prev.sha256_hex == current_sha,
                        prev.bytes.len(),
                    )
                };
                outcome = outcome_tmp;
                if !deduped && serialized.len() <= MAX_RETAINED_BYTES {
                    st.total_bytes.fetch_sub(prev_len, Ordering::Relaxed);
                    let _old = std::mem::replace(
                        o.get_mut(),
                        Snapshot {
                            bytes: serialized.to_vec(),
                            sha256_hex: current_sha.clone(),
                            seq,
                            account_key: account.clone(),
                            session_key: session.clone(),
                        },
                    );
                    st.total_bytes
                        .fetch_add(serialized.len(), Ordering::Relaxed);
                    grew = true;
                }
            }
        }
    }

    if grew {
        evict_if_over_cap();
    }

    crate::proxy::signature_source_diagnostics::emit_for_body(
        body,
        &outcome.trajectory_key,
        outcome.observation_seq,
    );
    tracing::info!("{}", outcome.format_line());
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    /// Serializes tests that touch the global snapshot store so assertions on
    /// `retained_session_count()` are deterministic under parallel execution.
    static STORE_LOCK: Mutex<()> = Mutex::new(());

    fn lock_store() -> std::sync::MutexGuard<'static, ()> {
        STORE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn session_body(contents: Value) -> Value {
        json!({
            "project": "proj-1",
            "request": {
                "systemInstruction": { "parts": [{ "text": "sys" }] },
                "tools": [],
                "generationConfig": { "temperature": 1 },
                "sessionId": 12345,
                "contents": contents
            },
            "model": "gemini-2.0-flash",
            "userAgent": "antigravity",
            "requestId": "req-1"
        })
    }

    fn text_content(role: &str, text: &str) -> Value {
        json!({ "role": role, "parts": [{ "text": text }] })
    }

    fn ser(v: &Value) -> Vec<u8> {
        serde_json::to_vec(v).unwrap()
    }

    /// Snapshot that mirrors the account/session identity of `body`.
    fn snap_for(body: &Value) -> Snapshot {
        let bytes = ser(body);
        Snapshot {
            bytes: bytes.clone(),
            sha256_hex: full_sha256_hex(&bytes),
            seq: 1,
            account_key: account_key(Some("acct")),
            session_key: observed_session_key(body),
        }
    }

    fn observe_inner_body(prev: Option<&Snapshot>, body: &Value) -> DiagnosticOutcome {
        let bytes = ser(body);
        let traj = derive_trajectory(body);
        observe_inner(
            prev,
            &bytes,
            body,
            "generateContent",
            Some("acct"),
            &traj,
            0,
        )
    }

    // ------------------------------------------------------------------
    // Baseline behavior
    // ------------------------------------------------------------------

    #[test]
    fn a_identical_request() {
        let body = session_body(json!([text_content("user", "hello")]));
        let bytes = ser(&body);
        let s = snap_for(&body);
        let out = observe_inner_body(Some(&s), &body);
        assert_eq!(out.classification, "DEDUPED");
        assert_eq!(out.lcp_bytes, Some(bytes.len()));
        assert_eq!(out.lcp_percent, Some(100.0));
        assert_eq!(out.first_diff_offset, Some(bytes.len()));
        assert!(!out.account_changed);
        assert!(!out.session_changed);
        assert_eq!(out.ordering, "arrival");
    }

    #[test]
    fn b_append_only_contents() {
        let prev = session_body(json!([text_content("user", "hello")]));
        let mut contents = vec![text_content("user", "hello")];
        contents.push(text_content("model", "world"));
        let curr = session_body(Value::Array(contents));

        let (path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "CONTENT_APPEND_ONLY");
        assert_eq!(path, "request.contents[1]");

        let pb = ser(&prev);
        let cb = ser(&curr);
        let lcp = longest_common_prefix(&pb, &cb);
        let contents_marker = b"\"contents\":[";
        let contents_start = pb
            .windows(contents_marker.len())
            .position(|w| w == contents_marker)
            .unwrap();
        assert!(lcp > contents_start);
        assert!(lcp < pb.len());
    }

    // ------------------------------------------------------------------
    // 1. Truncation classification
    // ------------------------------------------------------------------

    fn args_sentinel_body(content_text: &str) -> Value {
        session_body(json!([
            {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": "do_thing",
                        "args": { "_truncated": ARGS_TRUNCATION_TEXT },
                        "id": "call-1"
                    }
                }]
            },
            text_content("user", content_text)
        ]))
    }

    fn real_args_body(content_text: &str) -> Value {
        session_body(json!([
            {
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": "do_thing",
                        "args": { "big": "a".repeat(500) },
                        "id": "call-1"
                    }
                }]
            },
            text_content("user", content_text)
        ]))
    }

    #[test]
    fn c1_exact_args_truncation_sentinel_positive() {
        let prev = real_args_body("next");
        let curr = args_sentinel_body("next");
        let (path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "HISTORICAL_TOOL_ARGS_TRUNCATION");
        assert_eq!(path, "request.contents[0].parts[0].functionCall.args");
    }

    #[test]
    fn c2_arbitrary_function_call_args_edit_is_not_truncation() {
        let prev = real_args_body("next");
        let mut curr = prev.clone();
        curr["request"]["contents"][0]["parts"][0]["functionCall"]["args"]["big"] = json!("b");
        let (path, class) = classify_diff(&prev, &curr);
        assert_ne!(class, "HISTORICAL_TOOL_ARGS_TRUNCATION");
        assert_eq!(class, "HISTORICAL_TOOL_ARGS_CHANGE");
        assert!(path.contains("functionCall.args"));
    }

    #[test]
    fn c3_argument_growth_is_not_truncation() {
        let mut prev = real_args_body("next");
        prev["request"]["contents"][0]["parts"][0]["functionCall"]["args"] = json!({ "big": "a" });
        let mut curr = prev.clone();
        curr["request"]["contents"][0]["parts"][0]["functionCall"]["args"] =
            json!({ "big": "aaaaaaaaaa" });
        let (_path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "HISTORICAL_TOOL_ARGS_CHANGE");
    }

    #[test]
    fn c4_text_merely_containing_truncated_is_not_truncation() {
        let prev = session_body(json!([
            text_content("user", "ordinary text"),
            text_content("user", "next")
        ]));
        let curr = session_body(json!([
            text_content("user", "text with _truncated marker inside"),
            text_content("user", "next")
        ]));
        let (_path, class) = classify_diff(&prev, &curr);
        assert_ne!(class, "HISTORICAL_TOOL_ARGS_TRUNCATION");
        assert_ne!(class, "HISTORICAL_TOOL_OUTPUT_TRUNCATION");
    }

    #[test]
    fn c5_exact_tool_output_sentinel_positive() {
        let prev = session_body(json!([
            {
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": "do_thing",
                        "response": { "result": "x".repeat(400) },
                        "id": "call-1"
                    }
                }]
            },
            text_content("user", "next")
        ]));
        let curr = session_body(json!([
            {
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": "do_thing",
                        "response": {
                            "result": "[Tool output truncated to save context. Original length: 400]"
                        },
                        "id": "call-1"
                    }
                }]
            },
            text_content("user", "next")
        ]));
        let (path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "HISTORICAL_TOOL_OUTPUT_TRUNCATION");
        assert!(path.ends_with("functionResponse.response.result"));
    }

    #[test]
    fn c6_tool_output_sentinel_decoys_are_not_truncation() {
        let make = |s: &str| {
            session_body(json!([
                {
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": "do_thing",
                            "response": { "result": s },
                            "id": "call-1"
                        }
                    }]
                },
                text_content("user", "next")
            ]))
        };
        let prev = make("real output that is long enough to matter");
        for decoy in [
            "prefix [Tool output truncated to save context. Original length: 12]",
            "[Tool output truncated to save context. Original length: abc]",
            "[Tool output truncated to save context. Original length: 12",
            "[Tool output truncated to save context. Original length:]",
            "Tool output truncated to save context. Original length: 12]",
            "_truncated",
        ] {
            let curr = make(decoy);
            let (_p, class) = classify_diff(&prev, &curr);
            assert_ne!(
                class, "HISTORICAL_TOOL_OUTPUT_TRUNCATION",
                "decoy should not classify as truncation: {decoy}"
            );
        }
    }

    #[test]
    fn c7_real_args_replacing_sentinel_is_change_not_truncation() {
        let prev = args_sentinel_body("next");
        let curr = real_args_body("next");
        let (_p, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "HISTORICAL_TOOL_ARGS_CHANGE");
    }

    // ------------------------------------------------------------------
    // 2. requestId handling
    // ------------------------------------------------------------------

    #[test]
    fn i_request_id_only() {
        let prev = session_body(json!([text_content("user", "hi")]));
        let mut curr = prev.clone();
        curr["requestId"] = json!("req-2");
        let (path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "REQUEST_ID_ONLY");
        assert_eq!(path, "requestId");
    }

    #[test]
    fn i2_request_id_plus_append_is_append_only() {
        let prev = session_body(json!([text_content("user", "hi")]));
        let mut contents = vec![text_content("user", "hi")];
        contents.push(text_content("model", "yo"));
        let mut curr = session_body(Value::Array(contents));
        curr["requestId"] = json!("req-2");
        let (path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "CONTENT_APPEND_ONLY");
        assert_eq!(path, "request.contents[1]");
    }

    #[test]
    fn i3_nested_user_object_request_id_is_not_ignored() {
        let base = |v: &str| {
            session_body(json!([{
                "role": "user",
                "parts": [{ "text": "hi", "userMeta": { "requestId": v } }]
            }]))
        };
        let prev = base("user-owned-1");
        let curr = base("user-owned-2");
        let (path, class) = classify_diff(&prev, &curr);
        assert_ne!(class, "REQUEST_ID_ONLY");
        assert_eq!(class, "UNKNOWN");
        assert!(path.contains("userMeta"));
        assert!(path.contains("requestId"));
    }

    #[test]
    fn i4_nested_user_object_contents_is_not_ignored() {
        let base = |v: &str| {
            session_body(json!([{
                "role": "user",
                "parts": [{ "text": "hi", "userMeta": { "contents": v } }]
            }]))
        };
        let prev = base("user-owned-1");
        let curr = base("user-owned-2");
        let (path, class) = classify_diff(&prev, &curr);
        assert_ne!(class, "CONTENT_APPEND_ONLY");
        assert_eq!(class, "UNKNOWN");
        assert!(path.contains("contents"));
        assert!(path.contains("userMeta"));
    }

    // ------------------------------------------------------------------
    // 3. Classification precedence
    // ------------------------------------------------------------------

    fn appended(base: &Value, extra: Value) -> Value {
        let mut contents = base["request"]["contents"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        contents.push(extra);
        let mut curr = base.clone();
        curr["request"]["contents"] = Value::Array(contents);
        curr
    }

    #[test]
    fn p1_project_beats_contents_append() {
        let prev = session_body(json!([text_content("user", "hi")]));
        let mut curr = appended(&prev, text_content("model", "yo"));
        curr["project"] = json!("proj-2");
        let (_p, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "ACCOUNT_PROJECT_CHANGE");
    }

    #[test]
    fn p2_model_beats_generation_config() {
        let prev = session_body(json!([text_content("user", "hi")]));
        let mut curr = prev.clone();
        curr["model"] = json!("gemini-2.5-pro");
        curr["request"]["generationConfig"]["temperature"] = json!(0.1);
        let (_p, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "MODEL_CHANGE");
    }

    #[test]
    fn p3_session_beats_thought_signature() {
        let prev = session_body(json!([{
            "role": "model",
            "parts": [{ "thought": true, "thoughtSignature": "sig-a" }]
        }]));
        let mut curr = prev.clone();
        curr["request"]["sessionId"] = json!(99999);
        curr["request"]["contents"][0]["parts"][0]["thoughtSignature"] = json!("sig-b");
        let (_p, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "SESSION_ID_CHANGE");
    }

    #[test]
    fn p4_truncation_beats_request_id() {
        let prev = real_args_body("next");
        let mut curr = args_sentinel_body("next");
        curr["requestId"] = json!("req-rotated");
        let (_p, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "HISTORICAL_TOOL_ARGS_TRUNCATION");
    }

    #[test]
    fn p5_tool_schema_beats_contents_append() {
        let tool = |ty: &str| json!([{ "functionDeclarations": [{ "name": "t", "parameters": { "type": ty } }] }]);
        let mut prev = session_body(json!([text_content("user", "hi")]));
        prev["request"]["tools"] = tool("object");
        let mut curr = appended(&prev, text_content("model", "yo"));
        curr["request"]["tools"] = tool("string");
        let (path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "TOOL_DEFINITION_CHANGE");
        assert!(path.starts_with("request.tools"));
    }

    #[test]
    fn g_session_id_change() {
        let prev = session_body(json!([text_content("user", "hi")]));
        let mut curr = prev.clone();
        curr["request"]["sessionId"] = json!(99999);
        let (path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "SESSION_ID_CHANGE");
        assert_eq!(path, "request.sessionId");
    }

    #[test]
    fn h_project_model_user_agent_changes() {
        let base = session_body(json!([text_content("user", "hi")]));

        let mut p = base.clone();
        p["project"] = json!("proj-2");
        assert_eq!(classify_diff(&base, &p).1, "ACCOUNT_PROJECT_CHANGE");

        let mut m = base.clone();
        m["model"] = json!("gemini-2.5-pro");
        assert_eq!(classify_diff(&base, &m).1, "MODEL_CHANGE");

        let mut u = base.clone();
        u["userAgent"] = json!("antigravity/2");
        assert_eq!(classify_diff(&base, &u).1, "USER_AGENT_CHANGE");
    }

    #[test]
    fn e_thinking_signature_change() {
        let prev = session_body(json!([{
            "role": "model",
            "parts": [{ "thought": true, "thoughtSignature": "sig-aaa" }]
        }]));
        let curr = session_body(json!([{
            "role": "model",
            "parts": [{ "thought": true, "thoughtSignature": "sig-bbb" }]
        }]));
        let (path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "THINKING_SIGNATURE_CHANGE");
        assert!(path.contains("thoughtSignature"));
    }

    #[test]
    fn f_historical_thought_change() {
        let prev = session_body(json!([
            { "role": "model", "parts": [{ "thought": true, "text": "old thinking" }] },
            text_content("user", "next")
        ]));
        let curr = session_body(json!([
            { "role": "model", "parts": [{ "thought": true, "text": "new thinking" }] },
            text_content("user", "next")
        ]));
        let (path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "HISTORICAL_THOUGHT_CHANGE");
        assert!(path.contains("contents[0]"));
    }

    // ------------------------------------------------------------------
    // 2b. Thinking-signature representation transitions
    // ------------------------------------------------------------------

    /// Two distinct production-real signatures (>= MIN_REAL_SIGNATURE_LEN).
    const REAL_SIG_A: &str = "real_signature_A_0123456789012345678901234567890123456789";
    const REAL_SIG_B: &str = "real_signature_B_9876543210987654321098765432109876543210";

    /// A model part carrying an optional `thoughtSignature` slot.
    fn sig_body(sig: Option<&str>) -> Value {
        let mut part = json!({ "thought": true });
        if let Some(s) = sig {
            part["thoughtSignature"] = json!(s);
        }
        session_body(json!([{ "role": "model", "parts": [part] }]))
    }

    /// End-to-end transition observed for prev -> curr through `observe_inner`
    /// (i.e. from the exact serialized snapshots).
    fn sig_transition(prev: &Value, curr: &Value) -> &'static str {
        let s = snap_for(prev);
        observe_inner_body(Some(&s), curr).signature_transition
    }

    #[test]
    fn st1_sentinel_to_real() {
        assert!(REAL_SIG_A.len() >= MIN_REAL_SIGNATURE_LEN);
        let prev = sig_body(Some(SENTINEL_SIGNATURE));
        let curr = sig_body(Some(REAL_SIG_A));
        assert_eq!(classify_diff(&prev, &curr).1, "THINKING_SIGNATURE_CHANGE");
        assert_eq!(sig_transition(&prev, &curr), "SENTINEL_TO_REAL");
    }

    #[test]
    fn st2_real_a_to_real_b() {
        let prev = sig_body(Some(REAL_SIG_A));
        let curr = sig_body(Some(REAL_SIG_B));
        assert_eq!(classify_diff(&prev, &curr).1, "THINKING_SIGNATURE_CHANGE");
        assert_eq!(sig_transition(&prev, &curr), "REAL_TO_REAL_DIFFERENT");
    }

    #[test]
    fn d3_13_existing_classifications_unchanged() {
        // Pure trailing append still classifies as CONTENT_APPEND_ONLY.
        let prev = session_body(json!([text_content("user", "hi")]));
        let mut contents = vec![text_content("user", "hi")];
        contents.push(text_content("model", "yo"));
        let mut curr = session_body(Value::Array(contents));
        curr["requestId"] = json!("req-d3-13");
        let (_p, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "CONTENT_APPEND_ONLY");

        // Two distinct real signatures still classify as REAL_TO_REAL_DIFFERENT.
        let a = sig_body(Some(REAL_SIG_A));
        let b = sig_body(Some(REAL_SIG_B));
        assert_eq!(sig_transition(&a, &b), "REAL_TO_REAL_DIFFERENT");
    }

    #[test]
    fn d3_14_real_to_real_detected() {
        let prev = sig_body(Some(REAL_SIG_A));
        let curr = sig_body(Some(REAL_SIG_B));
        let (path, class) = classify_diff(&prev, &curr);
        assert_eq!(class, "THINKING_SIGNATURE_CHANGE");
        assert!(path.contains("thoughtSignature"));
        let out = observe_inner_body(Some(&snap_for(&prev)), &curr);
        assert_eq!(out.classification, "THINKING_SIGNATURE_CHANGE");
        assert_eq!(out.signature_transition, "REAL_TO_REAL_DIFFERENT");
    }

    #[test]
    fn st3_real_to_sentinel() {
        let prev = sig_body(Some(REAL_SIG_A));
        let curr = sig_body(Some(SENTINEL_SIGNATURE));
        assert_eq!(classify_diff(&prev, &curr).1, "THINKING_SIGNATURE_CHANGE");
        assert_eq!(sig_transition(&prev, &curr), "REAL_TO_SENTINEL");
    }

    #[test]
    fn st4_missing_to_real() {
        let prev = sig_body(None);
        let curr = sig_body(Some(REAL_SIG_A));
        assert_eq!(classify_diff(&prev, &curr).1, "THINKING_SIGNATURE_CHANGE");
        assert_eq!(sig_transition(&prev, &curr), "MISSING_TO_REAL");
    }

    #[test]
    fn st5_real_to_missing() {
        let prev = sig_body(Some(REAL_SIG_A));
        let curr = sig_body(None);
        assert_eq!(classify_diff(&prev, &curr).1, "THINKING_SIGNATURE_CHANGE");
        assert_eq!(sig_transition(&prev, &curr), "REAL_TO_MISSING");
    }

    #[test]
    fn st6_sentinel_to_missing_and_missing_to_sentinel() {
        let sentinel = sig_body(Some(SENTINEL_SIGNATURE));
        let missing = sig_body(None);
        assert_eq!(
            classify_diff(&sentinel, &missing).1,
            "THINKING_SIGNATURE_CHANGE"
        );
        assert_eq!(sig_transition(&sentinel, &missing), "SENTINEL_TO_MISSING");
        assert_eq!(
            classify_diff(&missing, &sentinel).1,
            "THINKING_SIGNATURE_CHANGE"
        );
        assert_eq!(sig_transition(&missing, &sentinel), "MISSING_TO_SENTINEL");
    }

    #[test]
    fn st6b_short_non_sentinel_string_is_unknown_transition() {
        // Present but not production-real: must not be called REAL or MISSING.
        let prev = sig_body(Some(REAL_SIG_A));
        let curr = sig_body(Some("short_non_sentinel"));
        assert_eq!(classify_diff(&prev, &curr).1, "THINKING_SIGNATURE_CHANGE");
        assert_eq!(sig_transition(&prev, &curr), "UNKNOWN_SIGNATURE_TRANSITION");
    }

    #[test]
    fn st7_unchanged_real_signature_produces_no_transition() {
        let prev = sig_body(Some(REAL_SIG_A));
        // Byte-identical snapshot: DEDUPED, no transition.
        assert_eq!(sig_transition(&prev, &prev.clone()), "none");

        // Signature unchanged, only the volatile requestId rotates.
        let mut rotated = prev.clone();
        rotated["requestId"] = json!("req-2");
        let out = observe_inner_body(Some(&snap_for(&prev)), &rotated);
        assert_eq!(out.classification, "REQUEST_ID_ONLY");
        assert_eq!(out.signature_transition, "none");
    }

    #[test]
    fn st8_nested_unrelated_thought_signature_user_key_does_not_trigger() {
        let make = |v: &str| {
            session_body(json!([{
                "role": "user",
                "parts": [{ "text": "hi", "userMeta": { "thoughtSignature": v } }]
            }]))
        };
        let prev = make("nested-a");
        let curr = make("nested-b");
        let (path, class) = classify_diff(&prev, &curr);
        assert_ne!(class, "THINKING_SIGNATURE_CHANGE");
        assert_eq!(class, "UNKNOWN");
        assert!(path.contains("thoughtSignature"));
        assert_eq!(sig_transition(&prev, &curr), "none");
    }

    #[test]
    fn st9_transition_detection_does_not_mutate_bodies() {
        let prev = sig_body(Some(SENTINEL_SIGNATURE));
        let curr = sig_body(Some(REAL_SIG_A));
        let prev_before = prev.clone();
        let curr_before = curr.clone();
        let prev_bytes = ser(&prev);
        let curr_bytes = ser(&curr);

        assert_eq!(sig_transition(&prev, &curr), "SENTINEL_TO_REAL");

        assert_eq!(prev, prev_before);
        assert_eq!(curr, curr_before);
        assert_eq!(ser(&prev), prev_bytes);
        assert_eq!(ser(&curr), curr_bytes);
    }

    #[test]
    fn st10_transition_metadata_leaks_no_signature_value() {
        let prev = sig_body(Some(REAL_SIG_A));
        let curr = sig_body(Some(REAL_SIG_B));
        let out = observe_inner_body(Some(&snap_for(&prev)), &curr);
        assert_eq!(out.classification, "THINKING_SIGNATURE_CHANGE");
        assert_eq!(out.signature_transition, "REAL_TO_REAL_DIFFERENT");

        let line = out.format_line();
        assert!(
            line.contains("signature_transition=REAL_TO_REAL_DIFFERENT"),
            "line must expose only the enum: {line}"
        );
        assert!(!line.contains(REAL_SIG_A), "line leaked previous signature");
        assert!(!line.contains(REAL_SIG_B), "line leaked current signature");
        assert!(
            !line.contains("real_signature_"),
            "line leaked signature fragment"
        );
    }

    #[test]
    fn st11_non_signature_classification_uses_none() {
        // The signature slot also differs, but a higher-precedence change
        // (model) wins; the line must not claim a signature transition.
        let prev = sig_body(Some(REAL_SIG_A));
        let mut curr = sig_body(Some(REAL_SIG_B));
        curr["model"] = json!("gemini-2.5-pro");
        let out = observe_inner_body(Some(&snap_for(&prev)), &curr);
        assert_eq!(out.classification, "MODEL_CHANGE");
        assert_eq!(out.signature_transition, "none");
    }

    // ------------------------------------------------------------------
    // 4/5. Trajectory identity, account/session events, concurrency
    // ------------------------------------------------------------------

    #[test]
    fn t1_missing_session_id_does_not_collapse_traffic() {
        let _g = lock_store();
        reset_for_tests();
        let mut a = session_body(json!([text_content("user", "conversation alpha")]));
        a["request"].as_object_mut().unwrap().remove("sessionId");
        let mut b = session_body(json!([text_content("user", "conversation beta")]));
        b["request"].as_object_mut().unwrap().remove("sessionId");
        a["requestId"] = json!("a");
        b["requestId"] = json!("b");

        let out_a = observe_gated(&ser(&a), &a, "generateContent", Some("acct"));
        let out_b = observe_gated(&ser(&b), &b, "generateContent", Some("acct"));
        assert!(out_a.session_key.is_none());
        assert!(out_b.session_key.is_none());
        assert_ne!(out_a.trajectory_key, out_b.trajectory_key);
        assert_eq!(retained_session_count(), 2);
        reset_for_tests();
    }

    #[test]
    fn t2_account_and_session_changes_are_events_not_new_streams() {
        let _g = lock_store();
        reset_for_tests();
        let mut a = session_body(json!([text_content("user", "stable conversation")]));
        a["request"]["sessionId"] = json!(111);
        a["requestId"] = json!("a");
        let out_a = observe_gated(&ser(&a), &a, "generateContent", Some("acct-1"));
        assert_eq!(out_a.classification, "NEW_SESSION");
        assert!(!out_a.account_changed);

        let mut b = session_body(json!([text_content("user", "stable conversation")]));
        b["request"]["sessionId"] = json!(222);
        b["requestId"] = json!("b");
        let out_b = observe_gated(&ser(&b), &b, "generateContent", Some("acct-2"));
        assert_eq!(out_b.trajectory_key, out_a.trajectory_key);
        assert_eq!(out_b.classification, "SESSION_ID_CHANGE");
        assert!(out_b.account_changed);
        assert!(out_b.session_changed);
        assert_eq!(retained_session_count(), 1);
        reset_for_tests();
    }

    #[test]
    fn t3_fallback_dedupe_without_conversation_anchor() {
        let _g = lock_store();
        reset_for_tests();
        // No user text anchor: image-only-ish body -> environment fallback.
        let mut body = session_body(json!([]));
        body["request"].as_object_mut().unwrap().remove("sessionId");
        let bytes = ser(&body);
        let out1 = observe_gated(&bytes, &body, "generateContent", Some("acct"));
        assert_eq!(out1.classification, "NEW_SESSION");
        assert_eq!(out1.trajectory_source, "environment_fallback");
        let out2 = observe_gated(&bytes, &body, "generateContent", Some("acct"));
        assert_eq!(out2.classification, "DEDUPED");
        assert_eq!(retained_session_count(), 1);
        reset_for_tests();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[allow(clippy::await_holding_lock)] // STORE_LOCK serializes store-touching tests across the concurrent phase
    async fn t4_concurrent_same_trajectory_is_serialized() {
        let _g = lock_store();
        reset_for_tests();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<u64>(64);
        let mut handles = Vec::new();
        for i in 0..16usize {
            let mut contents = vec![text_content("user", "shared first turn")];
            for k in 0..=i {
                contents.push(text_content("model", &format!("turn-{i}-{k}")));
            }
            let mut body = session_body(Value::Array(contents));
            body["request"]["sessionId"] = json!(12345);
            let bytes = ser(&body);
            let tx = tx.clone();
            handles.push(tokio::spawn(async move {
                let out = observe_gated(&bytes, &body, "generateContent", Some("acct"));
                tx.send(out.observation_seq).await.unwrap();
            }));
        }
        drop(tx);
        let mut seqs = Vec::new();
        while let Some(s) = rx.recv().await {
            seqs.push(s);
        }
        for h in handles {
            h.await.unwrap();
        }
        let total = seqs.len();
        seqs.sort_unstable();
        seqs.dedup();
        assert_eq!(total, 16);
        assert_eq!(seqs.len(), 16, "observation sequence must be unique");
        assert_eq!(retained_session_count(), 1);
        reset_for_tests();
    }

    // ------------------------------------------------------------------
    // 6. Memory bound
    // ------------------------------------------------------------------

    #[test]
    fn m_session_cap_evicts_oldest() {
        let _g = lock_store();
        reset_for_tests();
        for i in 0..(MAX_SESSIONS + 5) {
            let mut body = session_body(json!([text_content("user", &format!("conv {i}"))]));
            body["request"]["sessionId"] = json!(i);
            let bytes = ser(&body);
            observe_gated(&bytes, &body, "generateContent", Some("acct"));
        }
        assert_eq!(retained_session_count(), MAX_SESSIONS);
        reset_for_tests();
    }

    #[test]
    fn m_total_bytes_cap_evicts_oldest() {
        let _g = lock_store();
        reset_for_tests();
        set_total_cap_for_tests(4096);
        for i in 0..20 {
            let mut body = session_body(json!([
                text_content("user", &format!("conv {i}")),
                text_content("model", &"x".repeat(400))
            ]));
            body["request"]["sessionId"] = json!(i);
            let bytes = ser(&body);
            observe_gated(&bytes, &body, "generateContent", Some("acct"));
        }
        assert!(
            retained_bytes() <= 4096,
            "retained bytes {} exceeded cap",
            retained_bytes()
        );
        assert!(retained_session_count() < 20);
        set_total_cap_for_tests(0);
        reset_for_tests();
    }

    #[test]
    fn m_oversized_body_not_retained() {
        let _g = lock_store();
        reset_for_tests();
        let mut body = session_body(json!([text_content("user", "hi")]));
        body["big"] = json!("z".repeat(MAX_RETAINED_BYTES + 1));
        let bytes = ser(&body);
        assert!(bytes.len() > MAX_RETAINED_BYTES);
        observe_gated(&bytes, &body, "generateContent", Some("acct"));
        assert_eq!(retained_session_count(), 0);
        reset_for_tests();
    }

    // ------------------------------------------------------------------
    // 7. Privacy
    // ------------------------------------------------------------------

    #[test]
    fn pr1_format_line_contains_no_raw_sensitive_values() {
        let make = |arg_val: &str, out_val: &str| {
            json!({
                "project": "proj-secret",
                "request": {
                    "systemInstruction": { "parts": [{ "text": "SYSTEM_SECRET_TEXT" }] },
                    "tools": [],
                    "generationConfig": {},
                    "sessionId": "sess-secret-123",
                    "contents": [
                        {
                            "role": "model",
                            "parts": [{
                                "functionCall": {
                                    "name": "t",
                                    "args": { "SENSITIVE_KEY_abc": arg_val },
                                    "id": "1"
                                }
                            }]
                        },
                        {
                            "role": "user",
                            "parts": [{
                                "text": "PROMPT_SECRET_TEXT",
                                "functionResponse": {
                                    "name": "t",
                                    "response": { "result": out_val },
                                    "id": "1"
                                }
                            }]
                        },
                        { "role": "user", "parts": [{ "text": "latest" }] }
                    ]
                },
                "model": "gemini-2.0-flash",
                "userAgent": "antigravity",
                "requestId": "req-1"
            })
        };
        let prev = make("SENSITIVE_ARG_VALUE_1", "SENSITIVE_OUTPUT_VALUE_1");
        let curr = make("SENSITIVE_ARG_VALUE_2", "SENSITIVE_OUTPUT_VALUE_2");
        let curr_bytes = ser(&curr);
        let traj = derive_trajectory(&curr);
        let out = observe_inner(
            Some(&snap_for(&prev)),
            &curr_bytes,
            &curr,
            "generateContent",
            Some("acct-secret@example.com"),
            &traj,
            0,
        );
        let line = out.format_line();

        for secret in [
            "sess-secret-123",
            "acct-secret@example.com",
            "SYSTEM_SECRET_TEXT",
            "PROMPT_SECRET_TEXT",
            "SENSITIVE_KEY_abc",
            "SENSITIVE_ARG_VALUE_1",
            "SENSITIVE_ARG_VALUE_2",
            "SENSITIVE_OUTPUT_VALUE_1",
            "SENSITIVE_OUTPUT_VALUE_2",
            "proj-secret",
        ] {
            assert!(!line.contains(secret), "line leaked: {secret}\n{line}");
        }
        assert!(
            line.contains("<key:"),
            "dynamic key should be hashed: {line}"
        );
        assert!(line.contains("ordering=arrival"));
    }

    #[test]
    fn pr2_salted_hashes_stable_within_process() {
        let h1 = short_hash("secret-identifier");
        let h2 = short_hash("secret-identifier");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
        assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(!h1.contains("secret"));
        assert_ne!(h1, short_hash("secret-identifier-2"));
    }

    // ------------------------------------------------------------------
    // 10. Unchanged-request proof + gating
    // ------------------------------------------------------------------

    #[test]
    fn j_disabled_no_retention_and_no_mutation() {
        set_enabled_for_tests(false);
        reset_for_tests();

        let body = session_body(json!([text_content("user", "hi")]));
        let before = body.clone();
        let bytes = ser(&body);
        observe(&bytes, &body, "generateContent", Some("acct"));
        observe(&bytes, &body, "generateContent", Some("acct"));

        assert_eq!(retained_session_count(), 0);
        assert_eq!(body, before);
        assert_eq!(ser(&body), bytes);

        set_enabled_for_tests(false);
    }

    #[test]
    fn j2_enabled_zero_mutation_and_byte_identical() {
        let _g = lock_store();
        reset_for_tests();
        let mut body = session_body(json!([text_content("user", "hi")]));
        body["request"]["sessionId"] = json!("j2-unique");
        let before = body.clone();
        let bytes = ser(&body);

        let outcome = observe_gated(&bytes, &body, "generateContent", Some("acct"));
        assert_eq!(outcome.classification, "NEW_SESSION");
        observe_gated(&bytes, &body, "generateContent", Some("acct"));

        // The Value must be deep-equal and re-serialization byte-identical.
        assert_eq!(body, before);
        assert_eq!(ser(&body), bytes);
        assert_eq!(retained_session_count(), 1);
        reset_for_tests();
    }

    // ------------------------------------------------------------------
    // Helper arithmetic / UTF-8 edges
    // ------------------------------------------------------------------

    mod helpers {
        use super::*;

        #[test]
        fn longest_common_prefix_basics() {
            assert_eq!(longest_common_prefix(b"", b""), 0);
            assert_eq!(longest_common_prefix(b"abc", b""), 0);
            assert_eq!(longest_common_prefix(b"abcdef", b"abcxyz"), 3);
            assert_eq!(longest_common_prefix(b"same", b"same"), 4);
            assert_eq!(longest_common_prefix(b"a", b"b"), 0);
        }

        #[test]
        fn longest_common_prefix_utf8_edge() {
            // "你好" and "你号": both second characters begin with the same
            // UTF-8 lead byte (E5), so the shared byte prefix is 4 bytes,
            // which ends in the middle of the second character.
            let a = "你好".as_bytes();
            let b = "你号".as_bytes();
            assert_eq!(a[..3], b[..3]);
            assert_eq!(a[3], b[3]);
            assert_ne!(a[4], b[4]);
            assert_eq!(longest_common_prefix(a, b), 4);
            // "é1" vs "é2": shared 2-byte 'é' encoding, then differing ASCII.
            assert_eq!(longest_common_prefix("é1".as_bytes(), "é2".as_bytes()), 2);
        }

        #[test]
        fn lcp_percent_previous_or_current_shorter() {
            // current shorter than previous
            assert_eq!(lcp_percent(3, 3, 6), 100.0);
            // previous shorter than current
            assert_eq!(lcp_percent(3, 6, 3), 100.0);
            // partial match, previous shorter
            assert_eq!(lcp_percent(2, 6, 3), 66.67);
            // partial match, current shorter
            assert_eq!(lcp_percent(2, 3, 6), 66.67);
            assert_eq!(lcp_percent(0, 0, 0), 0.0);
        }

        #[test]
        fn estimate_tokens_basics() {
            assert_eq!(estimate_lcp_tokens(0), 0);
            assert_eq!(estimate_lcp_tokens(4), 1);
            assert_eq!(estimate_lcp_tokens(400), 100);
            assert_eq!(estimate_lcp_tokens(3), 0);
        }

        #[test]
        fn sanitize_path_replaces_dynamic_segments() {
            let p = "request.contents[0].parts[0].functionCall.args.SENSITIVE";
            let s = sanitize_path(p);
            assert!(!s.contains("SENSITIVE"));
            assert!(s.contains("functionCall"));
            assert!(s.contains("contents[0]"));
            assert!(s.contains("<key:"));
        }
    }
}
