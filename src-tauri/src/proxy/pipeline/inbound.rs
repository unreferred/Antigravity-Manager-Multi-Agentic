use super::policy::ProxyProtocol;
use serde_json::{json, Value};

/// 统一进站思考管线（InboundThinkingPipeline）
/// 接收任何协议转译成的 Google contents 统一报文，单向流转执行：
/// 1. 协议策略签名清洗 (Chat 协议丢弃客户端签名，其他协议验签)
/// 2. 思考块首位强制排序与占位符规范化
/// 3. 状态机历史思维链无损复活 (Hydration)
/// 4. 终审脱敏与前缀缓存格式规范化 (Finalize)
pub struct InboundThinkingPipeline;

impl InboundThinkingPipeline {
    /// 执行统一进站处理
    pub fn process_contents(
        contents: &mut Vec<Value>,
        protocol: ProxyProtocol,
        target_model: &str,
        is_thinking_enabled: bool,
        session_id: Option<&str>,
        _is_retry: bool,
    ) {
        let trusts_signature = protocol.trusts_client_signature();

        // 1. 协议策略清洗与位置规范化
        for content in contents.iter_mut() {
            let is_model = matches!(
                content.get("role").and_then(|r| r.as_str()),
                Some("model") | Some("assistant")
            );

            if let Some(parts) = content.get_mut("parts").and_then(|p| p.as_array_mut()) {
                if is_model {
                    let mut new_parts = Vec::with_capacity(parts.len());
                    let mut saw_non_thinking = false;

                    for part in parts.drain(..) {
                        let is_thought = part
                            .get("thought")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                            || (part.get("thoughtSignature").is_some()
                                && part.get("functionCall").is_none()
                                && part.get("functionResponse").is_none());

                        if is_thought {
                            let text = part.get("text").and_then(|v| v.as_str()).unwrap_or("");
                            let is_placeholder =
                                crate::proxy::thinking_store::is_placeholder_thought(text);
                            let final_thought_text = if is_placeholder || text.is_empty() {
                                "..."
                            } else {
                                text.trim()
                            };

                            // 若非首位，或者前面已有文本部件，降级为普通文本
                            if saw_non_thinking || !new_parts.is_empty() {
                                if !final_thought_text.is_empty() {
                                    new_parts.push(json!({ "text": final_thought_text }));
                                    saw_non_thinking = true;
                                }
                                continue;
                            }

                            // 校验客户端签名有效性与模型兼容性
                            let mut effective_sig = None;
                            if let Some(sig) = part
                                .get("thoughtSignature")
                                .or_else(|| part.get("thought_signature"))
                                .or_else(|| part.get("signature"))
                                .and_then(|s| s.as_str())
                            {
                                if sig == crate::proxy::thinking_store::SENTINEL_SIGNATURE {
                                    effective_sig = Some(sig.to_string());
                                } else if trusts_signature && sig.len() >= 50 {
                                    let cached_family = crate::proxy::SignatureCache::global()
                                        .get_signature_family(sig);
                                    let compatible = match cached_family {
                                        Some(family) => {
                                            crate::proxy::mappers::common_utils::is_model_compatible(
                                                &family,
                                                target_model,
                                            )
                                        }
                                        None => true,
                                    };
                                    if compatible {
                                        effective_sig = Some(sig.to_string());
                                    }
                                }
                            }

                            let mut thought_obj = json!({
                                "text": final_thought_text,
                                "thought": true,
                            });
                            if let Some(sig) = effective_sig {
                                thought_obj["thoughtSignature"] = json!(sig);
                            }
                            // [V2B] Preserve the internal fallback provenance marker
                            // (stripped later by finalize) so historical unresolved
                            // fallback thought signatures can be stabilized.
                            if part
                                .get(crate::proxy::thinking_store::SIG_FALLBACK_MARKER)
                                .is_some()
                            {
                                thought_obj[crate::proxy::thinking_store::SIG_FALLBACK_MARKER] =
                                    json!(true);
                            }
                            new_parts.push(thought_obj);
                        } else {
                            saw_non_thinking = true;
                            new_parts.push(part);
                        }
                    }
                    *parts = new_parts;
                }
            }
        }

        // 2. 状态机无损复活 (Hydration)
        if is_thinking_enabled {
            if let Some(s_id) = session_id {
                crate::proxy::thinking_store::hydrate_gemini_contents(s_id, contents);
            }
        }

        // 3. 终审把关与脱敏规范化 (Finalize)
        crate::proxy::thinking_store::finalize_gemini_contents_thinking(
            contents,
            is_thinking_enabled,
        );
    }
}
