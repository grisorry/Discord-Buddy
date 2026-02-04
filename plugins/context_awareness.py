from discord_buddy.plugin_system import get_plugin_guild_settings

PLUGIN_INFO = {
    "name": "context_awareness",
    "version": "0.1.0",
    "priority": 50
}

DEFAULT_SETTINGS = {
    "enabled": True,
    "max_history": 20,
    "max_words": 120,
    "history_mode": "keep",
    "decision_enabled": True,
    "decision_mode": "gate_autonomous",
    "decision_confidence_threshold": 0.6
}

def _parse_field(text: str, label: str) -> str:
    prefix = f"{label.lower()}:"
    for line in text.splitlines():
        if line.strip().lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _parse_confidence(text: str) -> float:
    raw = _parse_field(text, "Confidence")
    if not raw:
        return 0.0
    try:
        value = float(raw.split()[0].strip())
        return max(0.0, min(value, 1.0))
    except (ValueError, TypeError):
        return 0.0


def _parse_should_reply(text: str) -> Optional[bool]:
    raw = _parse_field(text, "Conclusion")
    if not raw:
        return None
    value = raw.lower()
    negative_markers = ["hold", "do not", "don't", "no reply", "no response", "skip", "stay silent", "wait"]
    positive_markers = ["reply", "respond", "answer", "yes", "engage", "go ahead"]
    if any(marker in value for marker in negative_markers):
        return False
    if any(marker in value for marker in positive_markers):
        return True
    return None


def register_hooks(hooks):
    async def build_context(payload):
        history = payload.get("history") or []
        user_message = str(payload.get("user_message") or "").strip()
        memory_text = str(payload.get("memory_override") or "").strip()
        context_blocks = payload.get("context_blocks") or []
        ai_manager = payload.get("ai_manager")
        provider_name = payload.get("provider_name")
        model_name = payload.get("model_name")
        custom_url = payload.get("custom_url")
        guild_id = payload.get("guild_id")
        autonomous_join = bool(payload.get("autonomous_join"))

        settings = get_plugin_guild_settings("context_awareness", guild_id, DEFAULT_SETTINGS)
        if not settings.get("enabled", True):
            return
        max_history = int(settings.get("max_history", DEFAULT_SETTINGS["max_history"]))
        max_history = max(1, min(max_history, 80))
        max_words = int(settings.get("max_words", DEFAULT_SETTINGS["max_words"]))
        max_words = max(40, min(max_words, 300))
        history_mode = str(settings.get("history_mode", DEFAULT_SETTINGS["history_mode"])).lower()
        decision_enabled = bool(settings.get("decision_enabled", True))
        decision_mode = str(settings.get("decision_mode", DEFAULT_SETTINGS["decision_mode"])).lower()
        try:
            decision_confidence_threshold = float(settings.get("decision_confidence_threshold", DEFAULT_SETTINGS["decision_confidence_threshold"]))
        except (TypeError, ValueError):
            decision_confidence_threshold = DEFAULT_SETTINGS["decision_confidence_threshold"]
        decision_confidence_threshold = max(0.0, min(decision_confidence_threshold, 1.0))

        if history_mode == "trim":
            history = history[-max_history:]
            payload["history"] = history
        elif history_mode == "curated_only":
            minimal = []
            if history:
                if history[-1].get("role") == "user":
                    # Keep the last assistant message (if any) plus the latest user message.
                    for idx in range(len(history) - 2, -1, -1):
                        if history[idx].get("role") == "assistant":
                            minimal = [history[idx], history[-1]]
                            break
                if not minimal:
                    minimal = [history[-1]]
            payload["history"] = minimal

        if not user_message:
            return
        if not history and not memory_text:
            return
        if not ai_manager or not provider_name or not model_name:
            return

        def flatten_content(content):
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif part.get("type") in ("image_url", "image"):
                            parts.append("[Image]")
                    elif isinstance(part, str):
                        parts.append(part)
                return " ".join(p for p in parts if p).strip()
            return str(content).strip()

        trimmed = history[-max_history:]
        history_lines = []
        for msg in trimmed:
            role = msg.get("role", "user")
            content = flatten_content(msg.get("content", ""))
            if not content:
                continue
            if len(content) > 400:
                content = content[:400] + "..."
            role_label = "USER" if role == "user" else "ASSISTANT" if role == "assistant" else role.upper()
            history_lines.append(f"{role_label}: {content}")

        history_text = "\n".join(history_lines)
        memory_block = memory_text if memory_text else "None."

        curator_system = (
            "You are a context curator for a Discord chat assistant. "
            "Your job is to extract ONLY the most relevant context for the latest user message. "
            "Be concise and avoid analysis. Use the exact format:\n"
            "Topic: ...\n"
            "Target: ...\n"
            "Key facts: ...\n"
            "Relevant snippets: ...\n"
            "Guidance: ...\n"
            "Conclusion: Reply now | Hold\n"
            "Confidence: 0.0-1.0\n"
            f"If a field is unknown, write 'Unknown'. Keep the entire output under {max_words} words."
        )

        curator_user = (
            f"Latest user message:\n{user_message}\n\n"
            f"Relevant memory:\n{memory_block}\n\n"
            f"Recent history:\n{history_text}\n"
        )

        provider = None
        if provider_name == "custom" and custom_url:
            provider = ai_manager.get_custom_provider(custom_url)
        else:
            provider = ai_manager.providers.get(provider_name)
        if not provider:
            return

        try:
            summary = await provider.generate_response(
                messages=[{"role": "user", "content": curator_user}],
                system_prompt=curator_system,
                temperature=0.2,
                model=model_name,
                max_tokens=300,
                reasoning=None
            )
        except Exception:
            return

        if isinstance(summary, str):
            summary = summary.strip()
            if summary:
                context_blocks.append(summary)
                payload["context_blocks"] = context_blocks

                if decision_enabled:
                    should_reply = _parse_should_reply(summary)
                    confidence = _parse_confidence(summary)
                    payload["context_decision"] = {
                        "should_reply": should_reply,
                        "confidence": confidence
                    }

                    gate_autonomous = decision_mode in ("gate_autonomous", "autonomous", "auto")
                    gate_all = decision_mode in ("gate_all", "all", "gate")

                    if should_reply is False and confidence >= decision_confidence_threshold:
                        if gate_all or (gate_autonomous and autonomous_join):
                            payload["skip_response"] = True
                            payload["skip_reason"] = f"context_decision_hold (confidence={confidence:.2f})"

    hooks.on_context_build(build_context, priority=50)
