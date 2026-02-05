import re
from typing import Optional, Tuple

import discord
from discord import app_commands

from discord_buddy.plugin_system import get_plugin_guild_settings, update_plugin_guild_settings

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
    "decision_confidence_threshold": 0.6,
    "curator_reasoning_enabled": False,
    "curator_reasoning_effort": "low"
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


LOW_EFFORT_WORDS = {
    "ok", "okay", "k", "lol", "lmao", "lmfao", "thx", "thanks", "ty",
    "yup", "nope", "idk", "brb", "gtg", "sure", "cool", "nice", "hmm", "meh"
}

OPEN_TRIGGERS = (
    "anyone", "anybody", "someone", "does anyone", "does anybody",
    "can someone", "can anybody", "any tips", "any advice", "any suggestions",
    "recommendations", "thoughts", "what do you think", "ideas", "help"
)


def _strip_special_instruction(text: str) -> str:
    if not text:
        return ""
    marker = "[SPECIAL INSTRUCTION]:"
    if marker in text:
        return text.split(marker, 1)[0].strip()
    return text.strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _looks_open_question(text: str) -> bool:
    if not text:
        return False
    if "?" in text:
        return True
    return any(phrase in text for phrase in OPEN_TRIGGERS)


def _is_low_effort(text: str) -> bool:
    if not text:
        return True
    words = re.findall(r"[a-z0-9']+", text)
    if not words:
        return True
    if len(words) <= 2 and all(word in LOW_EFFORT_WORDS for word in words):
        return True
    if len(words) <= 3 and len("".join(words)) <= 6 and "?" not in text:
        return True
    return False


def _recent_assistant_in_history(history, window: int = 4) -> bool:
    if not history:
        return False
    for msg in history[-window:]:
        if msg.get("role") == "assistant":
            return True
    return False


def _heuristic_autonomous_decision(user_message: str, history, original_message) -> Tuple[Optional[bool], float]:
    cleaned = _strip_special_instruction(user_message)
    normalized = _normalize_text(cleaned)
    if not normalized:
        return False, 0.9

    open_question = _looks_open_question(normalized)
    low_effort = _is_low_effort(normalized)

    replied_to_other = False
    mentions_other = False
    mention_everyone = False
    mention_role = False

    if original_message:
        try:
            if original_message.reference and original_message.reference.resolved:
                replied_to_other = True
        except Exception:
            pass
        try:
            mentions_other = bool(getattr(original_message, "mentions", []))
        except Exception:
            pass
        try:
            mention_everyone = bool(getattr(original_message, "mention_everyone", False))
        except Exception:
            pass
        try:
            mention_role = bool(getattr(original_message, "role_mentions", []))
        except Exception:
            pass

    recent_assistant = _recent_assistant_in_history(history)

    score = 0.5
    if open_question:
        score += 0.25
    if mention_everyone or mention_role:
        score += 0.2
    if recent_assistant:
        score += 0.15
    if low_effort:
        score -= 0.3
    if replied_to_other:
        score -= 0.35
    if mentions_other:
        score -= 0.25

    score = max(0.0, min(score, 1.0))
    should_reply = score >= 0.55
    confidence = score if should_reply else (1.0 - score)
    return should_reply, confidence


def _strip_decision_lines(text: str) -> str:
    if not text:
        return ""
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith("conclusion:") or stripped.startswith("confidence:"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def register_hooks(hooks):
    async def build_context(payload):
        history = payload.get("history") or []
        user_message = str(payload.get("user_message") or "").strip()
        memory_text = str(payload.get("memory_override") or "").strip()
        context_blocks = payload.get("context_blocks") or []
        original_message = payload.get("original_message")
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
        curator_reasoning_enabled = bool(settings.get("curator_reasoning_enabled", DEFAULT_SETTINGS["curator_reasoning_enabled"]))
        curator_reasoning_effort = str(settings.get("curator_reasoning_effort", DEFAULT_SETTINGS["curator_reasoning_effort"])).strip().lower()
        if curator_reasoning_effort not in ("minimal", "low", "medium", "high"):
            curator_reasoning_effort = DEFAULT_SETTINGS["curator_reasoning_effort"]

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

        gate_autonomous = decision_mode in ("gate_autonomous", "autonomous", "auto")
        gate_all = decision_mode in ("gate_all", "all", "gate")
        should_reply = None
        confidence = 0.0
        if decision_enabled:
            should_reply, confidence = _heuristic_autonomous_decision(user_message, history, original_message)
            payload["context_decision"] = {
                "should_reply": should_reply,
                "confidence": confidence
            }
            if should_reply is False and confidence >= decision_confidence_threshold:
                if gate_all or (gate_autonomous and autonomous_join):
                    payload["skip_response"] = True
                    payload["skip_reason"] = f"context_decision_hold (confidence={confidence:.2f})"
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
            reasoning_payload = {"effort": curator_reasoning_effort} if curator_reasoning_enabled else None
            summary = await provider.generate_response(
                messages=[{"role": "user", "content": curator_user}],
                system_prompt=curator_system,
                temperature=0.2,
                model=model_name,
                max_tokens=300,
                reasoning=reasoning_payload
            )
        except Exception:
            return

        if isinstance(summary, str):
            summary = summary.strip()
            if summary:
                cleaned_summary = _strip_decision_lines(summary)
                if cleaned_summary:
                    context_blocks.append(cleaned_summary)
                payload["context_blocks"] = context_blocks

    hooks.on_context_build(build_context, priority=50)


def _context_defaults():
    return {
        "enabled": DEFAULT_SETTINGS["enabled"],
        "max_history": DEFAULT_SETTINGS["max_history"],
        "max_words": DEFAULT_SETTINGS["max_words"],
        "history_mode": DEFAULT_SETTINGS["history_mode"],
        "curator_reasoning_enabled": DEFAULT_SETTINGS["curator_reasoning_enabled"],
        "curator_reasoning_effort": DEFAULT_SETTINGS["curator_reasoning_effort"]
    }


async def _ensure_admin(interaction: discord.Interaction) -> bool:
    if not interaction.guild:
        await interaction.followup.send("This command can only be used in servers.")
        return False
    if not interaction.user.guild_permissions.administrator:
        await interaction.followup.send("❌ Only administrators can use this command!")
        return False
    return True


@app_commands.command(name="context_plugin_info", description="Show Context Awareness plugin settings (Admin only)")
async def context_plugin_info(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    if not await _ensure_admin(interaction):
        return

    defaults = _context_defaults()
    settings = get_plugin_guild_settings("context_awareness", interaction.guild.id, defaults)

    embed = discord.Embed(
        title="🧠 Context Awareness Plugin",
        color=0x0099ff
    )
    embed.add_field(name="Enabled", value=str(bool(settings.get("enabled", True))), inline=True)
    embed.add_field(name="Max History", value=str(settings.get("max_history", defaults["max_history"])), inline=True)
    embed.add_field(name="Max Words", value=str(settings.get("max_words", defaults["max_words"])), inline=True)
    embed.add_field(name="History Mode", value=str(settings.get("history_mode", defaults["history_mode"])), inline=True)
    embed.add_field(
        name="Curator Reasoning",
        value=str(bool(settings.get("curator_reasoning_enabled", defaults["curator_reasoning_enabled"]))),
        inline=True
    )
    embed.add_field(
        name="Curator Effort",
        value=str(settings.get("curator_reasoning_effort", defaults["curator_reasoning_effort"])),
        inline=True
    )
    await interaction.followup.send(embed=embed)


@app_commands.command(name="context_plugin_set", description="Configure Context Awareness plugin (Admin only)")
@app_commands.describe(
    enabled="Enable or disable the context plugin",
    max_history="How many recent messages to scan (1-80)",
    max_words="Max words for the curated context (40-300)",
    history_mode="History handling: keep | trim | curated_only",
    curator_reasoning_enabled="Enable reasoning for the curator model",
    curator_reasoning_effort="Reasoning effort: minimal | low | medium | high"
)
async def context_plugin_set(
    interaction: discord.Interaction,
    enabled: Optional[bool] = None,
    max_history: Optional[int] = None,
    max_words: Optional[int] = None,
    history_mode: Optional[str] = None,
    curator_reasoning_enabled: Optional[bool] = None,
    curator_reasoning_effort: Optional[str] = None
):
    await interaction.response.defer(ephemeral=True)
    if not await _ensure_admin(interaction):
        return

    updates = {}
    if enabled is not None:
        updates["enabled"] = bool(enabled)
    if max_history is not None:
        updates["max_history"] = max(1, min(int(max_history), 80))
    if max_words is not None:
        updates["max_words"] = max(40, min(int(max_words), 300))
    if history_mode is not None:
        mode = str(history_mode).strip().lower()
        if mode in ("keep", "trim", "curated_only"):
            updates["history_mode"] = mode
        else:
            await interaction.followup.send("❌ Invalid history_mode. Use: keep | trim | curated_only.")
            return
    if curator_reasoning_enabled is not None:
        updates["curator_reasoning_enabled"] = bool(curator_reasoning_enabled)
    if curator_reasoning_effort is not None:
        effort = str(curator_reasoning_effort).strip().lower()
        if effort in ("minimal", "low", "medium", "high"):
            updates["curator_reasoning_effort"] = effort
        else:
            await interaction.followup.send("❌ Invalid curator_reasoning_effort. Use: minimal | low | medium | high.")
            return

    if not updates:
        await interaction.followup.send("No changes provided. Use at least one option.")
        return

    update_plugin_guild_settings("context_awareness", interaction.guild.id, updates)

    defaults = _context_defaults()
    settings = get_plugin_guild_settings("context_awareness", interaction.guild.id, defaults)
    await interaction.followup.send(
        f"✅ Updated Context Awareness plugin settings: "
        f"enabled={settings.get('enabled', True)}, "
        f"max_history={settings.get('max_history', defaults['max_history'])}, "
        f"max_words={settings.get('max_words', defaults['max_words'])}, "
        f"history_mode={settings.get('history_mode', defaults['history_mode'])}, "
        f"curator_reasoning_enabled={settings.get('curator_reasoning_enabled', defaults['curator_reasoning_enabled'])}, "
        f"curator_reasoning_effort={settings.get('curator_reasoning_effort', defaults['curator_reasoning_effort'])}"
    )


@app_commands.command(name="context_reasoning_toggle", description="Toggle curator reasoning for Context Awareness (Admin only)")
@app_commands.describe(
    enabled="Enable or disable curator reasoning",
    effort="Reasoning effort: minimal | low | medium | high"
)
async def context_reasoning_toggle(
    interaction: discord.Interaction,
    enabled: Optional[bool] = None,
    effort: Optional[str] = None
):
    await interaction.response.defer(ephemeral=True)
    if not await _ensure_admin(interaction):
        return

    updates = {}
    if enabled is not None:
        updates["curator_reasoning_enabled"] = bool(enabled)
    if effort is not None:
        normalized = str(effort).strip().lower()
        if normalized in ("minimal", "low", "medium", "high"):
            updates["curator_reasoning_effort"] = normalized
        else:
            await interaction.followup.send("❌ Invalid effort. Use: minimal | low | medium | high.")
            return

    if not updates:
        current = get_plugin_guild_settings(
            "context_awareness",
            interaction.guild.id,
            {"curator_reasoning_enabled": False, "curator_reasoning_effort": "low"}
        )
        await interaction.followup.send(
            f"Curator reasoning is currently "
            f"{'enabled' if current.get('curator_reasoning_enabled') else 'disabled'} "
            f"(effort={current.get('curator_reasoning_effort', 'low')}). "
            f"Use `/context_reasoning_toggle true` or set effort."
        )
        return

    update_plugin_guild_settings("context_awareness", interaction.guild.id, updates)
    current = get_plugin_guild_settings(
        "context_awareness",
        interaction.guild.id,
        {"curator_reasoning_enabled": False, "curator_reasoning_effort": "low"}
    )
    await interaction.followup.send(
        f"✅ Curator reasoning "
        f"{'enabled' if current.get('curator_reasoning_enabled') else 'disabled'} "
        f"(effort={current.get('curator_reasoning_effort', 'low')})."
    )


def register(tree, client):
    if not tree.get_command("context_plugin_info"):
        tree.add_command(context_plugin_info)
    if not tree.get_command("context_plugin_set"):
        tree.add_command(context_plugin_set)
    if not tree.get_command("context_reasoning_toggle"):
        tree.add_command(context_reasoning_toggle)
