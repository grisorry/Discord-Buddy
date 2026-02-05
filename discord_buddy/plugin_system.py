import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

HookFunc = Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]


@dataclass(order=True)
class HookEntry:
    priority: int
    order: int
    func: HookFunc
    plugin_name: str


class PluginHooks:
    """Registration helpers exposed to plugins."""

    def __init__(self, manager: "PluginManager", plugin_name: str, plugin_info: Optional[Dict[str, Any]] = None):
        self._manager = manager
        self._plugin_name = plugin_name
        self._plugin_info = plugin_info or {}

    def _priority(self, priority: Optional[int]) -> int:
        if priority is not None:
            return priority
        return int(self._plugin_info.get("priority", 100))

    def on_context_build(self, func: HookFunc, priority: Optional[int] = None) -> None:
        self._manager.register_hook("context_build", func, self._priority(priority), self._plugin_name)

    def on_before_generate(self, func: HookFunc, priority: Optional[int] = None) -> None:
        self._manager.register_hook("before_generate", func, self._priority(priority), self._plugin_name)

    def on_after_generate(self, func: HookFunc, priority: Optional[int] = None) -> None:
        self._manager.register_hook("after_generate", func, self._priority(priority), self._plugin_name)


class PluginManager:
    """Central registry for plugin hooks and execution."""

    def __init__(self):
        self._hooks: Dict[str, List[HookEntry]] = {}
        self._order_counter = 0
        self.on_error: Optional[Callable[[str, str, Exception], None]] = None

    def reset(self) -> None:
        self._hooks.clear()
        self._order_counter = 0

    def hooks_for(self, plugin_name: str, plugin_info: Optional[Dict[str, Any]] = None) -> PluginHooks:
        return PluginHooks(self, plugin_name, plugin_info)

    def register_hook(self, hook_name: str, func: HookFunc, priority: int, plugin_name: str) -> None:
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._order_counter += 1
        self._hooks[hook_name].append(HookEntry(priority=priority, order=self._order_counter, func=func, plugin_name=plugin_name))

    def unregister_plugin(self, plugin_name: str) -> None:
        """Remove all hooks registered by a specific plugin."""
        for hook_name, entries in list(self._hooks.items()):
            filtered = [entry for entry in entries if entry.plugin_name != plugin_name]
            if filtered:
                self._hooks[hook_name] = filtered
            else:
                self._hooks.pop(hook_name, None)

    def _normalize_payload(self, hook_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure common payload keys exist to reduce plugin boilerplate."""
        if not isinstance(payload, dict):
            return {}

        def ensure_list(key: str) -> None:
            value = payload.get(key)
            if isinstance(value, list):
                return
            payload[key] = []

        def ensure_str(key: str) -> None:
            value = payload.get(key)
            if isinstance(value, str):
                return
            payload[key] = "" if value is None else str(value)

        if hook_name == "context_build":
            ensure_list("context_blocks")
            ensure_list("history")
            ensure_list("attachments")
            ensure_str("user_message")
            payload.setdefault("memory_override", "")
            payload.setdefault("provider_name", None)
            payload.setdefault("model_name", None)
            payload.setdefault("custom_url", None)
        elif hook_name == "before_generate":
            ensure_list("context_blocks")
            ensure_list("history")
            ensure_str("system_prompt")
            ensure_str("user_message")
            payload.setdefault("temperature", None)
            payload.setdefault("reasoning", None)
            payload.setdefault("max_tokens", None)
        elif hook_name == "after_generate":
            ensure_list("history")
            ensure_str("system_prompt")
            ensure_str("user_message")
            ensure_str("response")

        return payload

    async def run_hook(self, hook_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        hooks = self._hooks.get(hook_name, [])
        if not hooks:
            return payload
        payload = self._normalize_payload(hook_name, payload)
        for entry in sorted(hooks):
            try:
                result = entry.func(payload)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, dict):
                    payload.update(result)
            except Exception as exc:
                if self.on_error:
                    self.on_error(entry.plugin_name, hook_name, exc)
        return payload


CONFIG_FILE = os.path.join("bot_data", "config.json")
LEGACY_PLUGIN_SETTINGS_FILE = os.path.join("bot_data", "plugin_settings.json")


def _load_config() -> Dict[str, Any]:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def _save_config(data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _load_legacy_plugin_settings() -> Dict[str, Any]:
    try:
        if os.path.exists(LEGACY_PLUGIN_SETTINGS_FILE):
            with open(LEGACY_PLUGIN_SETTINGS_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}


def _load_all_plugin_settings() -> Dict[str, Any]:
    config = _load_config()
    plugins = config.get("plugins", {})
    if isinstance(plugins, dict) and plugins:
        return plugins
    legacy = _load_legacy_plugin_settings()
    if legacy:
        config["plugins"] = legacy
        _save_config(config)
        return legacy
    return {}


def _save_all_plugin_settings(data: Dict[str, Any]) -> None:
    config = _load_config()
    config["plugins"] = data
    _save_config(config)


def get_plugin_settings(plugin_name: str) -> Dict[str, Any]:
    data = _load_all_plugin_settings()
    settings = data.get(plugin_name, {})
    return settings if isinstance(settings, dict) else {}


def get_plugin_guild_settings(plugin_name: str, guild_id: Optional[int], defaults: Dict[str, Any]) -> Dict[str, Any]:
    settings = get_plugin_settings(plugin_name)
    base = settings.get("default", {}) if isinstance(settings.get("default"), dict) else {}
    guild_settings = {}
    if guild_id is not None:
        guild_settings = settings.get("guilds", {}).get(str(guild_id), {}) if isinstance(settings.get("guilds"), dict) else {}
    merged = {}
    merged.update(defaults or {})
    merged.update(base)
    merged.update(guild_settings)
    return merged


def update_plugin_guild_settings(plugin_name: str, guild_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
    data = _load_all_plugin_settings()
    settings = data.get(plugin_name, {})
    if not isinstance(settings, dict):
        settings = {}
    if "guilds" not in settings or not isinstance(settings.get("guilds"), dict):
        settings["guilds"] = {}
    guild_key = str(guild_id)
    current = settings["guilds"].get(guild_key, {})
    if not isinstance(current, dict):
        current = {}
    current.update({k: v for k, v in updates.items()})
    settings["guilds"][guild_key] = current
    data[plugin_name] = settings
    _save_all_plugin_settings(data)
    return current


def update_plugin_defaults(plugin_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    data = _load_all_plugin_settings()
    settings = data.get(plugin_name, {})
    if not isinstance(settings, dict):
        settings = {}
    current = settings.get("default", {})
    if not isinstance(current, dict):
        current = {}
    current.update({k: v for k, v in updates.items()})
    settings["default"] = current
    data[plugin_name] = settings
    _save_all_plugin_settings(data)
    return current
