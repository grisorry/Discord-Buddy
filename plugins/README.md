# Plugins

This folder contains optional plugins loaded at startup by `load_plugins()` in `discord_buddy/core.py`.

## Quick Start

Create a file like `plugins/my_plugin.py`:

```python
PLUGIN_INFO = {
    "name": "my_plugin",
    "version": "0.1.0",
    "priority": 100
}

def register_hooks(hooks):
    async def on_context(payload):
        payload["context_blocks"].append("Short context note.")
    hooks.on_context_build(on_context, priority=50)

    async def on_before_generate(payload):
        # Optional: inject context blocks into the system prompt
        blocks = payload.get("context_blocks") or []
        if not blocks:
            return
        system_prompt = payload.get("system_prompt", "")
        injection = "\n\nPlugin notes:\n" + "\n".join(blocks)
        payload["system_prompt"] = system_prompt + injection
        return payload
    hooks.on_before_generate(on_before_generate, priority=50)
```

Reload the bot (or call `/sync`) and the plugin will load.

## Supported Hook Points

Plugins can register hooks to modify the pipeline:

- `context_build`  
  Build curated context blocks before the system prompt is constructed.

- `before_generate`  
  Last chance to mutate `system_prompt`, `history`, `temperature`, or `reasoning`.

- `after_generate`  
  Post-process the final response text.

## Hook Payloads

All hooks receive a `payload` dict. Relevant keys include:

`context_build`:
- `history`, `memory_override`, `user_message`, `attachments`
- `persona_lore` (character lore available to curator plugins)
- `context_blocks` (append strings here)
- `ai_manager`, `provider_name`, `model_name`, `custom_url`

`before_generate`:
- `system_prompt`, `history`, `temperature`, `reasoning`, `max_tokens`
 - `context_blocks` (if your plugin wants to inject context)

`after_generate`:
- `response`, `system_prompt`, `history`, `user_message`

## Plugin Settings

Plugin settings are stored in:

```
bot_data/config.json
```

Structure:

```json
{
  "plugins": {
    "my_plugin": {
      "default": {
        "enabled": true
      },
      "guilds": {
        "123456789012345678": {
          "enabled": false
        }
      }
    }
  }
}
```

## Notes

- Hooks run in priority order (lower number first).
- Hook errors are isolated and won't crash the bot.
- Plugins can still define `register(tree, client)` or `setup(tree, client)` for slash commands.
- Common payload keys (like `context_blocks`, `history`, `system_prompt`, `response`) are normalized before hooks run.
