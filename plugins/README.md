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
        payload["context_blocks"].append("Topic: ...")
    hooks.on_context_build(on_context, priority=50)
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
- `context_blocks` (append strings here)
- `ai_manager`, `provider_name`, `model_name`, `custom_url`

`before_generate`:
- `system_prompt`, `history`, `temperature`, `reasoning`, `max_tokens`

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
    "context_awareness": {
      "default": {
        "enabled": true,
        "max_history": 20,
        "max_words": 120,
        "history_mode": "keep"
      },
      "guilds": {
        "123456789012345678": {
          "enabled": true,
          "max_history": 25,
          "max_words": 150,
          "history_mode": "trim"
        }
      }
    }
  }
}
```

The `context_awareness` plugin uses `/context_plugin_set` and `/context_plugin_info` for configuration.

`history_mode` options:
- `keep` (default): keep full history for the main model
- `trim`: keep only the last `max_history` messages
- `curated_only`: keep only the latest user message (and last assistant if present)

## Notes

- Hooks run in priority order (lower number first).
- Hook errors are isolated and won't crash the bot.
- Plugins can still define `register(tree, client)` or `setup(tree, client)` for slash commands.
