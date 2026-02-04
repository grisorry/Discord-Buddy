import os
import discord
from discord import app_commands
from discord.ext import commands
from discord_buddy.core import *
from discord_buddy.plugin_system import get_plugin_guild_settings, update_plugin_guild_settings


class CommandsCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # SLASH COMMANDS - AI PROVIDER MANAGEMENT

    @app_commands.command(name="model_set", description="Set AI provider and model for this server (Admin only)")
    async def set_model(self, interaction: discord.Interaction, provider: str, model: str = None, custom_url: str = None):
        """Set AI provider and model for the server"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("❌ This command can only be used in servers!")
            return
    
        # Check admin permissions
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use this command!")
            return
    
        # Check if provider exists and is available
        if provider not in ai_manager.providers:
            available = list(ai_manager.providers.keys())
            await interaction.followup.send(f"❌ Invalid provider! Available: {', '.join(available)}")
            return
    
        # Custom provider requires URL
        if provider == "custom":
            if custom_url is None:
                await interaction.followup.send("❌ **Custom provider requires a URL!**\n\n"
                                               "**Usage:** `/model_set custom [model] <url>`\n"
                                               "**Examples:**\n"
                                               "• `/model_set custom llama-3.1-8b http://localhost:1234/v1`\n"
                                               "• `/model_set custom custom-model http://127.0.0.1:8000/v1`\n"
                                               "• `/model_set custom gpt-4 https://api.your-server.com/v1`")
                return
        
            # Validate URL format
            if not (custom_url.startswith('http://') or custom_url.startswith('https://')):
                await interaction.followup.send("❌ **Invalid URL format!**\n\n"
                                               "URL must start with `http://` or `https://`\n"
                                               "**Examples:**\n"
                                               "• `http://localhost:1234/v1`\n"
                                               "• `https://api.your-server.com/v1`")
                return
    
        if not ai_manager.providers[provider].is_available():
            if provider == "custom" and not CUSTOM_API_KEY:
                await interaction.followup.send(f"❌ **{provider.title()} is not available.**\n\n"
                                               f"The CUSTOM_API_KEY environment variable is not configured.\n"
                                               f"Please contact the bot administrator.")
            else:
                await interaction.followup.send(f"❌ **{provider.title()} is not available.**\n\n"
                                               f"The API key is not configured.")
            return
    
        # Handle model selection
        if provider == "custom":
            if model is None:
                model = ai_manager.providers[provider].get_default_model()
        else:
            available_models = ai_manager.get_provider_models(provider)
        
            if model is None:
                model = ai_manager.providers[provider].get_default_model()
            else:
                if model not in available_models:
                    await interaction.followup.send(f"❌ Model '{model}' not available for {provider}!\n"
                                                   f"Available models: {', '.join(available_models)}")
                    return
    
        # Set for guild
        success = ai_manager.set_guild_provider(interaction.guild.id, provider, model, custom_url)
        if success:
            response_text = f"✅ **Server AI Model Set!**\n" \
                           f"**Provider:** {provider.title()}\n" \
                           f"**Model:** {model}\n"
        
            if provider == "custom" and custom_url:
                response_text += f"**Custom URL:** `{custom_url}`\n"
        
            response_text += f"\n💡 This affects all conversations in this server, including DMs with server members."
        
            await interaction.followup.send(response_text)
        else:
            await interaction.followup.send("❌ Failed to set provider.")

    @app_commands.command(name="dm_server_select", description="Choose which server's settings to use for your DMs")
    async def dm_server_select(self, interaction: discord.Interaction, server_name: str = None):
        """Select which server's settings to use for DMs"""
        await interaction.response.defer(ephemeral=True)
    
        user_id = interaction.user.id
    
        # Collect all shared guilds and their settings
        shared_guilds = {}
        for guild in client.guilds:
            # Try both methods to find the member
            member = guild.get_member(user_id)
            if not member:
                try:
                    member = await guild.fetch_member(user_id)
                except (discord.NotFound, discord.Forbidden):
                    member = None
        
            if member:  # User is in this guild
                provider, model = ai_manager.get_guild_settings(guild.id)
                history_length = get_history_length(guild.id)
                temperature = get_temperature(guild.id)
            
                shared_guilds[guild.name.lower()] = {
                    "guild_id": guild.id,
                    "guild_name": guild.name,
                    "provider": provider,
                    "model": model,
                    "history_length": history_length,
                    "temperature": temperature
                }
    
        if not shared_guilds:
            await interaction.followup.send("❌ **No shared servers found!**\n"
                                           "Make sure you're in a server with the bot.")
            return
    
        # If no server name provided, show available options
        if server_name is None:
            embed = discord.Embed(
                title="🔧 Choose DM Server Settings",
                description="Select which server's settings to use in your DMs:",
                color=0x00ff99
            )
        
            # Get current setting
            current_guild_id = dm_server_selection.get(user_id)
            current_server = None
            if current_guild_id:
                for guild_data in shared_guilds.values():
                    if guild_data["guild_id"] == current_guild_id:
                        current_server = guild_data["guild_name"]
                        break
        
            if current_server:
                embed.add_field(
                    name="Current Setting",
                    value=f"Using settings from **{current_server}**",
                    inline=False
                )
            else:
                embed.add_field(
                    name="Current Setting",
                    value="Using automatic selection (first shared server found)",
                    inline=False
                )
        
            # List available servers with their settings
            server_list = []
            for guild_data in shared_guilds.values():
                server_info = f"• **{guild_data['guild_name']}**\n" \
                             f"  Model: {guild_data['provider'].title()} - {guild_data['model']}\n" \
                             f"  History: {guild_data['history_length']} messages\n" \
                             f"  Temperature: {guild_data['temperature']}"
                server_list.append(server_info)
        
            embed.add_field(
                name="Available Servers",
                value="\n\n".join(server_list),
                inline=False
            )
        
            embed.set_footer(text="Use /dm_server_select <server_name> to choose\nUse /dm_server_reset to go back to automatic")
            await interaction.followup.send(embed=embed)
            return
    
        # Find the server by name (case-insensitive)
        server_name_lower = server_name.lower()
        selected_guild = None
    
        # Try exact match first
        if server_name_lower in shared_guilds:
            selected_guild = shared_guilds[server_name_lower]
        else:
            # Try partial match
            for guild_name, guild_data in shared_guilds.items():
                if server_name_lower in guild_name:
                    selected_guild = guild_data
                    break
    
        if not selected_guild:
            available_servers = [guild_data["guild_name"] for guild_data in shared_guilds.values()]
            await interaction.followup.send(f"❌ **Server not found!**\n\n"
                                           f"Available servers: {', '.join(available_servers)}\n"
                                           f"Use `/dm_server_select` without arguments to see all options.")
            return
    
        # Set the DM server selection
        dm_server_selection[user_id] = selected_guild["guild_id"]
        save_json_data(DM_SERVER_SELECTION_FILE, dm_server_selection)
    
        await interaction.followup.send(f"✅ **DM Server Settings Set!**\n\n"
                                       f"**Server:** {selected_guild['guild_name']}\n"
                                       f"**Model:** {selected_guild['provider'].title()} - {selected_guild['model']}\n"
                                       f"**History Length:** {selected_guild['history_length']} messages\n"
                                       f"**Temperature:** {selected_guild['temperature']}\n\n"
                                       f"💬 Your DMs will now use these settings!\n"
                                       f"💡 Use `/dm_server_reset` to go back to automatic selection.")

    @dm_server_select.autocomplete('server_name')
    async def dm_server_name_autocomplete(self, interaction: discord.Interaction, current: str):
        """Autocomplete for server names in DM server selection"""
        user_id = interaction.user.id
        shared_servers = []
    
        for guild in client.guilds:
            member = guild.get_member(user_id)
            if not member:
                try:
                    member = await guild.fetch_member(user_id)
                except (discord.NotFound, discord.Forbidden):
                    continue
        
            if member and current.lower() in guild.name.lower():
                shared_servers.append(app_commands.Choice(name=guild.name, value=guild.name))
    
        return shared_servers[:25]  # Discord limits to 25 choices

    @app_commands.command(name="model_info", description="Show current AI provider and model settings")
    async def model_info(self, interaction: discord.Interaction):
        """Display current AI provider and model information"""
        await interaction.response.defer(ephemeral=True)
    
        embed = discord.Embed(
            title="🤖 AI Model Information",
            color=0x00ff99
        )
    
        # Show current settings
        if interaction.guild:
            provider, model = ai_manager.get_guild_settings(interaction.guild.id)
        
            settings_text = f"**Provider:** {provider.title()}\n**Model:** {model}"
        
            # Add custom URL info if using custom provider
            if provider == "custom":
                custom_url = ai_manager.get_guild_custom_url(interaction.guild.id)
                settings_text += f"\n**Custom URL:** `{custom_url}`"
        
            embed.add_field(
                name="Current Server Settings",
                value=settings_text,
                inline=False
            )
    
        # Show DM server selection
        user_id = interaction.user.id
        selected_guild_id = dm_server_selection.get(user_id)
    
        if selected_guild_id:
            selected_guild = client.get_guild(selected_guild_id)
            if selected_guild:
                provider, model = ai_manager.get_guild_settings(selected_guild_id)
                dm_settings_text = f"**Selected Server:** {selected_guild.name}\n" \
                                  f"**Model:** {provider.title()} - {model}\n" \
                                  f"**History Length:** {get_history_length(selected_guild_id)} messages\n" \
                                  f"**Temperature:** {get_temperature(selected_guild_id)}"
            
                embed.add_field(
                    name="Your DM Settings",
                    value=dm_settings_text,
                    inline=False
                )
            else:
                embed.add_field(
                    name="DM Information",
                    value="**Selected server no longer available** - will use automatic selection.\nUse `/dm_server_select` to choose a new server.",
                    inline=False
                )
        else:
            embed.add_field(
                name="DM Information",
                value="**Automatic selection** - DMs use settings from the first shared server.\nUse `/dm_server_select` to choose a specific server's settings.",
                inline=False
            )
    
        # Show provider availability
        providers_status = ai_manager.get_available_providers()
        status_lines = []
        for provider, available in providers_status.items():
            status = "✅" if available else "❌"
            if provider == "custom":
                status_lines.append(f"{status} {provider.title()} (requires URL)")
            else:
                status_lines.append(f"{status} {provider.title()}")
    
        embed.add_field(
            name="Provider Availability",
            value="\n".join(status_lines),
            inline=False
        )
    
        embed.set_footer(text="Use /model_set <provider> [model] to change server settings (Admin only)\nUse /dm_server_select to choose DM server")
        await interaction.followup.send(embed=embed)

    # Autocomplete for model commands
    @set_model.autocomplete('provider')
    async def provider_autocomplete(self, interaction: discord.Interaction, current: str):
        """Autocomplete for AI providers"""
        providers = list(ai_manager.providers.keys())
        return [app_commands.Choice(name=provider.title(), value=provider) 
                for provider in providers if current.lower() in provider.lower()]

    @set_model.autocomplete('custom_url')
    async def custom_url_autocomplete(self, interaction: discord.Interaction, current: str):
        """Autocomplete for custom URLs"""
        # Only show suggestions if custom provider is selected
        provider = None
        for option in interaction.data.get('options', []):
            if option['name'] == 'provider':
                provider = option['value']
                break
    
        if provider != "custom":
            return []
    
        # Provide common local API URLs as suggestions
        suggestions = [
            "http://localhost:8000/v1", 
            "http://127.0.0.1:1234/v1",
            "https://api.crystalsraw.me/v1",
            "https://openrouter.ai/api/v1"
        ]
    
        return [app_commands.Choice(name=url, value=url) 
                for url in suggestions if current.lower() in url.lower()]

    @set_model.autocomplete('model')
    async def model_autocomplete(self, interaction: discord.Interaction, current: str):
        """Autocomplete for models based on selected provider"""
        provider = None
        # Try to get the provider from the current interaction
        for option in interaction.data.get('options', []):
            if option['name'] == 'provider':
                provider = option['value']
                break
    
        if not provider or provider not in ai_manager.providers:
            return []
    
        # For custom provider, don't provide autocomplete
        if provider == "custom":
            return [app_commands.Choice(name="Type your model name", value="")]
    
        models = ai_manager.get_provider_models(provider)
        return [app_commands.Choice(name=model, value=model) 
                for model in models if current.lower() in model.lower()][:25]

    @app_commands.command(name="temperature_set", description="Set the AI temperature (creativity level) for this server (Admin only)")
    async def set_temperature(self, interaction: discord.Interaction, temperature: float):
        """Set AI temperature for server"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("Temperature can only be set in servers, not DMs.")
            return
    
        # Check admin permissions
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use this command!")
            return
    
        if not (0.0 <= temperature <= 2.0):
            await interaction.followup.send("Temperature must be between 0.0 and 2.0.\n"
                                           "• **0.0-0.3**: Very focused and deterministic\n"
                                           "• **0.4-0.7**: Balanced creativity\n"
                                           "• **0.8-1.2**: Creative and varied\n"
                                           "• **1.3-2.0**: Very creative and unpredictable")
            return
    
        guild_temperatures[interaction.guild.id] = temperature
        save_json_data(TEMPERATURE_FILE, guild_temperatures)
    
        # Provide helpful description based on temperature range
        if temperature <= 0.3:
            description = "Very focused and deterministic responses"
        elif temperature <= 0.7:
            description = "Balanced creativity and consistency"
        elif temperature <= 1.2:
            description = "Creative and varied responses"
        else:
            description = "Very creative and unpredictable responses"
    
        await interaction.followup.send(f"🌡️ Temperature set to **{temperature}** for this server!\n"
                                       f"**Style:** {description}\n\n"
                                       f"💡 *Lower values = more consistent, higher values = more creative*\n"
                                       f"🔒 *This setting also applies to DMs with server members*")

    @app_commands.command(name="reasoning_toggle", description="Toggle reasoning mode (Admin only in servers)")
    async def reasoning_toggle(self, interaction: discord.Interaction, enabled: bool = None):
        """Toggle reasoning mode for this context"""
        await interaction.response.defer(ephemeral=True)

        is_dm = isinstance(interaction.channel, discord.DMChannel)

        if not is_dm and not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use this command in servers!")
            return

        if enabled is None:
            current = is_reasoning_enabled(interaction.guild.id if interaction.guild else None, interaction.user.id if interaction.user else None, is_dm)
            await interaction.followup.send(f"Reasoning is currently {'enabled' if current else 'disabled'} for this {'DM' if is_dm else 'server'}.\nUse `/reasoning_toggle true` or `/reasoning_toggle false`.")
            return

        if is_dm:
            dm_reasoning_settings[interaction.user.id] = enabled
            save_json_data(DM_REASONING_SETTINGS_FILE, dm_reasoning_settings)
            await interaction.followup.send(f"✅ Reasoning {'enabled' if enabled else 'disabled'} for your DMs.")
        else:
            guild_reasoning_settings[interaction.guild.id] = enabled
            save_json_data(REASONING_SETTINGS_FILE, guild_reasoning_settings)
            await interaction.followup.send(f"✅ Reasoning {'enabled' if enabled else 'disabled'} for this server.")

    @app_commands.command(name="reasoning_effort", description="Set reasoning effort (Admin only in servers)")
    async def reasoning_effort(self, interaction: discord.Interaction, effort: str):
        """Set reasoning effort for this context"""
        await interaction.response.defer(ephemeral=True)

        is_dm = isinstance(interaction.channel, discord.DMChannel)

        if not is_dm and not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use this command in servers!")
            return

        effort = (effort or "").lower().strip()
        allowed = {"minimal", "low", "medium", "high"}
        if effort not in allowed:
            await interaction.followup.send("Effort must be one of: minimal, low, medium, high.")
            return

        if is_dm:
            dm_reasoning_effort[interaction.user.id] = effort
            save_json_data(DM_REASONING_EFFORT_FILE, dm_reasoning_effort)
            await interaction.followup.send(f"✅ Reasoning effort set to **{effort}** for your DMs.")
        else:
            guild_reasoning_effort[interaction.guild.id] = effort
            save_json_data(REASONING_EFFORT_FILE, guild_reasoning_effort)
            await interaction.followup.send(f"✅ Reasoning effort set to **{effort}** for this server.")

    @app_commands.command(name="reasoning_stream_toggle", description="Toggle CLI reasoning stream (Admin only in servers)")
    async def reasoning_stream_toggle(self, interaction: discord.Interaction, enabled: bool = None):
        """Toggle CLI streaming indicator for reasoning runs"""
        await interaction.response.defer(ephemeral=True)

        is_dm = isinstance(interaction.channel, discord.DMChannel)

        if not is_dm and not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use this command in servers!")
            return

        if enabled is None:
            status = "enabled" if cli_reasoning_stream_enabled else "disabled"
            await interaction.followup.send(f"CLI reasoning stream is currently **{status}**.\nUse `/reasoning_stream_toggle true` or `/reasoning_stream_toggle false`.")
            return

        set_cli_reasoning_stream_enabled(enabled)
        await interaction.followup.send(f"✅ CLI reasoning stream {'enabled' if enabled else 'disabled'}.")


    @app_commands.command(name="embeddings_toggle", description="Toggle embeddings for this context (Admin only in servers)")
    async def embeddings_toggle(self, interaction: discord.Interaction, enabled: bool = None):
        """Toggle embeddings usage for this context"""
        await interaction.response.defer(ephemeral=True)

        is_dm = isinstance(interaction.channel, discord.DMChannel)

        if not is_dm and not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use this command in servers!")
            return

        if enabled is None:
            current_enabled, current_model = get_effective_embedding_settings(
                interaction.guild.id if interaction.guild else None,
                interaction.user.id if interaction.user else None,
                is_dm
            )
            source = "DM override" if is_dm and interaction.user.id in dm_embedding_settings else "server/default"
            await interaction.followup.send(
                f"Embeddings are currently **{'enabled' if current_enabled else 'disabled'}** ({source}).\n"
                f"Model: `{current_model}`\n"
                f"Use `/embeddings_toggle true` or `/embeddings_toggle false`."
            )
            return

        if is_dm:
            set_embedding_settings_for_dm(interaction.user.id, enabled=enabled)
            await interaction.followup.send(f"✅ Embeddings {'enabled' if enabled else 'disabled'} for your DMs.")
        else:
            set_embedding_settings_for_guild(interaction.guild.id, enabled=enabled)
            await interaction.followup.send(f"✅ Embeddings {'enabled' if enabled else 'disabled'} for this server.")

    @app_commands.command(name="embeddings_model_set", description="Set the embeddings model (Admin only in servers)")
    async def embeddings_model_set(self, interaction: discord.Interaction, model: str):
        """Set embeddings model for this context"""
        await interaction.response.defer(ephemeral=True)

        is_dm = isinstance(interaction.channel, discord.DMChannel)

        if not is_dm and not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use this command in servers!")
            return

        model = (model or "").strip()
        if not model:
            await interaction.followup.send("❌ Please provide a valid embeddings model id.")
            return

        if is_dm:
            set_embedding_settings_for_dm(interaction.user.id, model=model)
            await interaction.followup.send(f"✅ Embeddings model set to `{model}` for your DMs.")
        else:
            set_embedding_settings_for_guild(interaction.guild.id, model=model)
            await interaction.followup.send(f"✅ Embeddings model set to `{model}` for this server.")

    @app_commands.command(name="sync", description="Sync slash commands (Admin only in servers)")
    async def sync_commands(self, interaction: discord.Interaction):
        """Manually sync slash commands for this server"""
        await interaction.response.defer(ephemeral=True)

        if not interaction.guild:
            await interaction.followup.send("Sync can only be used in servers, not DMs.")
            return

        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use this command!")
            return

        try:
            await self.bot.tree.sync()
            await self.bot.tree.sync(guild=interaction.guild)
            await interaction.followup.send("✅ Slash commands synced for this server.")
        except Exception as e:
            await interaction.followup.send(f"❌ Sync failed: {e}")

    # HELP COMMANDS

    @app_commands.command(name="health", description="Show bot health diagnostics")
    async def health_command(self, interaction: discord.Interaction):
        """Show basic health and status information"""
        await interaction.response.defer(ephemeral=True)
    
        uptime = format_duration(time.time() - bot_start_time)
        latency_ms = int(client.latency * 1000)
        guild_count = len(client.guilds)
        channel_id = interaction.channel.id if interaction.channel else None
        summary_status = "Yes" if (channel_id and conversation_summaries.get(channel_id)) else "No"
        summary_count = len(conversation_summaries)
        loaded_count = len(loaded_plugins)
        error_count = len(plugin_errors)
    
        embed = discord.Embed(
            title="🤖 Bot Health",
            color=0x00ff99
        )
    
        embed.add_field(name="Uptime", value=uptime, inline=True)
        embed.add_field(name="Latency", value=f"{latency_ms} ms", inline=True)
        embed.add_field(name="Guilds", value=str(guild_count), inline=True)
        embed.add_field(name="Summaries", value=f"{summary_count} stored (current: {summary_status})", inline=False)
        embed.add_field(name="Plugins", value=f"{loaded_count} loaded, {error_count} errors", inline=False)
    
        await interaction.followup.send(embed=embed)


    @app_commands.command(name="queue", description="Show current request queue status (Admin only)")
    async def queue_command(self, interaction: discord.Interaction):
        """Show queue stats"""
        await interaction.response.defer(ephemeral=True)
    
        if interaction.guild and not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Administrator permissions required for `/queue`.")
            return
    
        total_pending = sum(len(v) for v in request_queue.queues.values())
        channel_id = interaction.channel.id if interaction.channel else None
        channel_pending = len(request_queue.queues.get(channel_id, [])) if channel_id else 0
        processing = request_queue.processing.get(channel_id, False) if channel_id else False
    
        embed = discord.Embed(
            title="📬 Queue Status",
            color=0xffcc00
        )
        embed.add_field(name="Total Pending", value=str(total_pending), inline=True)
        embed.add_field(name="This Channel Pending", value=str(channel_pending), inline=True)
        embed.add_field(name="Processing This Channel", value="Yes" if processing else "No", inline=True)
    
        await interaction.followup.send(embed=embed)

    # CONTEXT AWARENESS PLUGIN COMMANDS

    @app_commands.command(name="context_plugin_info", description="Show Context Awareness plugin settings (Admin only)")
    async def context_plugin_info(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        if not interaction.guild:
            await interaction.followup.send("This command can only be used in servers.")
            return
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use this command!")
            return

        defaults = {"enabled": True, "max_history": 20, "max_words": 120, "history_mode": "keep"}
        settings = get_plugin_guild_settings("context_awareness", interaction.guild.id, defaults)

        embed = discord.Embed(
            title="🧠 Context Awareness Plugin",
            color=0x0099ff
        )
        embed.add_field(name="Enabled", value=str(bool(settings.get("enabled", True))), inline=True)
        embed.add_field(name="Max History", value=str(settings.get("max_history", defaults["max_history"])), inline=True)
        embed.add_field(name="Max Words", value=str(settings.get("max_words", defaults["max_words"])), inline=True)
        embed.add_field(name="History Mode", value=str(settings.get("history_mode", defaults["history_mode"])), inline=True)
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="context_plugin_set", description="Configure Context Awareness plugin (Admin only)")
    @app_commands.describe(
        enabled="Enable or disable the context plugin",
        max_history="How many recent messages to scan (1-80)",
        max_words="Max words for the curated context (40-300)",
        history_mode="History handling: keep | trim | curated_only"
    )
    async def context_plugin_set(
        self,
        interaction: discord.Interaction,
        enabled: Optional[bool] = None,
        max_history: Optional[int] = None,
        max_words: Optional[int] = None,
        history_mode: Optional[str] = None
    ):
        await interaction.response.defer(ephemeral=True)
        if not interaction.guild:
            await interaction.followup.send("This command can only be used in servers.")
            return
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can use this command!")
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

        if not updates:
            await interaction.followup.send("No changes provided. Use at least one option.")
            return

        update_plugin_guild_settings("context_awareness", interaction.guild.id, updates)

        defaults = {"enabled": True, "max_history": 20, "max_words": 120, "history_mode": "keep"}
        settings = get_plugin_guild_settings("context_awareness", interaction.guild.id, defaults)
        await interaction.followup.send(
            f"✅ Updated Context Awareness plugin settings: "
            f"enabled={settings.get('enabled', True)}, "
            f"max_history={settings.get('max_history', defaults['max_history'])}, "
            f"max_words={settings.get('max_words', defaults['max_words'])}, "
            f"history_mode={settings.get('history_mode', defaults['history_mode'])}"
        )

    @app_commands.command(name="help", description="Show all available commands and how to use the bot")
    async def help_command(self, interaction: discord.Interaction):
        """Display comprehensive help information"""
        await interaction.response.defer(ephemeral=True)
    
        embed = discord.Embed(
            title="🤖 Bot Help Guide",
            description="Here are all the available commands and how to use this bot!\nCreated by marinara_spaghetti 🍝\nConsider supporting at https://ko-fi.com/spicy_marinara\n",
            color=0x00ff00
        )
    
        embed.add_field(
            name="❓ Basic Usage",
            value="• **Activate by mentions!** Mention the bot (@botname) to chat directly\n• **Remembers the chat!** Bot stores conversation history automatically\n• **Bot sees images, gifs and audio messages!** Bot can see the different media files you upload.\n• **Works in DMs!** Just message the bot directly anytime\n• **Reactions!** Bot can react to your messages with emojis",
            inline=False
        )
    
        embed.add_field(
            name="🤖 AI Model Commands",
            value="`/model_set <provider> [model]` - Set AI provider and model (Admin only)\n`/model_info` - Show current model settings\n`/temperature_set <value>` - Set AI creativity (Admin only)\n`/reasoning_toggle [true/false]` - Toggle reasoning (Admin only)\n`/reasoning_effort <minimal|low|medium|high>` - Set reasoning effort (Admin only)\n`/reasoning_stream_toggle [true/false]` - Toggle CLI reasoning stream (Admin only in servers)\n`/embeddings_toggle [true/false]` - Toggle embeddings (Admin only in servers)\n`/embeddings_model_set <model>` - Set embeddings model (Admin only in servers)\n`/dm_server_select [server]` - Choose which server's settings to use in DMs",
            inline=False
        )
    
        embed.add_field(
            name="🎭 Personality Commands",
            value="`/personality_create <name> <display_name> <prompt>` - Create custom personality for the bot (Admin only)\n`/personality_import <file> [name] [display_name]` - Import personality from .txt (Admin only)\n`/personality_lore_set <personality> [lore_text] [lore_file]` - Set character lore (Admin only)\n`/personality_set [name]` - Set/view the bot's active personality (Admin only)\n`/personality_list` - List all personalities of the bot\n`/personality_edit <name> [display_name] [prompt]` - Edit personality (Admin only)\n`/personality_delete <name>` - Delete personality (Admin only)",
            inline=False
        )
    
        embed.add_field(
            name="💬 Response Format Commands",
            value="`/format_set <style> [scope]` - Set response format with dropdown choices! (conversational/asterisk/narrative)\n`/format_info` - Show current format and available options\n`/format_view [type]` - View format instruction templates\n`/format_edit <type> [instructions]` - Edit format templates or reset to default (Admin only)\n`/nsfw_set <enabled> [scope]` - Enable/disable NSFW content\n`/nsfw_info` - Show current NSFW settings",
            inline=False
        )
    
        embed.add_field(
        name="⚙️ Configuration Commands",
        value="`/history_length [number]` - Set conversation memory (Admin only)\n"
              "`/autonomous_set <channel> <enabled> [chance]` - Set autonomous behavior\n"
              "`/autonomous_list` - List autonomous channel settings\n"
              "`/dm_enable [true/false]` - Enable/disable DMs for server members (Admin only)\n",
        inline=False
        )
    

        embed.add_field(
            name="???? Plugin Commands",
            value="`/context_plugin_info` - Show Context Awareness plugin settings (Admin only)\n"
                "`/context_plugin_set [enabled] [max_history] [max_words] [history_mode]` - Configure Context Awareness plugin (Admin only)",
            inline=False
        )

        embed.add_field(
            name="🛠️ Utility Commands",
            value="`/clear` - Clear conversation history on the specific channel/DM\n"
                "`/activity <type> <text>` - Set bot activity\n"
                "`/status_set <status>` - Set bot online status\n"
                "`/delete_messages <number>` - Delete bot's last N messages\n"
                "`/add_prefill <text>` - Add a prefill message for conversations\n"
                "`/clear_prefill` - Remove the prefill message",
            inline=False
        )

        embed.add_field(
            name="ðŸ©º Diagnostics",
            value="`/health` - Bot health and uptime\n"
                "`/debug_history` - Show stored history (Admin only)\n"
                "`/queue` - Request queue status (Admin only)",
            inline=False
        )

        embed.add_field(
            name="💝 Fun Commands",
            value="`/kiss` - Give the bot a kiss and see how they react\n"
                "`/hug` - Give the bot a warm hug\n`"
                "/joke` - Ask the bot to tell you a joke\n"
                "`/bonk` - Bonk the bot's head\n"
                "`/bite` - Bite the bot\n"
                "`/affection` - Find out how much the bot likes you!",
            inline=False
        )

        embed.add_field(
        name="📚 Lore Commands (Context-Aware)",
        value="**Adding information about users, works in both servers and DMs!**\n"
            "`/lore_add [member] <lore>` - Add lore information about a specific user or yourself\n"
            "`/lore_edit [member] <new_lore>` - Edit existing lore\n"
            "`/lore_view [member]` - View lore entry\n"
            "`/lore_remove [member]` - Remove lore\n"
            "`/lore_list` - Show all lore entries\n"
            "`/lore_auto_update [member]` - Let the bot update lore based on conversations (Admin only)",
        inline=False
        )

        embed.add_field(
            name="🧠 Memory Commands (Context-Aware)",
            value="**Memories of conversations, works in both servers and DMs with separate storages!**\n"
                "`/memory_generate <num_messages>` - Generate memory summary\n"
                "`/memory_save <summary>` - Save a memory manually\n"
                "`/memory_list` - View all saved memories\n"
                "`/memory_view <number>` - View specific memory\n"
                "`/memory_edit <number> <new_summary>` - Edit specific memory\n"
                "`/memory_delete <number>` - Delete specific memory\n"
                "`/memory_clear` - Delete all memories",
            inline=False
        )

        embed.add_field(
            name="🔒 DM-Specific Commands",
            value="`/dm_server_select [server]` - Choose which server's settings to use in DMs\n"
                "`/dm_toggle [enabled]` - Toggle auto check-up messages (6+ hour reminder)\n"
                "`/dm_personality_list` - View personalities from your shared servers\n"
                "`/dm_personality_set [server_name]` - Choose server's personality for DMs\n"
                "`/dm_history_toggle [enabled]` - Toggle full DM history loading\n"
                "`/dm_regenerate` - Regenerate bot's last response\n"
                "`/dm_edit_last <new_message>` - Edit bot's last message",
            inline=False
        )
    
        embed.set_footer(text="💡 Many commands are context-aware and work differently in servers vs DMs!\n🔒 No logs stored, your privacy is respected!\n🤖 Supports Claude, Gemini, OpenAI, and custom providers!")
    
        await interaction.followup.send(embed=embed)

    # CUSTOM PROMPT COMMANDS REMOVED
    # All prompt-related commands have been removed as part of the restructuring

    # FORMAT COMMANDS

    @app_commands.command(name="format_set", description="Set conversation format style")
    @app_commands.describe(
        style="Choose your format style",
        scope="Where to apply this setting (only needed for servers, not DMs)"
    )
    @app_commands.choices(style=[
        app_commands.Choice(name="Conversational - Normal Discord chat", value="conversational"),
        app_commands.Choice(name="Asterisk - Roleplay with *actions*", value="asterisk"), 
        app_commands.Choice(name="Narrative - Rich storytelling format", value="narrative")
    ])
    @app_commands.choices(scope=[
        app_commands.Choice(name="Channel - Apply to this channel only", value="channel"),
        app_commands.Choice(name="Server - Apply as server default (Admin only)", value="server")
    ])
    async def set_format_style(self, interaction: discord.Interaction, style: str, scope: str = None):
        """Set conversation format with automatic DM/server detection and scope options"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        style = style.lower()
    
        # Get valid format styles
        valid_styles = ["conversational", "asterisk", "narrative"]
    
        # Check if style is valid
        if style not in valid_styles:
            available_list = ["conversational", "asterisk", "narrative"]
        
            await interaction.followup.send(f"❌ **Invalid format style!**\n\n"
                                           f"**Available styles:** {', '.join(available_list)}\n"
                                           f"**Usage:** `/format_set <style>` {'(DMs)' if is_dm else '[scope]'}")
            return
    
        # Get style descriptions
        style_descriptions = {
            "conversational": "Normal Discord chat (no roleplay actions)",
            "asterisk": "Roleplay with *action descriptions*",
            "narrative": "Rich, story-driven narrative roleplay"
        }
    
        style_description = style_descriptions.get(style, "Custom format style")
    
        if is_dm:
            dm_format_settings[interaction.user.id] = style
            save_json_data(DM_FORMAT_SETTINGS_FILE, dm_format_settings)
        
            await interaction.followup.send(f"✅ **Your DM format style set to {style.title()}!**\n"
                                        f"**Description:** {style_description}\n\n"
                                        f"💬 This setting applies to all your DMs with the bot.")
        else:
            # Server - handle scope options
            if scope is None:
                await interaction.followup.send(f"❌ **Please specify scope!**\n\n"
                                               f"**Valid scopes:** `channel` or `server`\n\n"
                                               f"**Examples:**\n"
                                               f"• `/format_set {style} channel` - Set for this channel only\n"
                                               f"• `/format_set {style} server` - Set for entire server (Admin only)")
                return
        
            # Validate scope
            scope = scope.lower()
            if scope not in ["channel", "server"]:
                await interaction.followup.send(f"❌ **Invalid scope!**\n\n"
                                               f"**Valid scopes:** `channel` or `server`")
                return
        
            if scope == "channel":
                # Set for current channel
                channel_format_settings[interaction.channel.id] = style
                save_json_data(FORMAT_SETTINGS_FILE, channel_format_settings)
            
                await interaction.followup.send(f"✅ **Format style set to {style.title()} for #{interaction.channel.name}!**\n"
                                               f"**Description:** {style_description}")
        
            elif scope == "server":
                # Check admin permissions for server-wide changes
                if not check_admin_permissions(interaction):
                    await interaction.followup.send(f"❌ **Administrator permissions required!**\n\n"
                                                   f"You need administrator permissions to set server-wide format styles.\n"
                                                   f"💡 You can still use `/format_set {style} channel` to set the format for this channel only.")
                    return
            
                # Set server default and save to persistent storage
                server_format_defaults[interaction.guild.id] = style
                save_json_data(SERVER_FORMAT_DEFAULTS_FILE, server_format_defaults)
            
                await interaction.followup.send(f"✅ **Server default format style set to {style.title()}!**\n"
                                               f"**Description:** {style_description}\n\n"
                                               f"🏰 This applies to all channels without specific format settings.")

    @app_commands.command(name="format_info", description="Show the current format style for this channel or DM")
    async def show_format_info(self, interaction: discord.Interaction):
        """Display current format style and available options"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            current_style = dm_format_settings.get(interaction.user.id, "conversational")
            title_prefix = "🔒 DM"
            embed_description = f"Current format: **{current_style.title()}**"
        else:
            # Check for channel-specific setting first
            channel_style = channel_format_settings.get(interaction.channel.id)
        
            if channel_style:
                current_style = channel_style
                style_source = f"Channel-specific (#{interaction.channel.name})"
            else:
                # Check for persistent server default
                server_style = server_format_defaults.get(interaction.guild.id)
            
                if server_style:
                    current_style = server_style
                    style_source = f"Server default ({interaction.guild.name})"
                else:
                    current_style = "conversational"
                    style_source = "Global default (not set)"
        
            title_prefix = "📢 Channel"
            embed_description = f"Current format: **{current_style.title()}**\nSource: {style_source}"
    
        # Create embed
        embed = discord.Embed(
            title=f"{title_prefix} Format Info",
            description=embed_description,
            color=0x00ff00
        )
    
        # Built-in format styles
        embed.add_field(
            name="📝 Available Format Styles",
            value="**`conversational`** - Normal Discord chat (no roleplay actions)\n"
                  "**`asterisk`** - Roleplay with *action descriptions*\n"
                  "**`narrative`** - Rich, story-driven narrative roleplay",
            inline=False
        )
    
        # Add footer
        if is_dm:
            embed.set_footer(text="Use /format_set <style> to change your DM format style!")
        else:
            embed.set_footer(text="Use /format_set <style> <scope> to change format style")
    
        await interaction.followup.send(embed=embed)

    # PERSONALITY COMMANDS

    # CONVERSATION STYLE COMMANDS

    @app_commands.command(name="format_edit", description="Edit format instruction templates (Admin only)")
    async def edit_format_instructions(self, interaction: discord.Interaction, format_type: str, instructions: str = None):
        """Edit format instruction templates for conversational, asterisk, or narrative styles"""
        await interaction.response.defer(ephemeral=True)
    
        # Check admin permissions
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can edit format instructions!")
            return
    
        # Validate format type
        format_type = format_type.lower()
        if format_type not in VALID_FORMAT_STYLES:
            await interaction.followup.send(f"❌ **Invalid format type!**\n\n"
                                           f"**Valid types:** {', '.join(VALID_FORMAT_STYLES)}")
            return
    
        # Handle reset to default
        if instructions is None or instructions.strip() == "":
            if format_type in custom_format_instructions:
                del custom_format_instructions[format_type]
                save_custom_format_instructions()
            
                embed = discord.Embed(
                    title="🔄 Format Instructions Reset",
                    description=f"Successfully reset **{format_type.title()}** format instructions to default!",
                    color=0xffaa00
                )
            
                embed.add_field(
                    name=f"📝 Format Type: {format_type.title()}",
                    value="Now using built-in default instructions.",
                    inline=False
                )
            else:
                embed = discord.Embed(
                    title="ℹ️ No Custom Instructions",
                    description=f"**{format_type.title()}** format is already using default instructions.",
                    color=0x888888
                )
        else:
            # Validate instructions length
            if not (10 <= len(instructions) <= 1000):
                await interaction.followup.send("❌ Format instructions must be between 10 and 1000 characters.")
                return
        
            # Save the custom format instructions
            custom_format_instructions[format_type] = instructions
            save_custom_format_instructions()
        
            embed = discord.Embed(
                title="✅ Format Instructions Updated",
                description=f"Successfully updated **{format_type.title()}** format instructions!",
                color=0x00ff00
            )
        
            embed.add_field(
                name=f"📝 Format Type: {format_type.title()}",
                value=f"**New Instructions:**\n{instructions}",
                inline=False
            )
    
        embed.set_footer(text="Changes will take effect immediately for new responses")
        await interaction.followup.send(embed=embed)

    # Autocomplete for format_edit command
    @edit_format_instructions.autocomplete('format_type')
    async def format_type_autocomplete(self, 
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Provide autocomplete options for format types"""
        choices = [
            app_commands.Choice(name="Conversational", value="conversational"),
            app_commands.Choice(name="Asterisk Roleplay", value="asterisk"), 
            app_commands.Choice(name="Narrative Roleplay", value="narrative")
        ]
    
        # Filter by current input
        if current:
            choices = [choice for choice in choices if current.lower() in choice.name.lower()]
    
        return choices

    @app_commands.command(name="format_view", description="View current format instruction templates")
    async def view_format_instructions(self, interaction: discord.Interaction, format_type: str = None):
        """View format instruction templates for all or specific format styles"""
        await interaction.response.defer(ephemeral=True)
    
        if format_type:
            format_type = format_type.lower()
            if format_type not in VALID_FORMAT_STYLES:
                await interaction.followup.send(f"❌ **Invalid format type!**\n\n"
                                               f"**Valid types:** {', '.join(VALID_FORMAT_STYLES)}")
                return
    
        # Get current format instructions (custom or default)
        format_instructions = {}
        for fmt_type in VALID_FORMAT_STYLES:
            if fmt_type in custom_format_instructions:
                format_instructions[fmt_type] = f"**Custom:** {custom_format_instructions[fmt_type]}"
            else:
                # Default instructions
                if fmt_type == "conversational":
                    format_instructions[fmt_type] = "Respond with up to one sentence, adapting internet language. You can reply with just one word or emoji, if you choose to. Avoid using asterisks and em-dashes. Do not repeat after yourself or others. If the user's message is unclear or very short, ask a brief clarifying question or invite detail instead of repeating. Avoid stock phrases or reusing distinctive lines from your persona."
                elif fmt_type == "asterisk":
                    format_instructions[fmt_type] = "Respond with one to three short paragraphs of asterisk roleplay. Enclose actions and descriptions in *asterisks*, while keeping dialogues as plain text. Avoid using em-dashes and nested asterisks; they break the formatting. Do not repeat after yourself or others. If the user's message is unclear or very short, ask a brief clarifying question or invite detail instead of repeating. Avoid stock phrases or reusing distinctive lines from your persona. Be creative."
                elif fmt_type == "narrative":
                    format_instructions[fmt_type] = "Respond with one to four short paragraphs of narrative roleplay. Use plain text for the narration and \"quotation marks\" for dialogues. Avoid using em-dashes and asterisks. Do not repeat after yourself or others. If the user's message is unclear or very short, ask a brief clarifying question or invite detail instead of repeating. Avoid stock phrases or reusing distinctive lines from your persona. Be creative. Show, don't tell."
    
        embed = discord.Embed(
            title="📋 Format Instruction Templates",
            color=0x00ff00
        )
    
        if format_type:
            # Show specific format
            embed.description = f"**{format_type.title()} Format Instructions:**"
            embed.add_field(
                name=f"📝 {format_type.title()} Style", 
                value=format_instructions[format_type],
                inline=False
            )
        else:
            # Show all formats
            embed.description = "Current format instruction templates:"
        
            for fmt_type in VALID_FORMAT_STYLES:
                embed.add_field(
                    name=f"📝 {fmt_type.title()} Style",
                    value=format_instructions[fmt_type],
                    inline=False
                )
    
        embed.add_field(
            name="💡 Note",
            value="**Custom** instructions are marked with 'Custom:' prefix. Use `/format_edit` to customize them (Admin only).",
            inline=False
        )
    
        embed.set_footer(text="Use /format_set to change your format style • /format_info for current settings")
        await interaction.followup.send(embed=embed)

    # Autocomplete for format_view command
    @view_format_instructions.autocomplete('format_type')
    async def format_view_autocomplete(self, 
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Provide autocomplete options for format types"""
        choices = [
            app_commands.Choice(name="Conversational", value="conversational"),
            app_commands.Choice(name="Asterisk Roleplay", value="asterisk"), 
            app_commands.Choice(name="Narrative Roleplay", value="narrative")
        ]
    
        # Filter by current input
        if current:
            choices = [choice for choice in choices if current.lower() in choice.name.lower()]
    
        return choices

    # NSFW COMMANDS

    @app_commands.command(name="nsfw_set", description="Enable or disable NSFW content for this server/DM")
    @app_commands.describe(
        enabled="Enable (True) or disable (False) NSFW content",
        scope="Server scope (channel/server) - only for servers, not DMs"
    )
    async def set_nsfw(self, 
        interaction: discord.Interaction,
        enabled: bool,
        scope: str = None
    ):
        """Enable or disable NSFW content with automatic DM/server detection"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            # DM - simple NSFW setting (scope not needed)
            if scope is not None:
                await interaction.followup.send(f"💡 **DM Mode**: You don't need to specify a scope in DMs.\n"
                                            f"Setting your DM NSFW preference...")
        
            dm_nsfw_settings[interaction.user.id] = enabled
            save_json_data(DM_NSFW_SETTINGS_FILE, dm_nsfw_settings)
        
            status = "enabled" if enabled else "disabled"
            emoji = "🔞" if enabled else "✅"
        
            await interaction.followup.send(f"{emoji} **NSFW content {status} for your DMs!**\n\n"
                                          f"💬 This setting applies to all your DMs with the bot.\n"
                                          f"🎭 Use `/format_set` to choose your conversation style.")
        else:
            # Server - handle scope options
            if scope is None:
                # Show scope selection instead of executing
                status_text = "enable" if enabled else "disable"
                embed = discord.Embed(
                    title="🔞 Choose NSFW Scope",
                    description=f"You want to **{status_text}** NSFW content.\n**Please specify the scope:**",
                    color=0xffaa00 if enabled else 0x00ff00
                )
            
                embed.add_field(
                    name="📢 For This Channel Only",
                    value=f"**Command:** `/nsfw_set {enabled} channel`\n"
                          f"**Effect:** {status_text.title()}s NSFW for **#{interaction.channel.name}** only",
                    inline=False
                )
            
                embed.add_field(
                    name="🏰 For Entire Server",
                    value=f"**Command:** `/nsfw_set {enabled} server`\n"
                          f"**Effect:** {status_text.title()}s NSFW for **all channels** in this server\n"
                          f"*(Requires admin permissions)*",
                    inline=False
                )
            
                embed.add_field(
                    name="⚠️ NSFW Warning",
                    value="NSFW mode allows: profanities, dark themes, obscene jokes, adult content, "
                          "controversial opinions, and gore. Only enable if all users are adults.",
                    inline=False
                )
            
                embed.set_footer(text="⚠️ Please run the command again with your chosen scope")
                await interaction.followup.send(embed=embed)
                return
        
            # Validate scope
            scope = scope.lower()
            if scope not in ["channel", "server"]:
                await interaction.followup.send(f"❌ **Invalid scope!**\n\n"
                                               f"**Valid scopes:** `channel` or `server`\n\n"
                                               f"**Examples:**\n"
                                               f"• `/nsfw_set {enabled} channel` - Set for this channel only\n"
                                               f"• `/nsfw_set {enabled} server` - Set for entire server (Admin only)")
                return
        
            status = "enabled" if enabled else "disabled"
            emoji = "🔞" if enabled else "✅"
        
            if scope == "channel":
                # For now, channel-specific NSFW isn't implemented, use server-wide
                await interaction.followup.send("⚠️ **Channel-specific NSFW not yet implemented.**\n\n"
                                               "NSFW settings currently apply server-wide only.\n"
                                               f"Use `/nsfw_set {enabled} server` instead.")
                return
        
            elif scope == "server":
                # Check admin permissions for server-wide changes
                if not check_admin_permissions(interaction):
                    await interaction.followup.send(f"❌ **Administrator permissions required!**\n\n"
                                                   f"You need administrator permissions to change server NSFW settings.")
                    return
            
                # Set server NSFW setting
                guild_nsfw_settings[interaction.guild.id] = enabled
                save_json_data(NSFW_SETTINGS_FILE, guild_nsfw_settings)
            
                await interaction.followup.send(f"{emoji} **NSFW content {status} for this server!**\n\n"
                                              f"🏰 This applies to all channels in the server.\n"
                                              f"🎭 Use `/format_set` to choose your conversation style.\n"
                                              f"💾 This setting is saved and will persist between bot restarts!")

    @app_commands.command(name="nsfw_info", description="Show current NSFW settings")
    async def nsfw_info(self, interaction: discord.Interaction):
        """Show current NSFW settings for this context"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            current_setting = dm_nsfw_settings.get(interaction.user.id, False)
            title_prefix = "🔒 DM"
            scope_text = "your DMs"
        else:
            current_setting = guild_nsfw_settings.get(interaction.guild.id, False)
            title_prefix = "📢 Server"
            scope_text = f"**{interaction.guild.name}**"
    
        status = "Enabled 🔞" if current_setting else "Disabled ✅"
        color = 0xff4444 if current_setting else 0x00ff00
    
        embed = discord.Embed(
            title=f"🔞 {title_prefix} NSFW Settings",
            description=f"NSFW content for {scope_text}: **{status}**",
            color=color
        )
    
        if current_setting:
            embed.add_field(
                name="🔞 NSFW Content Enabled",
                value="The bot can use profanities, dark themes, obscene jokes, "
                      "adult content, controversial opinions, and gore.",
                inline=False
            )
        else:
            embed.add_field(
                name="✅ Safe Content Only",
                value="The bot will maintain appropriate, family-friendly content.",
                inline=False
            )
    
        embed.add_field(
            name="💡 How to Change",
            value=f"Use `/nsfw_set true` to enable or `/nsfw_set false` to disable NSFW content.\n"
                  f"{'In servers, add `server` scope for server-wide changes.' if not is_dm else ''}",
            inline=False
        )
    
        embed.set_footer(text="NSFW settings work with all conversation styles")
        await interaction.followup.send(embed=embed)

    # PERSONALITY COMMANDS
    # PERSONALITY COMMANDS

    @app_commands.command(name="personality_create", description="Create a new personality for the bot (Admin only)")
    async def create_personality(self, interaction: discord.Interaction, name: str, display_name: str, personality_prompt: str):
        """Create new custom personality for server"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("Personalities can only be created in servers, not DMs.")
            return
    
        # Check admin permissions
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can create personalities!")
            return
    
        # Validate input parameters
        if not (2 <= len(name) <= 32):
            await interaction.followup.send("Personality name must be between 2 and 32 characters.")
            return
    
        if not (2 <= len(display_name) <= 64):
            await interaction.followup.send("Display name must be between 2 and 64 characters.")
            return
    
        if not (10 <= len(personality_prompt) <= 8192):
            await interaction.followup.send("Personality prompt must be between 10 and 8192 characters.")
            return
    
        clean_name = name.lower().replace(" ", "_")
    
        # Initialize guild's custom personalities if needed
        if interaction.guild.id not in custom_personalities:
            custom_personalities[interaction.guild.id] = {}
    
        # Check for existing personality
        if clean_name in custom_personalities[interaction.guild.id]:
            await interaction.followup.send(f"Personality '{clean_name}' already exists! Use `/personality_edit` to modify it.")
            return
    
        # Create personality
        custom_personalities[interaction.guild.id][clean_name] = {
            "name": display_name,
            "prompt": personality_prompt
        }
    
        save_personalities()
    
        prompt_preview = personality_prompt[:100] + ('...' if len(personality_prompt) > 100 else '')
        await interaction.followup.send(f"✅ Created personality **{display_name}** (`{clean_name}`)!\n"
                                       f"Use `/personality_set {clean_name}` to activate it.\n\n"
                                       f"**Prompt preview:** {prompt_preview}")

    @app_commands.command(name="personality_import", description="Import a personality from a .txt file (Admin only)")
    async def import_personality(self, interaction: discord.Interaction, file: discord.Attachment, name: str = None, display_name: str = None):
        """Import a personality from a text file"""
        await interaction.response.defer(ephemeral=True)

        if not interaction.guild:
            await interaction.followup.send("Personalities can only be imported in servers, not DMs.")
            return

        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can import personalities!")
            return

        if not file or not file.filename.lower().endswith(".txt"):
            await interaction.followup.send("❌ Please upload a valid .txt file.")
            return

        # Read file content
        try:
            raw_bytes = await file.read()
            text = raw_bytes.decode("utf-8", errors="ignore").strip()
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to read file: {str(e)}")
            return

        if not text:
            await interaction.followup.send("❌ The file is empty.")
            return

        # Derive defaults from filename if not provided
        filename_base = os.path.splitext(file.filename)[0]
        derived_name = filename_base.lower().replace(" ", "_") if filename_base else "imported_personality"

        if name is None or not name.strip():
            name = derived_name
        if display_name is None or not display_name.strip():
            display_name = filename_base if filename_base else "Imported Personality"

        # Validate input parameters
        if not (2 <= len(name) <= 32):
            await interaction.followup.send("Personality name must be between 2 and 32 characters.")
            return

        if not (2 <= len(display_name) <= 64):
            await interaction.followup.send("Display name must be between 2 and 64 characters.")
            return

        if not (10 <= len(text) <= 8192):
            await interaction.followup.send("Personality prompt must be between 10 and 8192 characters.")
            return

        clean_name = name.lower().replace(" ", "_")

        if interaction.guild.id not in custom_personalities:
            custom_personalities[interaction.guild.id] = {}

        if clean_name in custom_personalities[interaction.guild.id]:
            await interaction.followup.send(f"Personality '{clean_name}' already exists! Use `/personality_edit` to modify it.")
            return

        custom_personalities[interaction.guild.id][clean_name] = {
            "name": display_name,
            "prompt": text
        }

        save_personalities()

        prompt_preview = text[:100] + ('...' if len(text) > 100 else '')
        await interaction.followup.send(
            f"✅ Imported personality **{display_name}** (`{clean_name}`) from `{file.filename}`!\n"
            f"Use `/personality_set {clean_name}` to activate it.\n\n"
            f"**Prompt preview:** {prompt_preview}"
        )

    @app_commands.command(name="personality_lore_set", description="Set character lore for a personality (Admin only)")
    async def personality_lore_set(self, interaction: discord.Interaction, personality_name: str, lore_text: str = None, lore_file: discord.Attachment = None):
        """Set character lore for a personality"""
        await interaction.response.defer(ephemeral=True)

        if not interaction.guild:
            await interaction.followup.send("Character lore can only be set in servers, not DMs.")
            return

        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can set character lore!")
            return

        clean_name = personality_name.lower().replace(" ", "_")
        if interaction.guild.id not in custom_personalities or clean_name not in custom_personalities[interaction.guild.id]:
            await interaction.followup.send(f"Personality '{clean_name}' not found! Use `/personality_list` to see available personalities.")
            return

        lore = ""
        if lore_file is not None:
            if not lore_file.filename.lower().endswith(".txt"):
                await interaction.followup.send("❌ Please upload a valid .txt file for lore.")
                return
            try:
                raw_bytes = await lore_file.read()
                lore = raw_bytes.decode("utf-8", errors="ignore").strip()
            except Exception as e:
                await interaction.followup.send(f"❌ Failed to read file: {str(e)}")
                return
        elif lore_text is not None:
            lore = lore_text.strip()

        if not lore:
            await interaction.followup.send("❌ Provide lore text or a .txt file.")
            return

        if not (10 <= len(lore) <= 8192):
            await interaction.followup.send("Character lore must be between 10 and 8192 characters.")
            return

        custom_personalities[interaction.guild.id][clean_name]["lore"] = lore
        save_personalities()

        lore_preview = lore[:120] + ('...' if len(lore) > 120 else '')
        await interaction.followup.send(
            f"✅ Character lore set for **{custom_personalities[interaction.guild.id][clean_name]['name']}** (`{clean_name}`)!\n"
            f"**Lore preview:** {lore_preview}"
        )

    @app_commands.command(name="personality_set", description="Set the active personality for this server (Admin only)")
    async def set_personality(self, interaction: discord.Interaction, personality_name: str = None):
        """Set active personality for server or show current/available personalities"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("Personality can only be set in servers, not DMs.\n"
                                           "💡 **Note:** In DMs, the bot uses the personality from a server you both share!")
            return
    
        if personality_name is None:
            # Show current personality and available options
            current_name = guild_personalities.get(interaction.guild.id, "default")
            current_display = get_personality_name(interaction.guild.id)
        
            available = ["default"]
            if interaction.guild.id in custom_personalities:
                available.extend(custom_personalities[interaction.guild.id].keys())
        
            embed = discord.Embed(
                title="🎭 Bot Personalities",
                description=f"**Current:** {current_display} (`{current_name}`)",
                color=0x9932cc
            )
        
            embed.add_field(
                name="Available Personalities",
                value="\n".join([f"• `{name}`" for name in available]),
                inline=False
            )
        
            embed.set_footer(text="Use /personality_set <name> to change personality (Admin only)\n💡 This personality is also used in DMs!")
            await interaction.followup.send(embed=embed)
            return
    
        # Check admin permissions for setting personality
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can set personalities!")
            return
    
        clean_name = personality_name.lower().replace(" ", "_")
    
        # Set personality based on availability
        if clean_name == "default":
            guild_personalities[interaction.guild.id] = "default"
            display_name = DEFAULT_PERSONALITIES["default"]["name"]
        elif interaction.guild.id in custom_personalities and clean_name in custom_personalities[interaction.guild.id]:
            guild_personalities[interaction.guild.id] = clean_name
            display_name = custom_personalities[interaction.guild.id][clean_name]["name"]
        else:
            available = ["default"]
            if interaction.guild.id in custom_personalities:
                available.extend(custom_personalities[interaction.guild.id].keys())
            await interaction.followup.send(f"Personality '{clean_name}' not found!\nAvailable: {', '.join(available)}")
            return
    
        save_personalities()
        await interaction.followup.send(f"✅ Bot personality set to **{display_name}** (`{clean_name}`)!\n"
                                       f"💡 This personality will also be used in DMs with server members.")

    @app_commands.command(name="personality_list", description="List all personalities for this server")
    async def list_personalities(self, interaction: discord.Interaction):
        """Display all available personalities for server"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("Personality list can only be viewed in servers.")
            return
    
        embed = discord.Embed(
            title="🎭 Server Personalities",
            description=f"Personalities available in {interaction.guild.name}:",
            color=0x9932cc
        )
    
        current_personality = guild_personalities.get(interaction.guild.id, "default")
    
        # Add default personality
        default_marker = " ← **ACTIVE**" if current_personality == "default" else ""
        embed.add_field(
            name=f"default{default_marker}",
            value=f"**{DEFAULT_PERSONALITIES['default']['name']}**\n{DEFAULT_PERSONALITIES['default']['prompt'][:100]}...",
            inline=False
        )
    
        # Add custom personalities
        if interaction.guild.id in custom_personalities:
            for name, data in custom_personalities[interaction.guild.id].items():
                active_marker = " ← **ACTIVE**" if current_personality == name else ""
                prompt_preview = data['prompt'][:100] + ('...' if len(data['prompt']) > 100 else '')
                embed.add_field(
                    name=f"{name}{active_marker}",
                    value=f"**{data['name']}**\n{prompt_preview}",
                    inline=False
                )
    
        if len(embed.fields) == 1:
            embed.add_field(
                name="No Custom Personalities",
                value="Use `/personality_create` to add custom personalities! (Admin only)",
                inline=False
            )
    
        embed.set_footer(text="💡 The active personality is also used in DMs with server members!")
    
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="personality_edit", description="Edit an existing personality (Admin only)")
    async def edit_personality(self, interaction: discord.Interaction, personality_name: str, display_name: str = None, personality_prompt: str = None):
        """Edit existing custom personality"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("Personalities can only be edited in servers.")
            return
    
        # Check admin permissions
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can edit personalities!")
            return
    
        clean_name = personality_name.lower().replace(" ", "_")
    
        if clean_name == "default":
            await interaction.followup.send("Cannot edit the default personality. Create a custom one instead!")
            return
    
        # Check if personality exists
        if interaction.guild.id not in custom_personalities or clean_name not in custom_personalities[interaction.guild.id]:
            await interaction.followup.send(f"Personality '{clean_name}' not found! Use `/personality_list` to see available personalities.")
            return
    
        # Update provided fields
        updated_fields = []
    
        if display_name is not None:
            if not (2 <= len(display_name) <= 64):
                await interaction.followup.send("Display name must be between 2 and 64 characters.")
                return
            custom_personalities[interaction.guild.id][clean_name]["name"] = display_name
            updated_fields.append(f"Display name → {display_name}")
    
        if personality_prompt is not None:
            if not (10 <= len(personality_prompt) <= 8192):
                await interaction.followup.send("Personality prompt must be between 10 and 8192 characters.")
                return
            custom_personalities[interaction.guild.id][clean_name]["prompt"] = personality_prompt
            prompt_preview = personality_prompt[:50] + ('...' if len(personality_prompt) > 50 else '')
            updated_fields.append(f"Prompt → {prompt_preview}")
    
        if not updated_fields:
            await interaction.followup.send("No changes specified! Provide display_name and/or personality_prompt to edit.")
            return
    
        save_personalities()
        await interaction.followup.send(f"✅ Updated personality **{clean_name}**:\n" + "\n".join(updated_fields))

    @app_commands.command(name="personality_delete", description="Delete a custom personality (Admin only)")
    async def delete_personality(self, interaction: discord.Interaction, personality_name: str):
        """Delete custom personality"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("Personalities can only be deleted in servers.")
            return
    
        # Check admin permissions
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can delete personalities!")
            return
    
        clean_name = personality_name.lower().replace(" ", "_")
    
        if clean_name == "default":
            await interaction.followup.send("Cannot delete the default personality!")
            return
    
        # Check if personality exists
        if interaction.guild.id not in custom_personalities or clean_name not in custom_personalities[interaction.guild.id]:
            await interaction.followup.send(f"Personality '{clean_name}' not found!")
            return
    
        # Reset to default if this personality is currently active
        if guild_personalities.get(interaction.guild.id) == clean_name:
            guild_personalities[interaction.guild.id] = "default"
    
        # Delete personality
        display_name = custom_personalities[interaction.guild.id][clean_name]["name"]
        del custom_personalities[interaction.guild.id][clean_name]
    
        save_personalities()
        await interaction.followup.send(f"🗑️ Deleted personality **{display_name}** (`{clean_name}`)!")

    # CONFIGURATION COMMANDS

    @app_commands.command(name="history_length", description="Set how many messages to keep in conversation history for this server (Admin only)")
    async def set_history_length(self, interaction: discord.Interaction, length: int = None):
        """Configure conversation history length for server"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("History length can only be set in servers, not DMs.")
            return
    
        if length is None:
            current = get_history_length(interaction.guild.id)
            await interaction.followup.send(f"Current server history length: {current} messages\nUsage: `/history_length <number>` (Admin only)")
            return
    
        # Check admin permissions
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can set history length!")
            return
    
        if not (1 <= length <= 1000):
            await interaction.followup.send("History length must be between 1 and 1000 messages.")
            return
    
        guild_history_lengths[interaction.guild.id] = length
        save_json_data(HISTORY_LENGTHS_FILE, guild_history_lengths)
        await interaction.followup.send(f"Server history length set to {length} messages! 📚")

    @app_commands.command(name="autonomous_set", description="Set autonomous response behavior for a channel")
    async def set_autonomous(self, interaction: discord.Interaction, channel: discord.TextChannel, enabled: bool, chance: int = 10):
        """Configure autonomous response behavior for specific channel"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("Autonomous settings can only be configured in servers.")
            return
    
        if not (1 <= chance <= 100):
            await interaction.followup.send("Chance must be between 1 and 100 percent.")
            return
    
        autonomous_manager.set_channel_autonomous(interaction.guild.id, channel.id, enabled, chance)
    
        if enabled:
            await interaction.followup.send(f"✅ Autonomous responses **enabled** in {channel.mention}\n"
                                           f"Response chance: **{chance}%**\n"
                                           f"The bot will randomly participate in conversations!")
        else:
            await interaction.followup.send(f"❌ Autonomous responses **disabled** in {channel.mention}")

    @app_commands.command(name="autonomous_list", description="List all channels with autonomous behavior configured")
    async def list_autonomous(self, interaction: discord.Interaction):
        """Display all channels with autonomous behavior settings"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("Autonomous settings can only be viewed in servers.")
            return
    
        channels = autonomous_manager.list_autonomous_channels(interaction.guild.id)
    
        if not channels:
            await interaction.followup.send("No autonomous channels configured for this server.\n"
                                           "Use `/autonomous_set` to configure channels!")
            return
    
        embed = discord.Embed(
            title="🤖 Autonomous Channel Settings",
            description="Channels where the bot can respond autonomously:",
            color=0x00ff00
        )
    
        for channel_id, settings in channels.items():
            channel = interaction.guild.get_channel(channel_id)
            channel_name = channel.mention if channel else f"Unknown Channel ({channel_id})"
        
            status = "✅ Enabled" if settings["enabled"] else "❌ Disabled"
            chance = settings["chance"]
        
            embed.add_field(
                name=channel_name,
                value=f"{status}\nChance: {chance}%",
                inline=True
            )
    
        await interaction.followup.send(embed=embed)

    # BOT MANAGEMENT COMMANDS

    @app_commands.command(name="activity", description="Set bot's activity status.")
    async def set_activity(self, interaction: discord.Interaction, activity_type: str, status_text: str):
        """Set bot's Discord activity status"""
        await interaction.response.defer(ephemeral=True)
    
        activity_map = {
            "playing": lambda text: discord.Game(name=text),
            "watching": lambda text: discord.Activity(type=discord.ActivityType.watching, name=text),
            "listening": lambda text: discord.Activity(type=discord.ActivityType.listening, name=text),
            "streaming": lambda text: discord.Streaming(name=text, url="https://twitch.tv/placeholder"),
            "competing": lambda text: discord.Activity(type=discord.ActivityType.competing, name=text)
        }
    
        activity_type = activity_type.lower()
        if activity_type not in activity_map:
            await interaction.followup.send(f"Invalid activity type! Use: {', '.join(activity_map.keys())}")
            return
    
        try:
            activity = activity_map[activity_type](status_text)
            await client.change_presence(activity=activity)
        
            # Save activity for persistence
            global custom_activity
            custom_activity = f"{activity_type} {status_text}"
            save_json_data(ACTIVITY_FILE, {"custom_activity": custom_activity}, convert_keys=False)
        
            await interaction.followup.send(f"Activity set to: **{activity_type.title()}** `{status_text}` ✨")
        
        except Exception as e:
            await interaction.followup.send(f"Failed to set activity: {str(e)}")

    @app_commands.command(name="status_set", description="Set bot's online status.")
    async def set_status(self, interaction: discord.Interaction, status: str):
        """Set bot's online status"""
        await interaction.response.defer(ephemeral=True)
    
        status_map = {
            "online": discord.Status.online,
            "idle": discord.Status.idle,
            "dnd": discord.Status.dnd,
            "invisible": discord.Status.invisible
        }
    
        status = status.lower()
        if status not in status_map:
            await interaction.followup.send(f"Invalid status! Use: {', '.join(status_map.keys())}")
            return
    
        try:
            await client.change_presence(status=status_map[status])
            await interaction.followup.send(f"Status set to: **{status.title()}** 🔵")
        except Exception as e:
            await interaction.followup.send(f"Failed to set status: {str(e)}")

    @app_commands.command(name="add_prefill", description="Add a prefill message that appears as the bot's last response in conversations")
    async def add_prefill_command(self, interaction: discord.Interaction, prefill_text: str):
        """Add a prefill message that will be included as the bot's last response in conversation history"""
        await interaction.response.defer(ephemeral=True)
    
        # Store the prefill for this channel
        prefill_settings[interaction.channel.id] = prefill_text
    
        # Save to file
        save_json_data(PREFILL_SETTINGS_FILE, prefill_settings)
    
        await interaction.followup.send(f"✅ Prefill set for this channel!\n\n**Prefill text:** {prefill_text}\n\nThe bot will now include this as its last response in conversations. Use `/clear_prefill` to remove it.")

    @app_commands.command(name="clear_prefill", description="Remove the prefill message for this channel")
    async def clear_prefill_command(self, interaction: discord.Interaction):
        """Remove the prefill message for this channel"""
        await interaction.response.defer(ephemeral=True)
    
        if interaction.channel.id in prefill_settings:
            del prefill_settings[interaction.channel.id]
            save_json_data(PREFILL_SETTINGS_FILE, prefill_settings)
            await interaction.followup.send("✅ Prefill removed for this channel!")
        else:
            await interaction.followup.send("❌ No prefill is currently set for this channel.")

    @app_commands.command(name="bot_name_set", description="Change the bot's display name (Admin only)")
    async def set_bot_name(self, interaction: discord.Interaction, new_name: str):
        """Change bot's display name"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("❌ This command can only be used in servers!")
            return
    
        # Check admin permissions
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can change the bot's name!")
            return
    
        if not (1 <= len(new_name) <= 32):
            await interaction.followup.send("❌ Bot name must be between 1 and 32 characters!")
            return
    
        try:
            old_name = interaction.guild.me.display_name
            await interaction.guild.me.edit(nick=new_name)
            await interaction.followup.send(f"✅ **Bot name changed!**\n**Old:** {old_name}\n**New:** {new_name}")
        except discord.Forbidden:
            await interaction.followup.send("❌ I don't have permission to change my nickname in this server!")
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to change name: {str(e)}")

    @app_commands.command(name="bot_avatar_set", description="Change the bot's profile picture (Admin only)")
    async def set_bot_avatar(self, interaction: discord.Interaction, image_url: str = None):
        """Change bot's profile picture"""
        await interaction.response.defer(ephemeral=True)
    
        # Check admin permissions
        if interaction.guild and not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can change the bot's avatar!")
            return
    
        # Check for attachments first
        attachments = interaction.data.get('resolved', {}).get('attachments', {})
        if attachments:
            # Use the first attachment
            attachment_id = list(attachments.keys())[0]
            attachment = attachments[attachment_id]
        
            # Validate image
            if not any(attachment['filename'].lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                await interaction.followup.send("❌ Please upload a valid image file (PNG, JPG, GIF, or WebP)!")
                return
        
            if attachment['size'] > 8 * 1024 * 1024:  # 8MB limit
                await interaction.followup.send("❌ Image is too large! Must be under 8MB.")
                return
        
            try:
                # Download image data
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment['url']) as resp:
                        if resp.status != 200:
                            await interaction.followup.send("❌ Failed to download the uploaded image!")
                            return
                    
                        image_data = await resp.read()
            except Exception as e:
                await interaction.followup.send(f"❌ Failed to process uploaded image: {str(e)}")
                return
        elif image_url:
            # Download from URL
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as resp:
                        if resp.status != 200:
                            await interaction.followup.send("❌ Failed to download image from URL!")
                            return
                    
                        if resp.headers.get('content-type', '').startswith('image/'):
                            image_data = await resp.read()
                        else:
                            await interaction.followup.send("❌ URL does not point to a valid image!")
                            return
            except Exception as e:
                await interaction.followup.send(f"❌ Failed to download image: {str(e)}")
                return
        else:
            await interaction.followup.send("❌ Please provide an image URL or upload an image!\n"
                                           "**Usage:** `/bot_avatar_set https://example.com/image.png`\n"
                                           "**Or:** Upload an image with the command")
            return
    
        # Check file size (Discord limit is 8MB for avatars)
        if len(image_data) > 8 * 1024 * 1024:
            await interaction.followup.send("❌ Image is too large! Must be under 8MB.")
            return
    
        try:
            # Change avatar
            await client.user.edit(avatar=image_data)
            await interaction.followup.send("✅ **Bot avatar changed successfully!** 🎨\n"
                                           "*Note: It may take a few moments for the change to appear everywhere.*")
        
        except discord.HTTPException as e:
            if "rate limited" in str(e).lower():
                await interaction.followup.send("❌ Rate limited! You can only change the bot's avatar twice per hour.")
            else:
                await interaction.followup.send(f"❌ Discord error: {str(e)}")
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to change avatar: {str(e)}")

    @app_commands.command(name="bot_avatar_upload", description="Change the bot's profile picture using an uploaded image (Admin only)")
    async def set_bot_avatar_upload(self, interaction: discord.Interaction, image: discord.Attachment):
        """Change bot's profile picture using uploaded image"""
        await interaction.response.defer(ephemeral=True)
    
        # Check admin permissions
        if interaction.guild and not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can change the bot's avatar!")
            return
    
        # Validate image
        if not any(image.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
            await interaction.followup.send("❌ Please upload a valid image file (PNG, JPG, GIF, or WebP)!")
            return
    
        if image.size > 8 * 1024 * 1024:  # 8MB limit
            await interaction.followup.send("❌ Image is too large! Must be under 8MB.")
            return
    
        try:
            # Download image data
            image_data = await image.read()
        
            # Change avatar
            await client.user.edit(avatar=image_data)
            await interaction.followup.send("✅ **Bot avatar changed successfully!** 🎨\n"
                                           "*Note: It may take a few moments for the change to appear everywhere.*")
        
        except discord.HTTPException as e:
            if "rate limited" in str(e).lower():
                await interaction.followup.send("❌ Rate limited! You can only change the bot's avatar twice per hour.")
            else:
                await interaction.followup.send(f"❌ Discord error: {str(e)}")
        except Exception as e:
            await interaction.followup.send(f"❌ Failed to change avatar: {str(e)}")

    # LORE COMMANDS

    @app_commands.command(name='lore_add', description="Add lore information (Server: about members, DMs: about yourself)")
    async def add_lore(self, interaction: discord.Interaction, member: str = None, lore: str = None):
        """Add lore information - context-aware for servers vs DMs"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            # DM Mode: Add "About X" information for personal roleplay
            if member is not None:
                await interaction.followup.send("❌ **DM Mode**: You don't need to specify a member in DMs.\n"
                                               "**Usage:** `/lore_add <your_personal_info>`\n"
                                               "**Example:** `/lore_add I'm a 25-year-old artist who loves cats and fantasy novels`")
                return
        
            if lore is None:
                await interaction.followup.send("❌ **DM Mode**: Please provide your personal information.\n"
                                               "**Usage:** `/lore_add <about_yourself>`\n"
                                               "**Example:** `/lore_add I'm a 25-year-old artist who loves cats and fantasy novels`")
                return
        
            # Add DM-specific lore
            lore_book.add_dm_entry(interaction.user.id, lore)
            lore_preview = lore[:100] + ('...' if len(lore) > 100 else '')
            await interaction.followup.send(f"✅ **Personal DM lore added!** 📖\n"
                                           f"**About you:** `{lore_preview}`\n\n"
                                           f"💡 This information will be used in your DM conversations for better roleplay context!")
        else:
            # Server Mode: Add lore about server members
            if member is None:
                await interaction.followup.send("❌ **Server Mode**: Please specify a member.\n"
                                               "**Usage:** `/lore_add <member> <lore_about_them>`")
                return
        
            if lore is None:
                await interaction.followup.send("❌ **Server Mode**: Please provide lore information.\n"
                                               "**Usage:** `/lore_add <member> <lore_about_them>`")
                return
        
            # Convert string to Member object in server context
            member_obj = None
        
            # Try to find member by mention, ID, username, or display name
            if member.startswith('<@') and member.endswith('>'):
                # Handle mention format <@123456789> or <@!123456789>
                user_id = member.strip('<@!>')
                try:
                    member_obj = interaction.guild.get_member(int(user_id))
                except ValueError:
                    pass
            elif member.isdigit():
                # Handle raw user ID
                member_obj = interaction.guild.get_member(int(member))
            else:
                # Search by username or display name
                member_lower = member.lower()
                for guild_member in interaction.guild.members:
                    if (guild_member.name.lower() == member_lower or 
                        guild_member.display_name.lower() == member_lower or
                        guild_member.name.lower().startswith(member_lower) or
                        guild_member.display_name.lower().startswith(member_lower)):
                        member_obj = guild_member
                        break
        
            if member_obj is None:
                await interaction.followup.send(f"❌ **Member '{member}' not found!**\n"
                                               "Try using their exact username, display name, or mention them directly.")
                return
        
            # Add server lore
            lore_book.add_entry(interaction.guild.id, member_obj.id, lore)
            lore_preview = lore[:100] + ('...' if len(lore) > 100 else '')
            await interaction.followup.send(f"✅ **Server lore added for {member_obj.display_name}!** 📖\n"
                                           f"**Lore:** `{lore_preview}`")

    @app_commands.command(name="lore_edit", description="Edit lore information (Server: about members, DMs: about yourself)")
    async def edit_lore(self, interaction: discord.Interaction, member: str = None, new_lore: str = None):
        """Edit lore information - context-aware for servers vs DMs"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            if member is not None:
                await interaction.followup.send("❌ **DM Mode**: You don't need to specify a member in DMs.\n"
                                               "**Usage:** `/lore_edit <new_personal_info>`")
                return
        
            if new_lore is None:
                await interaction.followup.send("❌ **DM Mode**: Please provide new personal information.\n"
                                               "**Usage:** `/lore_edit <new_about_yourself>`")
                return
        
            # Check if DM lore exists
            current_lore = lore_book.get_dm_entry(interaction.user.id)
            if not current_lore:
                await interaction.followup.send("❌ **No personal lore found!**\n"
                                               "Use `/lore_add <about_yourself>` to create personal lore first.")
                return
        
            # Update DM lore
            lore_book.add_dm_entry(interaction.user.id, new_lore)
        
            # Show changes
            old_preview = current_lore[:100] + ('...' if len(current_lore) > 100 else '')
            new_preview = new_lore[:100] + ('...' if len(new_lore) > 100 else '')
        
            embed = discord.Embed(
                title="📖 Personal Lore Updated!",
                color=0x00ff99
            )
            embed.add_field(name="Previous Info", value=f"`{old_preview}`", inline=False)
            embed.add_field(name="New Info", value=f"`{new_preview}`", inline=False)
        
            await interaction.followup.send(embed=embed)
        else:
            # Server mode - convert string to member
            if member is None or new_lore is None:
                await interaction.followup.send("❌ **Server Mode**: Please specify both member and new lore.\n"
                                               "**Usage:** `/lore_edit <member> <new_lore>`")
                return
        
            # Find member (same logic as lore_add)
            member_obj = None
            if member.startswith('<@') and member.endswith('>'):
                user_id = member.strip('<@!>')
                try:
                    member_obj = interaction.guild.get_member(int(user_id))
                except ValueError:
                    pass
            elif member.isdigit():
                member_obj = interaction.guild.get_member(int(member))
            else:
                member_lower = member.lower()
                for guild_member in interaction.guild.members:
                    if (guild_member.name.lower() == member_lower or 
                        guild_member.display_name.lower() == member_lower or
                        guild_member.name.lower().startswith(member_lower) or
                        guild_member.display_name.lower().startswith(member_lower)):
                        member_obj = guild_member
                        break
        
            if member_obj is None:
                await interaction.followup.send(f"❌ **Member '{member}' not found!**")
                return
        
            current_lore = lore_book.get_entry(interaction.guild.id, member_obj.id)
            if not current_lore:
                await interaction.followup.send(f"❌ No lore found for {member_obj.display_name}!\n"
                                               "Use `/lore_add` to create lore first.")
                return
        
            lore_book.add_entry(interaction.guild.id, member_obj.id, new_lore)
        
            old_preview = current_lore[:100] + ('...' if len(current_lore) > 100 else '')
            new_preview = new_lore[:100] + ('...' if len(new_lore) > 100 else '')
        
            embed = discord.Embed(
                title="📖 Lore Updated!",
                description=f"Updated lore for {member_obj.display_name}",
                color=0x00ff99
            )
            embed.add_field(name="Previous Lore", value=f"`{old_preview}`", inline=False)
            embed.add_field(name="New Lore", value=f"`{new_preview}`", inline=False)
        
            await interaction.followup.send(embed=embed)

    @app_commands.command(name="lore_view", description="View lore information (Server: about members, DMs: about yourself)")
    async def view_lore(self, interaction: discord.Interaction, member: str = None):
        """View lore information - context-aware for servers vs DMs"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            if member is not None:
                await interaction.followup.send("❌ **DM Mode**: You don't need to specify a member in DMs.\n"
                                               "**Usage:** `/lore_view` (shows your personal info)")
                return
        
            lore = lore_book.get_dm_entry(interaction.user.id)
            if not lore:
                await interaction.followup.send("❌ **No personal lore found!**\n"
                                               "Use `/lore_add <about_yourself>` to create personal lore.")
                return
        
            embed = discord.Embed(
                title=f"📖 About {interaction.user.display_name}",
                description=lore,
                color=0x9932cc
            )
            embed.set_thumbnail(url=interaction.user.display_avatar.url)
            embed.set_footer(text="Use /lore_edit to modify this information")
        
            await interaction.followup.send(embed=embed)
        else:
            # Server mode (existing logic)
            if member is None:
                await interaction.followup.send("❌ **Server Mode**: Please specify a member.\n"
                                               "**Usage:** `/lore_view <member>`")
                return
        
            # Find member (same logic as other commands)
            member_obj = None
            if member.startswith('<@') and member.endswith('>'):
                user_id = member.strip('<@!>')
                try:
                    member_obj = interaction.guild.get_member(int(user_id))
                except ValueError:
                    pass
            elif member.isdigit():
                member_obj = interaction.guild.get_member(int(member))
            else:
                member_lower = member.lower()
                for guild_member in interaction.guild.members:
                    if (guild_member.name.lower() == member_lower or 
                        guild_member.display_name.lower() == member_lower or
                        guild_member.name.lower().startswith(member_lower) or
                        guild_member.display_name.lower().startswith(member_lower)):
                        member_obj = guild_member
                        break
        
            if member_obj is None:
                await interaction.followup.send(f"❌ **Member '{member}' not found!**")
                return
        
            lore = lore_book.get_entry(interaction.guild.id, member_obj.id)
            if not lore:
                await interaction.followup.send(f"❌ No lore found for {member_obj.display_name}!")
                return
        
            embed = discord.Embed(
                title=f"📖 Lore for {member_obj.display_name}",
                description=lore,
                color=0x9932cc
            )
            embed.set_thumbnail(url=member_obj.display_avatar.url)
            embed.set_footer(text="Use /lore_edit to modify this entry")
        
            await interaction.followup.send(embed=embed)

    @app_commands.command(name="lore_remove", description="Remove lore information (Server: about members, DMs: about yourself)")
    async def remove_lore(self, interaction: discord.Interaction, member: str = None):
        """Remove lore information - context-aware for servers vs DMs"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            if member is not None:
                await interaction.followup.send("❌ **DM Mode**: You don't need to specify a member in DMs.\n"
                                               "**Usage:** `/lore_remove` (removes your personal info)")
                return
        
            if interaction.user.id not in lore_book.dm_entries:
                await interaction.followup.send("❌ **No personal lore to remove!**")
                return
        
            lore_book.remove_dm_entry(interaction.user.id)
            await interaction.followup.send("✅ **Personal lore removed!** 🗑️")
        else:
            # Server mode (existing logic)
            if member is None:
                await interaction.followup.send("❌ **Server Mode**: Please specify a member.\n"
                                               "**Usage:** `/lore_remove <member>`")
                return
        
            # Find member (same logic as other commands)
            member_obj = None
            if member.startswith('<@') and member.endswith('>'):
                user_id = member.strip('<@!>')
                try:
                    member_obj = interaction.guild.get_member(int(user_id))
                except ValueError:
                    pass
            elif member.isdigit():
                member_obj = interaction.guild.get_member(int(member))
            else:
                member_lower = member.lower()
                for guild_member in interaction.guild.members:
                    if (guild_member.name.lower() == member_lower or 
                        guild_member.display_name.lower() == member_lower or
                        guild_member.name.lower().startswith(member_lower) or
                        guild_member.display_name.lower().startswith(member_lower)):
                        member_obj = guild_member
                        break
        
            if member_obj is None:
                await interaction.followup.send(f"❌ **Member '{member}' not found!**")
                return
        
            lore_book.remove_entry(interaction.guild.id, member_obj.id)
            await interaction.followup.send(f"✅ **Lore removed for {member_obj.display_name}!** 🗑️")

    @app_commands.command(name="lore_list", description="List lore entries (Server: all members, DMs: just yourself)")
    async def list_lore(self, interaction: discord.Interaction):
        """List lore entries - context-aware for servers vs DMs"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            lore = lore_book.get_dm_entry(interaction.user.id)
            if not lore:
                await interaction.followup.send("❌ **No personal lore found!**\n"
                                               "Use `/lore_add <about_yourself>` to create personal lore for better DM roleplay.")
                return
        
            embed = discord.Embed(
                title="📖 Your Personal DM Lore",
                description=lore,
                color=0x9932cc
            )
            embed.set_footer(text="This information helps create better roleplay context in your DMs!")
            await interaction.followup.send(embed=embed)
        else:
            # Server mode - need to add the missing list_entries method to LoreBook
            if interaction.guild.id not in lore_book.entries or not lore_book.entries[interaction.guild.id]:
                await interaction.followup.send("❌ **No lore entries found for this server.**\n"
                                               "Use `/lore_add <member> <lore>` to create lore entries.")
                return
        
            embed = discord.Embed(title="📖 Server Lore Book", color=0x9932cc)
            for user_id, lore in lore_book.entries[interaction.guild.id].items():
                member = interaction.guild.get_member(user_id)
                name = member.display_name if member else f"User {user_id}"
                lore_preview = lore[:100] + ("..." if len(lore) > 100 else "")
                embed.add_field(name=name, value=lore_preview, inline=False)
        
            await interaction.followup.send(embed=embed)

    # MEMORY COMMANDS

    @app_commands.command(name="memory_generate", description="Generate a memory summary from recent messages (context-aware)")
    async def generate_memory(self, interaction: discord.Interaction, num_messages: int):
        """Generate and save memory summary from recent conversation"""
        await interaction.response.defer(ephemeral=True)
    
        if not (1 <= num_messages <= 100):
            await interaction.followup.send("Number of messages must be between 1 and 100.")
            return
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        try:
            # Check if there's conversation history first
            history = get_conversation_history(interaction.channel.id)
            if not history:
                await interaction.followup.send("❌ **No conversation history found!**\n"
                                               "Start a conversation with the bot first, then try generating a memory.")
                return
        
            if len(history) < num_messages:
                await interaction.followup.send(f"⚠️ **Only {len(history)} messages available in history.**\n"
                                               f"Generating memory from all available messages instead of {num_messages}.")
        
            async with interaction.channel.typing():
                if is_dm:
                    user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
                    memory_summary = await generate_memory_summary(
                        interaction.channel.id, 
                        num_messages, 
                        guild=None, 
                        user_id=interaction.user.id,
                        username=user_name
                    )
                    context = "DM conversation"
                else:
                    memory_summary = await generate_memory_summary(
                        interaction.channel.id, 
                        num_messages, 
                        interaction.guild
                    )
                    context = "server conversation"
        
            # Check if memory generation failed
            if not memory_summary:
                await interaction.followup.send("❌ **Failed to generate memory summary: No response from AI**")
                return
            elif memory_summary.startswith("❌") or memory_summary.startswith("Error"):
                await interaction.followup.send(f"❌ **Failed to generate memory summary:**\n{memory_summary}")
                return
        
            # Save the memory
            if is_dm:
                memory_index = memory_manager.save_dm_memory(interaction.user.id, memory_summary)
            else:
                memory_index = memory_manager.save_memory(interaction.guild.id, memory_summary)
        
            embed = discord.Embed(
                title="🧠 Generated and Saved Memory",
                description=f"Summary of the last {min(num_messages, len(history))} messages from your {context} (Memory #{memory_index + 1}):",
                color=0x9932cc
            )
        
            # Handle long summaries
            if len(memory_summary) > 1024:
                parts = [memory_summary[i:i+1024] for i in range(0, len(memory_summary), 1024)]
                for i, part in enumerate(parts):
                    field_name = f"Memory Summary (Part {i+1})" if len(parts) > 1 else "Memory Summary"
                    embed.add_field(name=field_name, value=part, inline=False)
            else:
                embed.add_field(name="Memory Summary", value=memory_summary, inline=False)
        
            embed.set_footer(text="✅ Memory saved! The bot will recall this when relevant topics come up.")
            await interaction.followup.send(embed=embed)
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            # print(f"Error in memory_generate command: {e}")
            # print(f"Full traceback: {error_details}")
            await interaction.followup.send(f"❌ **Error generating memory:** {str(e)}\n"
                                           f"Check the console for detailed error information.")

    @app_commands.command(name="memory_save", description="Manually save a memory summary (context-aware)")
    async def save_memory(self, interaction: discord.Interaction, summary: str):
        """Manually save a memory summary"""
        await interaction.response.defer(ephemeral=True)
    
        if len(summary) < 10:
            await interaction.followup.send("Memory summary must be at least 10 characters long.")
            return
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            memory_index = memory_manager.save_dm_memory(interaction.user.id, summary)
            context = "DM"
        else:
            memory_index = memory_manager.save_memory(interaction.guild.id, summary)
            context = "server"
    
        summary_preview = summary[:200] + ('...' if len(summary) > 200 else '')
        await interaction.followup.send(f"✅ **{context.title()} memory saved!** (Memory #{memory_index + 1})\n\n**Summary:** {summary_preview}")

    @app_commands.command(name="memory_list", description="View all saved memories (context-aware)")
    async def list_memories(self, interaction: discord.Interaction):
        """Display all saved memories"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            memories = memory_manager.get_all_dm_memories(interaction.user.id)
            context = "DM"
            location = "your DMs"
        else:
            memories = memory_manager.get_all_memories(interaction.guild.id)
            context = "server"
            location = interaction.guild.name
    
        if not memories:
            await interaction.followup.send(f"❌ **No {context} memories saved yet!**\n"
                                           f"Use `/memory_generate` or `/memory_save` to create some.")
            return
    
        embed = discord.Embed(
            title=f"🧠 {context.title()} Memory Bank",
            description=f"Saved memories for {location}:",
            color=0x9932cc
        )
    
        # Show last 10 memories with numbers
        start_index = max(0, len(memories) - 10)
        for i, memory in enumerate(memories[start_index:], start=start_index + 1):
            memory_text = memory["memory"][:100] + ("..." if len(memory["memory"]) > 100 else "")
            keywords = ", ".join(memory["keywords"][:5])
            embed.add_field(
                name=f"Memory #{i}",
                value=f"**Summary:** {memory_text}\n**Keywords:** {keywords}",
                inline=False
            )
    
        if len(memories) > 10:
            embed.set_footer(text=f"Showing last 10 memories out of {len(memories)} total\nUse /memory_view <number> to see full details")
        else:
            embed.set_footer(text="Use /memory_view <number> to see full details | /memory_edit <number> to edit | /memory_delete <number> to delete")
    
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="memory_clear", description="Delete all saved memories (context-aware)")
    async def clear_memories(self, interaction: discord.Interaction):
        """Delete all memories"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            memory_count = len(memory_manager.get_all_dm_memories(interaction.user.id))
            context = "DM"
        else:
            memory_count = len(memory_manager.get_all_memories(interaction.guild.id))
            context = "server"
    
        if memory_count == 0:
            await interaction.followup.send(f"❌ **No {context} memories to clear.**")
            return
    
        if is_dm:
            memory_manager.delete_all_dm_memories(interaction.user.id)
        else:
            memory_manager.delete_all_memories(interaction.guild.id)
    
        await interaction.followup.send(f"🗑️ **Cleared all {memory_count} {context} memories!**\n"
                                       f"The bot's {context} memory has been wiped clean.")

    @app_commands.command(name="memory_edit", description="Edit a specific saved memory (context-aware)")
    async def edit_memory(self, interaction: discord.Interaction, memory_number: int, new_summary: str):
        """Edit a specific memory by number"""
        await interaction.response.defer(ephemeral=True)
    
        if len(new_summary) < 10:
            await interaction.followup.send("Memory summary must be at least 10 characters long.")
            return
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        # Convert to 0-based index
        memory_index = memory_number - 1
    
        if is_dm:
            # DM mode
            old_memory = memory_manager.get_dm_memory_by_index(interaction.user.id, memory_index)
            if not old_memory:
                total_memories = len(memory_manager.get_all_dm_memories(interaction.user.id))
                await interaction.followup.send(f"❌ **DM Memory #{memory_number} not found!**\n"
                                               f"Available memories: 1-{total_memories}\n"
                                               f"Use `/memory_list` to see all memories.")
                return
        
            success = memory_manager.edit_dm_memory(interaction.user.id, memory_index, new_summary)
            context = "DM"
        else:
            # Server mode
            old_memory = memory_manager.get_memory_by_index(interaction.guild.id, memory_index)
            if not old_memory:
                total_memories = len(memory_manager.get_all_memories(interaction.guild.id))
                await interaction.followup.send(f"❌ **Server Memory #{memory_number} not found!**\n"
                                               f"Available memories: 1-{total_memories}\n"
                                               f"Use `/memory_list` to see all memories.")
                return
        
            success = memory_manager.edit_memory(interaction.guild.id, memory_index, new_summary)
            context = "Server"
    
        if success:
            old_preview = old_memory["memory"][:100] + ('...' if len(old_memory["memory"]) > 100 else '')
            new_preview = new_summary[:100] + ('...' if len(new_summary) > 100 else '')
        
            embed = discord.Embed(
                title=f"🧠 {context} Memory Updated!",
                description=f"Updated Memory #{memory_number}",
                color=0x9932cc
            )
            embed.add_field(name="Previous Memory", value=old_preview, inline=False)
            embed.add_field(name="New Memory", value=new_preview, inline=False)
        
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send("❌ Failed to edit memory.")

    @app_commands.command(name="memory_delete", description="Delete a specific saved memory (context-aware)")
    async def delete_memory(self, interaction: discord.Interaction, memory_number: int):
        """Delete a specific memory by number"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        # Convert to 0-based index
        memory_index = memory_number - 1
    
        if is_dm:
            # DM mode
            memory_to_delete = memory_manager.get_dm_memory_by_index(interaction.user.id, memory_index)
            if not memory_to_delete:
                total_memories = len(memory_manager.get_all_dm_memories(interaction.user.id))
                await interaction.followup.send(f"❌ **DM Memory #{memory_number} not found!**\n"
                                               f"Available memories: 1-{total_memories}\n"
                                               f"Use `/memory_list` to see all memories.")
                return
        
            success = memory_manager.delete_dm_memory(interaction.user.id, memory_index)
            context = "DM"
        else:
            # Server mode
            memory_to_delete = memory_manager.get_memory_by_index(interaction.guild.id, memory_index)
            if not memory_to_delete:
                total_memories = len(memory_manager.get_all_memories(interaction.guild.id))
                await interaction.followup.send(f"❌ **Server Memory #{memory_number} not found!**\n"
                                               f"Available memories: 1-{total_memories}\n"
                                               f"Use `/memory_list` to see all memories.")
                return
        
            success = memory_manager.delete_memory(interaction.guild.id, memory_index)
            context = "Server"
    
        if success:
            deleted_preview = memory_to_delete["memory"][:150] + ('...' if len(memory_to_delete["memory"]) > 150 else '')
            await interaction.followup.send(f"🗑️ **{context} Memory #{memory_number} deleted!**\n\n"
                                           f"**Deleted memory:** {deleted_preview}")
        else:
            await interaction.followup.send("❌ Failed to delete memory.")

    @app_commands.command(name="memory_view", description="View a specific memory in full detail (context-aware)")
    async def view_memory(self, interaction: discord.Interaction, memory_number: int):
        """View a specific memory in detail"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        # Convert to 0-based index
        memory_index = memory_number - 1
    
        if is_dm:
            # DM mode
            memory = memory_manager.get_dm_memory_by_index(interaction.user.id, memory_index)
            if not memory:
                total_memories = len(memory_manager.get_all_dm_memories(interaction.user.id))
                await interaction.followup.send(f"❌ **DM Memory #{memory_number} not found!**\n"
                                               f"Available memories: 1-{total_memories}\n"
                                               f"Use `/memory_list` to see all memories.")
                return
            context = "DM"
        else:
            # Server mode
            memory = memory_manager.get_memory_by_index(interaction.guild.id, memory_index)
            if not memory:
                total_memories = len(memory_manager.get_all_memories(interaction.guild.id))
                await interaction.followup.send(f"❌ **Server Memory #{memory_number} not found!**\n"
                                               f"Available memories: 1-{total_memories}\n"
                                               f"Use `/memory_list` to see all memories.")
                return
            context = "Server"
    
        embed = discord.Embed(
            title=f"🧠 {context} Memory #{memory_number}",
            description=memory["memory"],
            color=0x9932cc
        )
    
        keywords = ", ".join(memory["keywords"][:10])  # Show first 10 keywords
        if len(memory["keywords"]) > 10:
            keywords += "..."
    
        embed.add_field(name="Keywords", value=keywords, inline=False)
    
        # Format timestamp
        timestamp = datetime.datetime.fromtimestamp(memory["timestamp"])
        embed.set_footer(text=f"Created: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
        await interaction.followup.send(embed=embed)

    # UTILITY COMMANDS

    @app_commands.command(name="delete_messages", description="Delete the bot's last N messages from this channel/DM")
    async def delete_messages(self, interaction: discord.Interaction, number: int):
        """Delete bot's last N logical messages from channel or DM"""
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        # Set limits based on channel type
        if is_dm:
            max_number = 20
            if not (1 <= number <= max_number):
                await interaction.response.send_message(f"Number must be between 1 and {max_number} for DMs.", ephemeral=True)
                return
        else:
            max_number = 50
            if not (1 <= number <= max_number):
                await interaction.response.send_message(f"Number must be between 1 and {max_number} for servers.", ephemeral=True)
                return
        
            # Check permissions for servers
            if not interaction.channel.permissions_for(interaction.guild.me).manage_messages:
                await interaction.response.send_message("❌ I don't have permission to delete messages in this channel.", ephemeral=True)
                return
    
        if is_dm:
            # For DMs: Defer ephemerally and delete directly
            await interaction.response.defer(ephemeral=True)
        
            # Track deleted messages for DM history loading
            if interaction.user.id not in recently_deleted_dm_messages:
                recently_deleted_dm_messages[interaction.user.id] = set()
        
            # Collect message IDs before deletion for tracking
            bot_messages_to_delete = []
            async for message in interaction.channel.history(limit=200):
                if message.author == client.user and len(message.content.strip()) > 0:
                    bot_messages_to_delete.append(message)
                    if len(bot_messages_to_delete) >= number * 3:  # Get extra in case of multipart
                        break
        
            # Perform deletion directly
            deleted_count = await delete_bot_messages(interaction.channel, number, set())
        
            # Track the deleted message IDs
            for msg in bot_messages_to_delete[:deleted_count * 2]:  # Approximate tracking
                recently_deleted_dm_messages[interaction.user.id].add(msg.id)
        
            # Follow up with result (ephemeral)
            if deleted_count > 0:
                await interaction.followup.send(f"✅ Deleted {deleted_count} logical message(s)!")
            else:
                await interaction.followup.send("❌ No messages found to delete.")
        else:
            # For servers: Send public status message first
            await interaction.response.send_message(f"🗑️ Deleting {number} of my logical messages...")
        
            # Get the response message to exclude it
            response_msg = await interaction.original_response()
            exclude_ids = {response_msg.id}
        
            # Perform deletion
            deleted_count = await delete_bot_messages(interaction.channel, number, exclude_ids)
        
            # Edit the status message with result
            if deleted_count > 0:
                await interaction.edit_original_response(content=f"✅ Deleted {deleted_count} logical message(s)!")
            else:
                await interaction.edit_original_response(content="❌ No messages found to delete (or permission denied).")
        
            # Delete the status message after a few seconds
            await asyncio.sleep(3)
            try:
                await response_msg.delete()
            except:
                pass

    @app_commands.command(name="clear", description="Clear the conversation history for this specific channel/DM")
    async def clear(self, interaction: discord.Interaction):
        """Clear conversation history for current channel"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        channel_id = interaction.channel.id
    
        # Check if DM full history is enabled (which would reload history anyway)
        dm_full_history_warning = ""
        if is_dm and dm_manager.is_dm_full_history_enabled(interaction.user.id):
            dm_full_history_warning = "\n\n⚠️ **Note:** You have DM full history loading enabled, so the bot will reload our conversation history on the next message. Use `/dm_history_toggle false` first if you want a true fresh start."
    
        # Clear stored conversation history
        history_cleared = False
        if channel_id in conversations:
            del conversations[channel_id]
            history_cleared = True
    
        # Clear recent participants (server only)
        participants_cleared = False
        if interaction.guild and channel_id in recent_participants:
            del recent_participants[channel_id]
            participants_cleared = True
    
        # Clear multipart response tracking
        multipart_cleared = False
        if channel_id in multipart_responses:
            del multipart_responses[channel_id]
            multipart_cleared = True
    
        if channel_id in multipart_response_counter:
            del multipart_response_counter[channel_id]
    
        # Provide detailed feedback
        if history_cleared or participants_cleared or multipart_cleared:
            cleared_items = []
            if history_cleared:
                cleared_items.append("conversation history")
            if participants_cleared:
                cleared_items.append("participant tracking")
            if multipart_cleared:
                cleared_items.append("message tracking")
        
            context = "DM" if is_dm else "channel"
            await interaction.followup.send(f"✅ **{context.title()} memory cleared!**\n\n"
                                           f"**Cleared:** {', '.join(cleared_items)}\n"
                                           f"The bot will start fresh from the next message.{dm_full_history_warning}")
        else:
            context = "DM" if is_dm else "channel"
            await interaction.followup.send(f"✅ **No {context} memory to clear!**\n"
                                           f"This {context} was already starting fresh.{dm_full_history_warning}")


    @app_commands.command(name="debug_history", description="Show current stored history and context info (Admin only in servers)")
    async def debug_history(self, interaction: discord.Interaction, limit: int = 12):
        """Debug view of current conversation history and settings"""
        await interaction.response.defer(ephemeral=True)

        # Admin check for servers
        if interaction.guild and not check_admin_permissions(interaction):
            await interaction.followup.send("? Only administrators can use this command in servers.")
            return

        if not (1 <= limit <= 50):
            await interaction.followup.send("Limit must be between 1 and 50.")
            return

        channel_id = interaction.channel.id
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        history = get_conversation_history(channel_id)
        summary_text = conversation_summaries.get(channel_id, "")

        # Basic context info
        info_lines = []
        context_label = "DM" if is_dm else f"Server: {interaction.guild.name}"
        info_lines.append(f"Context: {context_label}")

        if is_dm:
            dm_full_history = dm_manager.is_dm_full_history_enabled(interaction.user.id)
            info_lines.append(f"DM full history enabled: {dm_full_history}")
            selected_guild_id = dm_server_selection.get(interaction.user.id)
            if selected_guild_id:
                info_lines.append(f"DM server selection: {selected_guild_id}")
            info_lines.append(f"Reasoning enabled: {is_reasoning_enabled(None, interaction.user.id, True)}")
            info_lines.append(f"Reasoning effort: {get_reasoning_effort(None, interaction.user.id, True)}")
        else:
            info_lines.append(f"History length setting: {get_history_length(interaction.guild.id)}")
            info_lines.append(f"Reasoning enabled: {is_reasoning_enabled(interaction.guild.id, None, False)}")
            info_lines.append(f"Reasoning effort: {get_reasoning_effort(interaction.guild.id, None, False)}")

        info_lines.append(f"Stored history entries: {len(history)}")
        if summary_text:
            info_lines.append(f"Summary stored: Yes ({min(len(summary_text), 500)} chars)")
        else:
            info_lines.append("Summary stored: No")
        # Recent reaction micro-memory
        reactions = get_recent_reaction_memory(channel_id)
        if reactions:
            last_reaction = reactions[-1]
            info_lines.append(f"Recent reactions stored: {len(reactions)} (latest: {last_reaction.get('reaction')})")
        else:
            info_lines.append("Recent reactions stored: 0")

        header = "\n".join(info_lines)

        # Render recent history
        lines = []
        recent = history[-limit:] if history else []
        for i, msg in enumerate(recent, start=max(1, len(history) - len(recent) + 1)):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "[non-text content]"
            content = str(content).strip()
            user_tag = ""
            if role == "user" and msg.get("user_id"):
                user_tag = f" <@{msg.get('user_id')}>"

            # Expand grouped user messages for clarity
            if role == "user" and "\n" in content:
                parts = [p.strip() for p in content.split("\n") if p.strip()]
                for idx, part in enumerate(parts, start=1):
                    display = part
                    if len(display) > 220:
                        display = display[:220] + "..."
                    lines.append(f"{i}.{idx} {role}{user_tag}: {display}")
            else:
                display = content.replace("\n", " / ")
                if len(display) > 220:
                    display = display[:220] + "..."
                lines.append(f"{i}. {role}{user_tag}: {display}")

        history_block = "\n".join(lines) if lines else "No stored history for this channel."

        # Reactions block (expanded)
        reactions_block = "None."
        if reactions:
            reaction_lines = []
            for idx, r in enumerate(reactions, start=1):
                reaction_lines.append(f"{idx}. {r.get('reaction')} to <@{r.get('user_id')}>")
            reactions_block = "\n".join(reaction_lines)

        # Chunk output to avoid message limits
        full_text = f"**History Debug**\n{header}\n\n**Recent {len(recent)} entries**\n{history_block}\n\n**Recent Reactions**\n{reactions_block}"
        if len(full_text) <= 1900:
            await interaction.followup.send(full_text)
        else:
            # Send header first, then chunk history
            await interaction.followup.send(f"**History Debug**\n{header}\n\n**Recent {len(recent)} entries**")
            chunk = ""
            for line in lines:
                if len(chunk) + len(line) + 1 > 1800:
                    await interaction.followup.send(chunk)
                    chunk = ""
                chunk += (line + "\n")
            if chunk:
                await interaction.followup.send(chunk)
            await interaction.followup.send(f"**Recent Reactions**\n{reactions_block}")

    # FUN COMMANDS

    @app_commands.command(name="kiss", description="Give the bot a kiss! 💋")
    async def kiss_command(self, interaction: discord.Interaction):
        """User kisses the bot - bot reacts accordingly"""
        await interaction.response.defer(ephemeral=False)
    
        user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        kiss_instruction = f"[SPECIAL INSTRUCTION]: {user_name} just gave you a kiss! React naturally to being kissed by them. Be flustered, surprised, happy, aroused, or however your character would react to suddenly being kissed. NO ASTERISKS ROLEPLAY OR REACTIONS!"
        guild_id = interaction.guild.id if interaction.guild else None

        await add_to_history(interaction.channel.id, "user", f"[{user_name} used /kiss]", interaction.user.id, guild_id, user_name=user_name)
    
        async with interaction.channel.typing():
            response = await generate_response(
                interaction.channel.id, 
                f"[{user_name} used /kiss]\n{kiss_instruction}", 
                interaction.guild, 
                None,
                user_name, 
                is_dm, 
                interaction.user.id,
                None
            )
        
            await send_fun_command_response(interaction, response)

    @app_commands.command(name="hug", description="Give the bot a warm hug! 🤗")
    async def hug_command(self, interaction: discord.Interaction):
        """User hugs the bot - bot reacts accordingly"""
        await interaction.response.defer(ephemeral=False)
    
        user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        hug_instruction = f"[SPECIAL INSTRUCTION]: {user_name} just gave you a warm hug! React naturally to being hugged by them. Be comforted, happy, grateful, upset, or however your character would respond to a nice hug. NO ASTERISKS ROLEPLAY OR REACTIONS!"
        guild_id = interaction.guild.id if interaction.guild else None

        await add_to_history(interaction.channel.id, "user", f"[{user_name} used /hug]", interaction.user.id, guild_id, user_name=user_name)

        async with interaction.channel.typing():
            response = await generate_response(
                interaction.channel.id, 
                f"[{user_name} used /hug]\n{hug_instruction}", 
                interaction.guild, 
                None,
                user_name, 
                is_dm, 
                interaction.user.id,
                None
            )
        
            await send_fun_command_response(interaction, response)

    @app_commands.command(name="joke", description="Ask the bot to tell you a joke! 😄")
    async def joke_command(self, interaction: discord.Interaction):
        """Bot tells a joke"""
        await interaction.response.defer(ephemeral=False)
    
        user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        joke_instruction = f"[SPECIAL INSTRUCTION]: {user_name} wants you to tell them a joke! Tell a funny, uncensored joke that fits your character's personality. Make it entertaining and engaging. You can be witty, punny, clever, or whatever style fits your character! NO ASTERISKS ROLEPLAY OR REACTIONS!"
        guild_id = interaction.guild.id if interaction.guild else None

        await add_to_history(interaction.channel.id, "user", f"[{user_name} used /joke]", interaction.user.id, guild_id, user_name=user_name)

        async with interaction.channel.typing():
            response = await generate_response(
                interaction.channel.id, 
                f"[{user_name} used /joke]\n{joke_instruction}", 
                interaction.guild, 
                None,
                user_name, 
                is_dm, 
                interaction.user.id,
                None
            )
        
            await send_fun_command_response(interaction, response)

    @app_commands.command(name="bonk", description="Bonk the bot's head! 🔨")
    async def bonk_command(self, interaction: discord.Interaction):
        """Bot gets bonked"""
        await interaction.response.defer(ephemeral=False)
    
        user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        bonk_instruction = f"[SPECIAL INSTRUCTION]: {user_name} just bonked your head! React naturally to being bonked by them. Be in pain, upset, grateful, furious, or however your character would respond to a silly bonk. NO ASTERISKS ROLEPLAY OR REACTIONS!"
        guild_id = interaction.guild.id if interaction.guild else None

        await add_to_history(interaction.channel.id, "user", f"[{user_name} used /bonk]", interaction.user.id, guild_id, user_name=user_name)
    
        async with interaction.channel.typing():
            response = await generate_response(
                interaction.channel.id, 
                f"[{user_name} used /bonk]\n{bonk_instruction}", 
                interaction.guild, 
                None,
                user_name, 
                is_dm, 
                interaction.user.id,
                None
            )
        
            await send_fun_command_response(interaction, response)

    @app_commands.command(name="bite", description="Bite the bot! Chomp! 🧛")
    async def bite_command(self, interaction: discord.Interaction):
        """User bites the bot - bot reacts accordingly"""
        await interaction.response.defer(ephemeral=False)
    
        user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
        is_dm = isinstance(interaction.channel, discord.DMChannel)
        bite_instruction = f"[SPECIAL INSTRUCTION]: {user_name} just bit you! React naturally to being bit by them. Be in pain, amused, laughing, upset, or however your character would respond to a playful bite. REMEMBER: NO ASTERISKS ROLEPLAY OR REACTIONS!"
        guild_id = interaction.guild.id if interaction.guild else None

        await add_to_history(interaction.channel.id, "user", f"[{user_name} used /bite]", interaction.user.id, guild_id, user_name=user_name)

        async with interaction.channel.typing():
            response = await generate_response(
                interaction.channel.id, 
                f"[{user_name} used /bite]\n{bite_instruction}", 
                interaction.guild, 
                None,
                user_name, 
                is_dm, 
                interaction.user.id,
                None
            )
        
            await send_fun_command_response(interaction, response)

    @app_commands.command(name="affection", description="Ask how much the bot likes you! 💕")
    async def affection_command(self, interaction: discord.Interaction):
        """Bot evaluates affection level based on chat history"""
        await interaction.response.defer(ephemeral=False)
    
        user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        # Get conversation history for analysis
        if is_dm and dm_manager.is_dm_full_history_enabled(interaction.user.id):
            try:
                shared_guild = get_shared_guild(interaction.user.id)
                history = await load_all_dm_history(interaction.channel, interaction.user.id, shared_guild)
            except:
                history = get_conversation_history(interaction.channel.id)
        else:
            history = get_conversation_history(interaction.channel.id)
    
        # Analyze user interactions from history
        user_messages = []
        for msg in history:
            content = msg.get("content")
            if isinstance(content, str) and msg["role"] == "user":
                if is_dm or f"{user_name}:" in content:
                    user_messages.append(content)
    
        # Create interaction context
        if user_messages:
            recent_interactions = user_messages[-10:]
            interaction_summary = " | ".join([msg[:50] + "..." if len(msg) > 50 else msg for msg in recent_interactions])
            interaction_context = f"Recent interactions with {user_name}: {interaction_summary}"
        else:
            interaction_context = f"This is one of the first interactions with {user_name}."
    
        affection_instruction = f"[SPECIAL INSTRUCTION]: {user_name} wants to know how much you like them! Based on your chat history and interactions, give them a percentage score (0-100%) of how much you like them, and explain why. Be honest. Consider things like: how often you've talked, how nice they've been, shared interests, funny moments, etc. REMEMBER: NO ASTERISKS ROLEPLAY OR REACTIONS!\n{interaction_context}"
        guild_id = interaction.guild.id if interaction.guild else None

        await add_to_history(interaction.channel.id, "user", f"[{user_name} used /affection]", interaction.user.id, guild_id, user_name=user_name)
    
        async with interaction.channel.typing():
            response = await generate_response(
                interaction.channel.id, 
                f"[{user_name} used /affection]\n{affection_instruction}", 
                interaction.guild, 
                None,
                user_name, 
                is_dm, 
                interaction.user.id,
                None
            )
        
            await send_fun_command_response(interaction, response)

    # DM-SPECIFIC COMMANDS

    @app_commands.command(name="dm_toggle", description="Toggle auto check-up messages in DMs (bot will message you once if inactive for 6+ hours)")
    async def dm_toggle_command(self, interaction: discord.Interaction, enabled: bool = None):
        """Toggle DM auto check-up feature"""
        await interaction.response.defer(ephemeral=True)
    
        if enabled is None:
            # Show current status
            current_status = dm_manager.is_dm_toggle_enabled(interaction.user.id)
            reminder_sent = dm_manager.check_up_sent.get(interaction.user.id, False)
        
            status_text = "✅ Enabled" if current_status else "❌ Disabled"
            if current_status and reminder_sent:
                status_text += " (reminder already sent this session)"
        
            await interaction.followup.send(f"**DM Auto Check-up Status:** {status_text}\n\n"
                                           f"When enabled, I'll send you **one** message if you haven't talked to me for 6+ hours.\n"
                                           f"The reminder resets when you become active again.\n\n"
                                           f"Use `/dm_toggle true` to enable or `/dm_toggle false` to disable.")
            return
    
        dm_manager.set_dm_toggle(interaction.user.id, enabled)
    
        if enabled:
            await interaction.followup.send("✅ **DM Auto Check-up Enabled!**\n\n"
                                           "I'll now send you a caring message if you haven't talked to me for 6+ hours.\n"
                                           "**Note:** Only one reminder per session - it resets when you become active again.\n"
                                           "The reminder includes context from your recent messages for continuity.\n\n"
                                           "💡 You can disable this anytime with `/dm_toggle false`")
        else:
            await interaction.followup.send("❌ **DM Auto Check-up Disabled.**\n\n"
                                           "I won't send you automatic check-up messages anymore.\n"
                                           "You can re-enable this anytime with `/dm_toggle true`")

    @app_commands.command(name="dm_personality_list", description="List all personalities available for your DMs")
    async def dm_personality_list(self, interaction: discord.Interaction):
        """List all personalities available from shared servers"""
        await interaction.response.defer(ephemeral=True)
    
        user_id = interaction.user.id
        available_personalities = {}
    
        shared_guilds_count = 0
        # Collect personalities from all shared guilds
        for guild in client.guilds:
            # Try both methods to find the member
            member = guild.get_member(user_id)
            if not member:
                try:
                    member = await guild.fetch_member(user_id)
                except (discord.NotFound, discord.Forbidden):
                    member = None
        
            if member:  # User is in this guild
                shared_guilds_count += 1
            
                # Get current server personality
                current_personality = guild_personalities.get(guild.id, "default")
            
                # Add the active personality for this guild
                if current_personality == "default":
                    display_name = DEFAULT_PERSONALITIES["default"]["name"]
                    prompt = DEFAULT_PERSONALITIES["default"]["prompt"]
                elif guild.id in custom_personalities and current_personality in custom_personalities[guild.id]:
                    personality_data = custom_personalities[guild.id][current_personality]
                    display_name = personality_data["name"]
                    prompt = personality_data["prompt"]
                else:
                    display_name = "Unknown"
                    prompt = "Personality data not found"
            
                available_personalities[guild.id] = {
                    "name": display_name,
                    "personality_key": current_personality,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "guild_name": guild.name,
                }
    
        if shared_guilds_count == 0:
            await interaction.followup.send("❌ **No shared servers found!**\n"
                                           "Make sure you're in a server with the bot and that the bot has proper permissions.\n\n"
                                           f"**Debug info:** Bot is in {len(client.guilds)} servers total.")
            return
    
        embed = discord.Embed(
            title="🎭 Available DM Personalities",
            description=f"Your DMs will automatically use personalities from servers you share with the bot.\nFound {shared_guilds_count} shared server(s):",
            color=0x9932cc
        )
    
        # Add personalities from each shared server
        for guild_id, data in available_personalities.items():
            field_name = f"{data['guild_name']}"
            field_value = f"**{data['name']}** (`{data['personality_key']}`)\n{data['prompt']}"
            embed.add_field(name=field_name, value=field_value, inline=False)
    
        embed.set_footer(text="💡 DM personality is automatically determined by your shared servers!\nUse /personality_set in servers to change personalities.")
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="dm_personality_set", description="Choose which server's personality to use in your DMs")
    async def dm_personality_set(self, interaction: discord.Interaction, server_name: str = None):
        """Set which server's personality to use in DMs"""
        await interaction.response.defer(ephemeral=True)
    
        user_id = interaction.user.id
    
        # Collect all shared guilds and their personalities
        shared_guilds = {}
        for guild in client.guilds:
            # Try both methods to find the member
            member = guild.get_member(user_id)
            if not member:
                try:
                    member = await guild.fetch_member(user_id)
                except (discord.NotFound, discord.Forbidden):
                    member = None
        
            if member:  # User is in this guild
                current_personality = guild_personalities.get(guild.id, "default")
            
                # Get personality details
                if current_personality == "default":
                    display_name = DEFAULT_PERSONALITIES["default"]["name"]
                elif guild.id in custom_personalities and current_personality in custom_personalities[guild.id]:
                    personality_data = custom_personalities[guild.id][current_personality]
                    display_name = personality_data["name"]
                else:
                    display_name = "Unknown"
            
                shared_guilds[guild.name.lower()] = {
                    "guild_id": guild.id,
                    "guild_name": guild.name,
                    "personality_key": current_personality,
                    "personality_name": display_name
                }
    
        if not shared_guilds:
            await interaction.followup.send("❌ **No shared servers found!**\n"
                                           "Make sure you're in a server with the bot.")
            return
    
        # If no server name provided, show available options
        if server_name is None:
            embed = discord.Embed(
                title="🎭 Choose DM Personality",
                description="Select which server's personality to use in your DMs:",
                color=0x9932cc
            )
        
            # Get current setting
            current_guild_id = dm_manager.dm_personalities.get(user_id, (None, None))[0]
            current_server = None
            if current_guild_id:
                for guild_data in shared_guilds.values():
                    if guild_data["guild_id"] == current_guild_id:
                        current_server = guild_data["guild_name"]
                        break
        
            if current_server:
                embed.add_field(
                    name="Current Setting",
                    value=f"Using personality from **{current_server}**",
                    inline=False
                )
            else:
                embed.add_field(
                    name="Current Setting",
                    value="Using automatic selection (first shared server found)",
                    inline=False
                )
        
            # List available servers
            server_list = []
            for guild_data in shared_guilds.values():
                server_list.append(f"• **{guild_data['guild_name']}** - {guild_data['personality_name']} (`{guild_data['personality_key']}`)")
        
            embed.add_field(
                name="Available Servers",
                value="\n".join(server_list),
                inline=False
            )
        
            embed.set_footer(text="Use /dm_personality_set <server_name> to choose\nUse /dm_personality_reset to go back to automatic")
            await interaction.followup.send(embed=embed)
            return
    
        # Find the server by name (case-insensitive)
        server_name_lower = server_name.lower()
        selected_guild = None
    
        # Try exact match first
        if server_name_lower in shared_guilds:
            selected_guild = shared_guilds[server_name_lower]
        else:
            # Try partial match
            for guild_name, guild_data in shared_guilds.items():
                if server_name_lower in guild_name:
                    selected_guild = guild_data
                    break
    
        if not selected_guild:
            available_servers = [guild_data["guild_name"] for guild_data in shared_guilds.values()]
            await interaction.followup.send(f"❌ **Server not found!**\n\n"
                                           f"Available servers: {', '.join(available_servers)}\n"
                                           f"Use `/dm_personality_set` without arguments to see all options.")
            return
    
        # Set the DM personality
        dm_manager.dm_personalities[user_id] = (selected_guild["guild_id"], selected_guild["personality_key"])
        dm_manager.save_data()
    
        await interaction.followup.send(f"✅ **DM Personality Set!**\n\n"
                                       f"**Server:** {selected_guild['guild_name']}\n"
                                       f"**Personality:** {selected_guild['personality_name']} (`{selected_guild['personality_key']}`)\n\n"
                                       f"💬 Your DMs will now use this personality!\n"
                                       f"💡 Use `/dm_personality_reset` to go back to automatic selection.")

    @app_commands.command(name="dm_personality_reset", description="Reset DM personality to automatic selection")
    async def dm_personality_reset(self, interaction: discord.Interaction):
        """Reset DM personality to automatic selection"""
        await interaction.response.defer(ephemeral=True)
    
        user_id = interaction.user.id
    
        if user_id in dm_manager.dm_personalities:
            del dm_manager.dm_personalities[user_id]
            dm_manager.save_data()
            await interaction.followup.send("✅ **DM Personality Reset!**\n\n"
                                           "Your DMs will now automatically use the personality from the first shared server found.\n"
                                           "Use `/dm_personality_set` to choose a specific server's personality.")
        else:
            await interaction.followup.send("✅ **Already using automatic selection!**\n\n"
                                           "Your DMs automatically use personalities from shared servers.\n"
                                           "Use `/dm_personality_set` to choose a specific server's personality.")

    @dm_personality_set.autocomplete('server_name')
    async def server_name_autocomplete(self, interaction: discord.Interaction, current: str):
        """Autocomplete for server names"""
        user_id = interaction.user.id
        shared_servers = []
    
        for guild in client.guilds:
            member = guild.get_member(user_id)
            if not member:
                try:
                    member = await guild.fetch_member(user_id)
                except (discord.NotFound, discord.Forbidden):
                    continue
        
            if member and current.lower() in guild.name.lower():
                shared_servers.append(app_commands.Choice(name=guild.name, value=guild.name))
    
        return shared_servers[:25]  # Discord limits to 25 choices

    @app_commands.command(name="dm_history_toggle", description="Toggle loading full DM conversation history (DMs only)")
    async def dm_history_toggle(self, interaction: discord.Interaction, enabled: bool = None):
        """Toggle full DM history loading"""
        await interaction.response.defer(ephemeral=True)
    
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.followup.send("❌ This command only works in DMs!")
            return
    
        if enabled is None:
            # Show current status
            current_status = dm_manager.is_dm_full_history_enabled(interaction.user.id)
            status_text = "✅ Enabled" if current_status else "❌ Disabled"
            await interaction.followup.send(f"**DM Full History Loading:** {status_text}\n\n"
                                           f"When enabled, I'll load our entire conversation history (up to token limits) so I remember everything we've talked about, even across bot restarts.\n\n"
                                           f"**Privacy Note:** This only reads existing DM messages - nothing is saved to files.\n\n"
                                           f"Use `/dm_history_toggle true` to enable or `/dm_history_toggle false` to disable.")
            return
    
        dm_manager.set_dm_full_history(interaction.user.id, enabled)
    
        if enabled:
            await interaction.followup.send("✅ **DM Full History Loading Enabled!**\n\n"
                                           "I'll now load our complete conversation history each time we chat, so I remember everything we've discussed.\n\n"
                                           "**Benefits:**\n"
                                           "• I remember past conversations even after restarts\n"
                                           "• Better context and continuity\n"
                                           "• More personalized responses\n\n"
                                           "**Privacy:** Only reads existing messages, nothing is saved to files.\n\n"
                                           "💡 You can disable this anytime with `/dm_history_toggle false`")
        else:
            await interaction.followup.send("❌ **DM Full History Loading Disabled.**\n\n"
                                           "I'll only remember our recent conversation (standard behavior).\n"
                                           "You can re-enable this anytime with `/dm_history_toggle true`")

    @app_commands.command(name="dm_edit_last", description="Edit the bot's last message in this DM (DMs only)")
    async def dm_edit_last_message(self, interaction: discord.Interaction, new_content: str):
        """Edit the bot's last logical message (DMs only)"""
        await interaction.response.defer(ephemeral=True)
    
        # Check if this is a DM
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.followup.send("❌ This command only works in DMs! Bot message editing is not allowed in servers to maintain natural conversation flow.")
            return
    
        try:
            # Get the bot's last logical message (handles multipart responses)
            messages_to_edit, original_content = await get_bot_last_logical_message(interaction.channel)
        
            if not messages_to_edit:
                await interaction.followup.send("❌ No recent bot message found to edit!")
                return
        
            # Delete ALL the old messages (entire logical response)
            for msg in messages_to_edit:
                try:
                    await msg.delete()
                except:
                    pass
        
            # Send the new content (split by newlines if needed)
            message_parts = split_message_by_newlines(new_content)
        
            sent_messages = []
            if len(message_parts) > 1:
                for part in message_parts:
                    if len(part) > 4000:
                        for i in range(0, len(part), 4000):
                            sent_msg = await interaction.channel.send(part[i:i+4000])
                            sent_messages.append(sent_msg)
                    else:
                        sent_msg = await interaction.channel.send(part)
                        sent_messages.append(sent_msg)
            else:
                if len(new_content) > 4000:
                    for i in range(0, len(new_content), 4000):
                        sent_msg = await interaction.channel.send(new_content[i:i+4000])
                        sent_messages.append(sent_msg)
                else:
                    sent_msg = await interaction.channel.send(new_content)
                    sent_messages.append(sent_msg)
        
            # Store as multipart response if needed
            if len(sent_messages) > 1:
                store_multipart_response(interaction.channel.id, [msg.id for msg in sent_messages], new_content)
        
            # Update conversation history (for DMs, check if using full history)
            if dm_manager.is_dm_full_history_enabled(interaction.user.id):
                await interaction.followup.send("✅ **Message edited!** The edit is automatically reflected in conversation history.")
            else:
                # Update stored conversation history
                if interaction.channel.id in conversations:
                    history = conversations[interaction.channel.id]
                
                    # Find the most recent assistant message and update it
                    for i in range(len(history) - 1, -1, -1):
                        if history[i]["role"] == "assistant":
                            history[i]["content"] = new_content
                            break
            
                await interaction.followup.send("✅ **Message edited and conversation history updated!**")
        
        except Exception as e:
            await interaction.followup.send(f"❌ Error editing message: {str(e)}")

    @app_commands.command(name="dm_regenerate", description="Regenerate the bot's last response with a different answer (DMs only)")
    async def dm_regenerate_last_response(self, interaction: discord.Interaction):
        """Regenerate the bot's last logical message with a new response (DMs only)"""
        await interaction.response.defer(ephemeral=True)
    
        # Check if this is a DM
        if not isinstance(interaction.channel, discord.DMChannel):
            await interaction.followup.send("❌ This command only works in DMs! Message regeneration is not allowed in servers to maintain natural conversation flow.")
            return
    
        try:
            # Get the bot's last logical message
            messages_to_delete, original_content = await get_bot_last_logical_message(interaction.channel)
        
            if not messages_to_delete:
                await interaction.followup.send("❌ No recent bot message found to regenerate!")
                return
        
            # Find the user message that triggered this response
            user_message_before = None
            oldest_bot_message = min(messages_to_delete, key=lambda m: m.created_at)
        
            async for message in interaction.channel.history(limit=50, before=oldest_bot_message):
                if message.author != client.user:
                    user_message_before = message
                    break
        
            if not user_message_before:
                await interaction.followup.send("❌ Couldn't find the user message that triggered the bot's response!")
                return
        
            # Delete all the old bot messages
            for msg in messages_to_delete:
                try:
                    await msg.delete()
                except:
                    pass
        
            # Get user info
            user_name = interaction.user.display_name if hasattr(interaction.user, 'display_name') else interaction.user.name
            guild_id = None
        
            # Get guild from selected server or shared guild
            selected_guild_id = dm_server_selection.get(interaction.user.id)
            if selected_guild_id:
                guild = client.get_guild(selected_guild_id)
                guild_id = selected_guild_id
            else:
                guild = get_shared_guild(interaction.user.id)
                if guild:
                    guild_id = guild.id
        
            # If using regular history, remove the old bot response
            if not dm_manager.is_dm_full_history_enabled(interaction.user.id):
                if interaction.channel.id in conversations:
                    history = conversations[interaction.channel.id]
                    if history and history[-1]["role"] == "assistant":
                        history.pop()
        
            # Generate a new response
            async with interaction.channel.typing():
                new_response = await generate_response(
                    interaction.channel.id,
                    user_message_before.content,
                    guild,
                    user_message_before.attachments,
                    user_name,
                    is_dm=True,
                    user_id=interaction.user.id,
                    original_message=user_message_before
                )
        
            # Send the new response
            if new_response:
                message_parts = split_message_by_newlines(new_response)
            
                sent_messages = []
                if len(message_parts) > 1:
                    for part in message_parts:
                        if len(part) > 4000:
                            for i in range(0, len(part), 4000):
                                sent_msg = await interaction.channel.send(part[i:i+4000])
                                sent_messages.append(sent_msg)
                        else:
                            sent_msg = await interaction.channel.send(part)
                            sent_messages.append(sent_msg)
                else:
                    if len(new_response) > 4000:
                        for i in range(0, len(new_response), 4000):
                            sent_msg = await interaction.channel.send(new_response[i:i+4000])
                            sent_messages.append(sent_msg)
                    else:
                        sent_msg = await interaction.channel.send(new_response)
                        sent_messages.append(sent_msg)
            
                # Store as multipart response if needed
                if len(sent_messages) > 1:
                    store_multipart_response(interaction.channel.id, [msg.id for msg in sent_messages], new_response)
        
            await interaction.followup.send("✅ **Response regenerated!** I've created a new response to your previous message.")
        
        except Exception as e:
            await interaction.followup.send(f"❌ Error regenerating response: {str(e)}")

    @app_commands.command(name="dm_enable", description="Enable/disable DMs with the bot for all server members (Admin only)")
    async def toggle_dm_enable(self, interaction: discord.Interaction, enabled: bool = None):
        """Enable or disable DMs for all server members"""
        await interaction.response.defer(ephemeral=True)
    
        if not interaction.guild:
            await interaction.followup.send("❌ This command can only be used in servers!")
            return
    
        # Check admin permissions
        if not check_admin_permissions(interaction):
            await interaction.followup.send("❌ Only administrators can control DM settings!")
            return
    
        if enabled is None:
            # Show current status
            current_status = guild_dm_enabled.get(interaction.guild.id, True)  # Default to enabled
            status_text = "✅ Enabled" if current_status else "❌ Disabled"
        
            await interaction.followup.send(f"**DM Status for {interaction.guild.name}:** {status_text}\n\n"
                                           f"When enabled, server members can DM the bot directly.\n"
                                           f"When disabled, the bot will inform users that DMs are not allowed and to contact server admins.\n\n"
                                           f"Use `/dm_enable true` to enable or `/dm_enable false` to disable.")
            return
    
        # Set the DM enabled status
        guild_dm_enabled[interaction.guild.id] = enabled
        save_json_data(DM_ENABLED_FILE, guild_dm_enabled)
    
        if enabled:
            await interaction.followup.send(f"✅ **DMs Enabled for {interaction.guild.name}!**\n\n"
                                           f"Server members can now DM the bot directly.\n"
                                           f"💡 Use `/dm_enable false` to disable DMs in the future.")
        else:
            await interaction.followup.send(f"❌ **DMs Disabled for {interaction.guild.name}!**\n\n"
                                           f"Server members will be told DMs are disabled and to contact server admins.\n"
                                           f"💡 Use `/dm_enable true` to re-enable DMs in the future.")

    @app_commands.command(name="lore_auto_update", description="Let the bot update lore entries based on what it learned (Admin only)")
    async def lore_auto_update(self, interaction: discord.Interaction, member: str = None):
        """Let the bot automatically update lore based on conversation history"""
        await interaction.response.defer(ephemeral=True)
    
        is_dm = isinstance(interaction.channel, discord.DMChannel)
    
        if is_dm:
            # DM mode - update personal lore
            user_id = interaction.user.id  # Define user_id here
            history = get_conversation_history(interaction.channel.id)
            if not history:
                await interaction.followup.send("❌ No conversation history found to analyze!")
                return
        
            # Get existing lore
            existing_lore = lore_book.get_dm_entry(user_id)
        
            # Create instruction for AI to analyze and update lore
            lore_instruction = f"""Analyze the conversation history and update the user's lore entry with new information you've learned about them.

    Current lore: {existing_lore if existing_lore else "No existing lore"}

    Instructions:
    - Extract key information about the user (personality, interests, background, preferences, etc.).
    - Merge new information with existing lore, don't duplicate.
    - Keep it concise and relevant for future conversations.
    - Format as a brief character description (max 300 characters).
    - Only include factual information explicitly mentioned by the user."""

            # Generate updated lore
            update_prompt = f"Based on our conversation, create an updated lore entry about {interaction.user.display_name}."
        
            # Use the AI to generate the update
            temp_messages = [{"role": "user", "content": update_prompt}]
        
            guild = get_shared_guild(user_id)
            guild_id = guild.id if guild else None

            # Use appropriate guild ID for temperature
            temp_guild_id = guild.id if guild else (dm_server_selection.get(user_id) if user_id else None)
            if not temp_guild_id and user_id:
                shared_guild = get_shared_guild(user_id)
                temp_guild_id = shared_guild.id if shared_guild else None
        
            updated_lore = await ai_manager.generate_response(
                messages=temp_messages,
                system_prompt=lore_instruction,
                temperature=get_temperature(temp_guild_id) if temp_guild_id else 1.0,
                user_id=user_id,
                guild_id=guild_id,
                is_dm=is_dm,
                max_tokens=500
            )
        
            if updated_lore and not updated_lore.startswith("❌"):
                # Show preview and confirm
                embed = discord.Embed(
                    title="📖 Auto-Generated Lore Update",
                    description="Based on our conversation, here's what I've learned:",
                    color=0x00ff99
                )
            
                if existing_lore:
                    embed.add_field(name="Current Lore", value=existing_lore[:300] + "..." if len(existing_lore) > 300 else existing_lore, inline=False)
            
                embed.add_field(name="Updated Lore", value=updated_lore[:300] + "..." if len(updated_lore) > 300 else updated_lore, inline=False)
            
                # Actually update the lore
                lore_book.add_dm_entry(user_id, updated_lore)
            
                embed.set_footer(text="✅ Lore has been updated! Use /lore_view to see the full entry.")
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send("❌ Failed to generate lore update.")
            
        else:
            # Server mode
            if not check_admin_permissions(interaction):
                await interaction.followup.send("❌ Only administrators can use auto-update in servers!")
                return
            
            if member is None:
                await interaction.followup.send("❌ Please specify a member to update lore for.\n"
                                               "**Usage:** `/lore_auto_update <member>`")
                return
        
            # Find member (same logic as other lore commands)
            member_obj = None
            if member.startswith('<@') and member.endswith('>'):
                user_id_str = member.strip('<@!>')
                try:
                    member_obj = interaction.guild.get_member(int(user_id_str))
                except ValueError:
                    pass
            elif member.isdigit():
                member_obj = interaction.guild.get_member(int(member))
            else:
                member_lower = member.lower()
                for guild_member in interaction.guild.members:
                    if (guild_member.name.lower() == member_lower or 
                        guild_member.display_name.lower() == member_lower):
                        member_obj = guild_member
                        break
        
            if member_obj is None:
                await interaction.followup.send(f"❌ Member '{member}' not found!")
                return
        
            # Check if member has participated in recent conversations
            if interaction.channel.id not in recent_participants or member_obj.id not in recent_participants[interaction.channel.id]:
                await interaction.followup.send(f"❌ {member_obj.display_name} hasn't participated in recent conversations in this channel!")
                return
        
            # Get conversation history
            history = get_conversation_history(interaction.channel.id)
        
            # Filter messages involving this member
            member_messages = []
            for msg in history:
                if msg["role"] == "user" and isinstance(msg["content"], str):
                    if f"<@{member_obj.id}>" in msg["content"] or member_obj.display_name in msg["content"]:
                        member_messages.append(msg["content"])
        
            if not member_messages:
                await interaction.followup.send(f"❌ No messages from {member_obj.display_name} found in recent history!")
                return
        
            # Get existing lore
            existing_lore = lore_book.get_entry(interaction.guild.id, member_obj.id)
        
            # Create instruction for AI
            recent_activity = "\n".join(member_messages[-10:])  # Last 10 messages
            lore_instruction = f"""Analyze {member_obj.display_name}'s messages and update their lore entry.

    Current lore: {existing_lore if existing_lore else "No existing lore"}

    Recent messages from {member_obj.display_name}:
    {recent_activity}

    Instructions:
    - Extract key information about them (personality, interests, role in server, etc.).
    - Merge with existing lore, don't duplicate.
    - Keep it concise and relevant.
    - Format as a brief character description (max 500 characters).
    - Only include factual information from their messages."""

            # Generate updated lore
            update_prompt = f"Based on the conversation, create an updated lore entry about {member_obj.display_name}."
        
            temp_messages = [{"role": "user", "content": update_prompt}]
        
            # Use the guild temperature for server mode
            temp_guild_id = interaction.guild.id
        
            updated_lore = await ai_manager.generate_response(
                messages=temp_messages,
                system_prompt=lore_instruction,
                temperature=get_temperature(temp_guild_id) if temp_guild_id else 1.0,
                guild_id=interaction.guild.id,
                is_dm=False,
                max_tokens=500
            )
        
            if updated_lore and not updated_lore.startswith("❌"):
                # Show preview
                embed = discord.Embed(
                    title=f"📖 Auto-Generated Lore Update for {member_obj.display_name}",
                    description="Based on recent conversations:",
                    color=0x00ff99
                )
            
                if existing_lore:
                    embed.add_field(name="Current Lore", value=existing_lore[:300] + "..." if len(existing_lore) > 300 else existing_lore, inline=False)
            
                embed.add_field(name="Updated Lore", value=updated_lore[:300] + "..." if len(updated_lore) > 300 else updated_lore, inline=False)
            
                # Update the lore
                lore_book.add_entry(interaction.guild.id, member_obj.id, updated_lore)
            
                embed.set_footer(text="✅ Lore has been updated!")
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send("❌ Failed to generate lore update.")

async def setup(bot: commands.Bot):
    await bot.add_cog(CommandsCog(bot))
