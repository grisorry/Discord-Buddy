import asyncio
import re
import discord
from discord.ext import commands
from discord_buddy.core import *

LOW_EFFORT_WORDS = {
    "ok", "okay", "k", "lol", "lmao", "lmfao", "thx", "thanks", "ty",
    "yup", "nope", "idk", "brb", "gtg", "sure", "cool", "nice", "hmm", "meh"
}

OPEN_TRIGGERS = (
    "anyone", "anybody", "someone", "does anyone", "does anybody",
    "can someone", "can anybody", "any tips", "any advice", "any suggestions",
    "recommendations", "thoughts", "what do you think", "ideas", "help"
)


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


class EventsCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_ready(self):
        """Bot startup initialization and status logging"""
        # Display AI provider status
        providers_status = ai_manager.get_available_providers()
        for provider, available in providers_status.items():
            status = "✅ Available" if available else "❌ No API Key"
    
        # Restore bot activity if set
        if custom_activity:
            try:
                parts = custom_activity.split(' ', 1)
                if len(parts) == 2:
                    activity_type, status_text = parts
                    activity_map = {
                        "playing": lambda text: discord.Game(name=text),
                        "watching": lambda text: discord.Activity(type=discord.ActivityType.watching, name=text),
                        "listening": lambda text: discord.Activity(type=discord.ActivityType.listening, name=text),
                        "streaming": lambda text: discord.Streaming(name=text, url="https://twitch.tv/placeholder"),
                        "competing": lambda text: discord.Activity(type=discord.ActivityType.competing, name=text)
                    }
                    if activity_type.lower() in activity_map:
                        activity = activity_map[activity_type.lower()](status_text)
                        await client.change_presence(activity=activity)
            except Exception:
                pass
    
        # Start the background task for DM check-ups
        asyncio.create_task(check_up_task())
    
        # Load plugins before syncing commands
        try:
            await load_plugins()
        except Exception as e:
            print(f"Plugin loading failed: {e}")

        await self.bot.tree.sync()
        for guild in self.bot.guilds:
            try:
                await self.bot.tree.sync(guild=guild)
            except Exception as e:
                print(f"Guild sync failed for {guild.id}: {e}")
        print("Bot is ready!")

    @commands.Cog.listener()
    async def on_guild_join(self, guild: discord.Guild):
        """Handle bot being added to a new server"""
        # Find who added the bot (if possible)
        async for entry in guild.audit_logs(action=discord.AuditLogAction.bot_add, limit=1):
            if entry.target == client.user:
                # Send welcome DM to the user who added the bot
                await send_welcome_dm(entry.user)
                break
        else:
            # Fallback: try to send to guild owner
            if guild.owner:
                await send_welcome_dm(guild.owner)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """Handle incoming messages and determine response behavior"""
        # Skip ONLY this bot's own messages and commands
        if message.author == client.user or message.content.startswith('/'):
            return
    
        is_dm = isinstance(message.channel, discord.DMChannel)
        mentions_bot = client and client.user in message.mentions

        # Check if DMs are allowed for this user
        if is_dm and not message.author.bot:
            # Find a shared guild to check DM permissions
            user_shared_guilds = []
            for guild in client.guilds:
                member = guild.get_member(message.author.id)
                if member:
                    user_shared_guilds.append(guild)
        
            # Check if any shared guild has DMs disabled
            dm_blocked = False
            blocking_guild = None
        
            for guild in user_shared_guilds:
                if not guild_dm_enabled.get(guild.id, True):  # Default to enabled
                    dm_blocked = True
                    blocking_guild = guild
                    break
        
            if dm_blocked:
                # Send DM disabled message
                embed = discord.Embed(
                    title="🔒 DMs Disabled",
                    description=f"Sorry, but DMs with this bot have been disabled by the administrators of **{blocking_guild.name}**.",
                    color=0xff4444
                )
                embed.add_field(
                    name="What can you do?",
                    value=f"• Contact the administrators of **{blocking_guild.name}** to request DM access\n"
                          f"• Use the bot in the server channels instead\n"
                          f"• Administrators can enable DMs using `/dm_enable true`",
                    inline=False
                )
                embed.set_footer(text="This message is sent to inform you why the bot isn't responding to your DMs.")
            
                try:
                    await message.reply(embed=embed)
                except:
                    # Fallback if embed fails
                    await message.reply(f"🔒 **DMs Disabled**\n\n"
                                              f"Sorry, but DMs with this bot have been disabled by the administrators of **{blocking_guild.name}**.\n\n"
                                              f"Please contact the server administrators to request DM access, or use the bot in server channels instead.")
                return

        # Handle other bots as "users" for roleplay scenarios
        is_other_bot = message.author.bot and message.author != client.user
    
        # Get proper display name (works for both users and other bots)
        if hasattr(message.author, 'display_name') and message.author.display_name:
            user_name = message.author.display_name
        elif hasattr(message.author, 'global_name') and message.author.global_name:
            user_name = message.author.global_name
        else:
            user_name = message.author.name

        # Check if this message is a reply to another message
        reply_to_name = None
        if message.reference and message.reference.resolved:
            replied_message = message.reference.resolved
            replied_author = getattr(replied_message, "author", None)
            if replied_author:
                if hasattr(replied_author, 'display_name') and replied_author.display_name:
                    reply_to_name = replied_author.display_name
                elif hasattr(replied_author, 'global_name') and replied_author.global_name:
                    reply_to_name = replied_author.global_name
                else:
                    reply_to_name = replied_author.name

        guild_id = message.guild.id if message.guild else None

        # Update DM interaction tracking (only for real users, not other bots)
        if is_dm and not is_other_bot and dm_manager.is_dm_toggle_enabled(message.author.id):
            dm_manager.update_last_interaction(message.author.id)

        # Track participants for lore activation (servers only) - include other bots
        if not is_dm and message.channel.id not in recent_participants:
            recent_participants[message.channel.id] = set()
        if not is_dm:
            recent_participants[message.channel.id].add(message.author.id)

        # Process voice messages (skip for other bots since they don't send voice)
        voice_text = None
        voice_message_detected = False
        if not is_other_bot and message.attachments:
            for attachment in message.attachments:
                if any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
                    voice_message_detected = True
                    async with message.channel.typing():
                        voice_text = await process_voice_message(attachment)
                        if voice_text:
                            break

        # Process stickers (skip for other bots)
        sticker_info = None
        if not is_other_bot and message.stickers:
            # Get the first sticker (messages can have multiple but usually just one)
            sticker = message.stickers[0]
            sticker_info = f"{user_name} sent a sticker: '{sticker.name}' ({sticker.format.name})"

        # Determine if bot should respond
        should_respond = False
        autonomous_trigger = False
        direct_trigger = False
        is_reply_to_bot = False
        if message.reference and message.reference.resolved:
            try:
                replied_author = getattr(message.reference.resolved, "author", None)
                is_reply_to_bot = replied_author == client.user if replied_author else False
            except Exception:
                is_reply_to_bot = False
    
        # EXPLICIT CHECK: Never respond to our own messages (double protection)
        if message.author == client.user:
            should_respond = False
        # Respond to mentions, DMs, voice messages (but NOT @here or @everyone)
        elif (client.user.mentioned_in(message) and not message.mention_everyone) or is_dm or voice_message_detected or is_reply_to_bot:
            should_respond = True
            direct_trigger = True
        # Autonomous responses (with explicit protection against own messages)
        elif (guild_id and 
              message.author != client.user and  # EXPLICIT PROTECTION
              autonomous_manager.should_respond_autonomously(guild_id, message.channel.id)):
            should_respond = True
            autonomous_trigger = True

        if should_respond and autonomous_trigger and not direct_trigger:
            normalized = _normalize_text(message.content)
            open_question = _looks_open_question(normalized)
            low_effort = _is_low_effort(normalized)

            replied_to_other = False
            if message.reference and message.reference.resolved:
                replied_to_other = True

            mentions_other = False
            if message.mentions:
                mentions_other = any(m != client.user for m in message.mentions)

            mention_everyone = bool(getattr(message, "mention_everyone", False))
            mention_role = bool(getattr(message, "role_mentions", []))

            if replied_to_other:
                should_respond = False
                autonomous_trigger = False
            elif mentions_other and not (mention_everyone or mention_role):
                should_respond = False
                autonomous_trigger = False
            elif low_effort and not (open_question or mention_everyone or mention_role):
                should_respond = False
                autonomous_trigger = False

        if should_respond:
            meta_tags = build_message_meta_tags(message)
            # Handle voice messages privately (only for real users)
            if voice_text and not is_other_bot:
                content = f"{user_name} sent you a voice message, transcript: {voice_text}"
            elif sticker_info and not is_other_bot:
                content = sticker_info
            else:
                # For other bots, use their message content directly without removing mentions
                if is_other_bot:
                    content = message.content.strip()
                else:
                    # Replace bot mention with bot's display name
                    bot_display_name = message.guild.me.display_name if message.guild else client.user.display_name
                    content = message.content.replace(f'<@{client.user.id}>', bot_display_name).strip()
                
                    if not content and not message.attachments and not voice_message_detected and not message.stickers:
                        content = "Hello!" if not is_other_bot else f"{user_name} sent a message."
                    elif not content and voice_message_detected and not voice_text and not is_other_bot:
                        content = f"{user_name} sent you a voice message, but I couldn't transcribe it."
                    elif not content and message.stickers and not is_other_bot:
                        content = sticker_info

            # Add to request queue instead of processing immediately
            special_instruction = None
            if autonomous_trigger and not direct_trigger:
                special_instruction = (
                    "You are joining an ongoing group discussion via random chance. "
                    "The latest message may not be addressed to you. "
                    "Respond as a participant without assuming it is a direct request. "
                    "If it seems directed to someone else, make a brief neutral interjection "
                    "or ask a short clarifying question."
                )
            elif direct_trigger:
                if re.search(r"\breact\b|\breaction\b|\bemoji\b", content.lower()):
                    special_instruction = (
                        "The user explicitly asked for a reaction. "
                        "Include a [REACT: emoji] tag with an appropriate emoji."
                    )

            added = await request_queue.add_request(
                message.channel.id,
                message,
                content,
                message.guild,
                message.attachments if not is_other_bot else [],
                user_name,
                is_dm,
                message.author.id,
                reply_to_name,  # Pass the reply information
                special_instruction,
                autonomous_trigger,
                meta_tags
            )
        
            if not added:
                # Request was rejected (spam prevention)
                return
            
        else:
            # Add user/bot message to history for context even if not responding
            # BUT SKIP THIS FOR DMs WITH FULL HISTORY ENABLED to prevent duplication
            skip_history = (is_dm and not is_other_bot and dm_manager.is_dm_full_history_enabled(message.author.id))
        
            if not skip_history:
                if voice_text and not is_other_bot:
                    content_for_history = f"{user_name} sent you a voice message, transcript: {voice_text}"
                elif voice_message_detected and not voice_text and not is_other_bot:
                    content_for_history = f"{user_name} sent you a voice message, but it couldn't be transcribed."
                elif sticker_info and not is_other_bot:
                    content_for_history = sticker_info
                else:
                    content_for_history = message.content
                
                await add_to_history(
                    message.channel.id, 
                    "user",  # Treat other bots as "users" in conversation history
                    content_for_history, 
                    message.author.id, 
                    guild_id, 
                    message.attachments if not is_other_bot else [], 
                    user_name,
                    reply_to=reply_to_name
                )

    async def check_up_task(self, ):
        """Background task to send check-up messages to users who haven't been active"""
        await client.wait_until_ready()
    
        while not client.is_closed():
            try:
                users_needing_check_up = dm_manager.get_users_needing_check_up()
            
                for user_id in users_needing_check_up:
                    try:
                        user = client.get_user(user_id)
                        if user:
                            # Get settings from selected server or shared server
                            selected_guild_id = dm_server_selection.get(user_id)
                            if selected_guild_id:
                                shared_guild = client.get_guild(selected_guild_id)
                            else:
                                shared_guild = get_shared_guild(user_id)
                        
                            # Get DM channel
                            if user.dm_channel is None:
                                await user.create_dm()
                            dm_channel = user.dm_channel
                        
                            # Get last 5 messages for context
                            recent_messages = []
                            try:
                                async for message in dm_channel.history(limit=10):
                                    if message.author != client.user:
                                        recent_messages.append(message.content.strip())
                                        if len(recent_messages) >= 5:
                                            break
                            
                                # Reverse to get chronological order
                                recent_messages.reverse()
                            except:
                                recent_messages = []
                        
                            # Create context string
                            if recent_messages:
                                context_messages = "\n".join([f"- {msg}" for msg in recent_messages[-5:]])
                                context_info = f"\nFor context, here are {user.display_name}'s last few messages:\n{context_messages}"
                            else:
                                context_info = ""
                        
                            check_up_instruction = f"[SPECIAL INSTRUCTION]: It's been over 6 hours since you last talked to {user.display_name}. Send them a check-up message asking how they're doing or if they're there. Keep it brief and natural. You can reference recent conversation topics if appropriate. NO REACTS! {context_info}"
                        
                            # Generate contextual check-up message using the proper generate_response function
                            # This will apply the correct prompt type, personality, and all other settings
                            response = await generate_response(
                                dm_channel.id,
                                check_up_instruction,
                                shared_guild,
                                None,  # attachments
                                user.display_name,  # user_name
                                True,  # is_dm
                                user_id,  # user_id
                                None  # original_message
                            )
                        
                            # Send the check-up message
                            if response:
                                if len(response) > 4000:
                                    for i in range(0, len(response), 4000):
                                        await dm_channel.send(response[i:i+4000])
                                else:
                                    await dm_channel.send(response)
                                dm_manager.mark_check_up_sent(user_id)
                        
                            # Small delay to avoid rate limits
                            await asyncio.sleep(2)
                        
                    except Exception:
                        pass
            
            except Exception:
                pass
        
            # Wait 30 minutes before checking again
            await asyncio.sleep(30 * 60)

    async def send_fun_command_response(self, interaction: discord.Interaction, response: str):
        """Helper function to clean and send fun command responses"""
        if response is None:
            return
        
        # Check for error responses and use dismissible error handler
        if response.startswith("❌"):
            await send_dismissible_error(interaction.channel, interaction.user, response)
            return
        
        # Apply the same cleaning pipeline as regular messages
        guild = interaction.guild
    
        # CLEAN BOT NAME PREFIX (remove persona name from output)
        response = clean_bot_name_prefix(response, guild.id if guild else None, interaction.user.id, isinstance(interaction.channel, discord.DMChannel))
    
        # CLEAN EM-DASHES (after bot name cleaning)
        response = clean_em_dashes(response)
    
        # Remove reaction instructions but preserve surrounding spaces
        reaction_pattern = r'\s*\[REACT:\s*([^\]]+)\]\s*'
        cleaned_response = re.sub(reaction_pattern, ' ', response).strip()
        cleaned_response = re.sub(r'  +', ' ', cleaned_response)
    
        # CLEAN EMOJIS (after reactions are processed)
        if cleaned_response:
            cleaned_response = clean_malformed_emojis(cleaned_response, guild)
    
        # Finally sanitize user mentions
        if cleaned_response and not cleaned_response.startswith("❌"):
            cleaned_response = sanitize_user_mentions(cleaned_response, guild)
    
        # Send as single message
        if len(cleaned_response) > 4000:
            for i in range(0, len(cleaned_response), 4000):
                await asyncio.sleep(1.0)  # Add human-like delay
                await interaction.followup.send(cleaned_response[i:i+4000])
        else:
            await asyncio.sleep(1.0)  # Add human-like delay
            await interaction.followup.send(cleaned_response)

    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        """Handle message edits - update conversation history and potentially respond"""
        # Skip bot's own messages and commands
        if after.author == client.user or after.content.startswith('/'):
            return
    
        # Skip if content didn't actually change
        if before.content == after.content:
            return
    
        user_name = after.author.display_name if hasattr(after.author, 'display_name') else after.author.name
        guild_id = after.guild.id if after.guild else None
        is_dm = isinstance(after.channel, discord.DMChannel)
    
        # For DMs with full history enabled, we don't need to update stored history
        # since it loads fresh from Discord each time
        if is_dm and dm_manager.is_dm_full_history_enabled(after.author.id):
            return
    
        # For regular conversations, find and update the message in stored history
        if after.channel.id in conversations:
            history = conversations[after.channel.id]
        
            # Find the user's message in history and update it
            for i in range(len(history) - 1, -1, -1):
                msg = history[i]
                if msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, str):
                        # Check if this message belongs to the editing user
                        if is_dm:
                            # In DMs, just check if the content matches
                            if before.content.strip() in content:
                                # Replace the old content with new content
                                updated_content = content.replace(before.content.strip(), after.content.strip())
                                history[i]["content"] = updated_content
                                break
                        else:
                            # In servers, check for user name in content
                            expected_prefix = f"{user_name} (<@{after.author.id}>):"
                            if content.startswith(expected_prefix) and before.content in content:
                                # Update the content while preserving the username format
                                new_content = content.replace(before.content, after.content)
                                history[i]["content"] = new_content
                                break
        
            # Update DM interaction tracking
            if is_dm and dm_manager.is_dm_toggle_enabled(after.author.id):
                dm_manager.update_last_interaction(after.author.id)

    # Helper function to delete bot messages
    async def delete_bot_messages(self, channel, number: int, exclude_message_ids: set = None) -> int:
        """Delete bot's last N logical messages from channel AND remove them from conversation history"""
        deleted_count = 0
        exclude_message_ids = exclude_message_ids or set()
        deleted_message_ids = []  # Track which messages we actually delete
    
        try:
            # Check permissions first
            if hasattr(channel, 'guild') and channel.guild:
                permissions = channel.permissions_for(channel.guild.me)
                if not permissions.manage_messages:
                    return 0
        
            # Collect all bot messages first (excluding the status messages)
            all_bot_messages = []
            async for message in channel.history(limit=200):
                if (message.author == client.user and 
                    len(message.content.strip()) > 0 and 
                    message.id not in exclude_message_ids):
                    all_bot_messages.append(message)
        
            if not all_bot_messages:
                return 0
        
            # Group messages by timestamp proximity (messages sent within 5 seconds = same logical message)
            logical_messages = []
            current_group = []
        
            for i, message in enumerate(all_bot_messages):
                if not current_group:
                    current_group = [message]
                else:
                    # Check if this message was sent within 5 seconds of the previous one
                    time_diff = abs((message.created_at - current_group[-1].created_at).total_seconds())
                    if time_diff <= 5:
                        current_group.append(message)
                    else:
                        # Start a new group
                        logical_messages.append(current_group)
                        current_group = [message]
        
            # Don't forget the last group
            if current_group:
                logical_messages.append(current_group)
        
            # Delete the requested number of logical messages
            for i, logical_message in enumerate(logical_messages):
                if deleted_count >= number:
                    break
            
                # Delete all messages in this logical group
                success = True
                for j, message in enumerate(logical_message):
                    try:
                        await message.delete()
                        deleted_message_ids.append(message.id)  # Track successful deletions
                        await asyncio.sleep(1.0)  # Rate limit protection
                    except discord.errors.NotFound:
                        deleted_message_ids.append(message.id)  # Consider it deleted
                        pass
                    except discord.errors.Forbidden:
                        success = False
                        break
                    except discord.errors.HTTPException as e:
                        if "rate limited" in str(e).lower():
                            await asyncio.sleep(5.0)
                            try:
                                await message.delete()
                                deleted_message_ids.append(message.id)
                            except Exception:
                                success = False
                        else:
                            success = False
                        continue
                    except Exception:
                        success = False
                        continue
            
                if success:
                    deleted_count += 1
                else:
                    break
        
            # NOW REMOVE FROM CONVERSATION HISTORY
            # Remove the deleted messages from the stored conversation history
            if channel.id in conversations and deleted_message_ids:
                await remove_deleted_messages_from_history(channel.id, deleted_count)
                    
        except Exception:
            pass
    
        return deleted_count

    async def remove_deleted_messages_from_history(self, channel_id: int, logical_messages_deleted: int):
        """Remove the last N assistant messages from conversation history"""
        if channel_id not in conversations:
            return
    
        history = conversations[channel_id]
        assistant_messages_removed = 0
    
        # Go backwards through history and remove assistant messages
        for i in range(len(history) - 1, -1, -1):
            if assistant_messages_removed >= logical_messages_deleted:
                break
            
            if history[i]["role"] == "assistant":
                del history[i]
                assistant_messages_removed += 1
    
        # Clean up any orphaned multipart response tracking
        if channel_id in multipart_responses:
            # Remove multipart entries that no longer have valid messages
            to_remove = []
            for response_id in multipart_responses[channel_id]:
                to_remove.append(response_id)
        
            for response_id in to_remove[-logical_messages_deleted:]:
                if response_id in multipart_responses[channel_id]:
                    del multipart_responses[channel_id][response_id]

    @commands.Cog.listener()
    async def on_message_delete(self, message: discord.Message):
        """Handle when messages are deleted - remove from conversation history if it's a bot message"""
        # Only handle bot's own messages
        if message.author != client.user:
            return
    
        # Remove from conversation history
        if message.channel.id in conversations:
            history = conversations[message.channel.id]
        
            # Find and remove the corresponding message from history
            # We'll match by looking for recent assistant messages and remove the last one
            # This isn't perfect but handles the most common case
            for i in range(len(history) - 1, -1, -1):
                if history[i]["role"] == "assistant":
                    # Check if the content matches (approximately)
                    stored_content = history[i]["content"]
                    if isinstance(stored_content, str):
                        # Simple content matching
                        if len(stored_content) > 50:
                            # For longer messages, check if the first 50 chars match
                            if message.content[:50] in stored_content[:50]:
                                del history[i]
                                break
                        else:
                            # For shorter messages, check exact match
                            if message.content.strip() == stored_content.strip():
                                del history[i]
                                break
                    else:
                        # For complex content (with images), just remove the most recent
                        del history[i]
                        break

    @commands.Cog.listener()
    async def on_bulk_message_delete(self, messages):
        """Handle bulk message deletions"""
        for message in messages:
            if message.author == client.user:
                # Handle each bot message deletion
                await on_message_delete(message)

async def setup(bot: commands.Bot):
    await bot.add_cog(EventsCog(bot))
