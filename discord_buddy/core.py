# Created by Marinara and Claude Sonnet 4
# Shoutout to Il Dottore, my beloved.

# Imports
import datetime
import speech_recognition as sr
import io
import tempfile
import discord
from discord import app_commands
import asyncio
import json
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Set, Tuple, Optional
import aiohttp
import random
import re
import base64
import hashlib
import math
from openai import AsyncOpenAI
from collections import defaultdict
import time
import logging
import warnings
import importlib.util
import sys
from discord_buddy.plugin_system import PluginManager
from discord_buddy.providers.openrouter import (
    post_openrouter_json,
    extract_openrouter_usage,
    extract_openrouter_generation_id,
    extract_responses_output_text,
    extract_chat_output_text,
    extract_responses_reasoning_summary,
    stream_openrouter_responses,
)
from discord_buddy.providers.base import AIProvider
from discord_buddy.providers.claude import ClaudeProvider
from discord_buddy.providers.gemini import GeminiProvider
from discord_buddy.providers.openai_provider import OpenAIProvider
from discord_buddy.providers.token_usage import (
    last_token_usage,
    update_last_token_usage,
    reset_last_token_usage,
)

# Suppress warnings BEFORE importing pydub
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", category=RuntimeWarning)

from pydub import AudioSegment

# Suppress Discord connection warnings/errors
logging.getLogger('discord').setLevel(logging.CRITICAL)
logging.getLogger('aiohttp').setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Only show critical errors
logging.basicConfig(level=logging.CRITICAL)

# Environment setup
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
CLAUDE_API_KEY = os.getenv('CLAUDE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CUSTOM_API_KEY = os.getenv('CUSTOM_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
EMBEDDINGS_ENABLED = os.getenv('EMBEDDINGS_ENABLED', 'true').lower() in ('1', 'true', 'yes', 'on')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'openai/text-embedding-3-small')
try:
    EMBEDDING_DUP_THRESHOLD = float(os.getenv('EMBEDDING_DUP_THRESHOLD', '0.92'))
except ValueError:
    EMBEDDING_DUP_THRESHOLD = 0.92
try:
    EMBEDDING_MEMORY_THRESHOLD = float(os.getenv('EMBEDDING_MEMORY_THRESHOLD', '0.78'))
except ValueError:
    EMBEDDING_MEMORY_THRESHOLD = 0.78
try:
    EMBEDDING_CACHE_MAX = int(os.getenv('EMBEDDING_CACHE_MAX', '500'))
except ValueError:
    EMBEDDING_CACHE_MAX = 500
try:
    EMBEDDING_MEMORY_MAX_CANDIDATES = int(os.getenv('EMBEDDING_MEMORY_MAX_CANDIDATES', '50'))
except ValueError:
    EMBEDDING_MEMORY_MAX_CANDIDATES = 50
try:
    EMBEDDING_MIN_TEXT_CHARS = int(os.getenv('EMBEDDING_MIN_TEXT_CHARS', '12'))
except ValueError:
    EMBEDDING_MIN_TEXT_CHARS = 12
try:
    DUPLICATE_RETRY_MAX = int(os.getenv('DUPLICATE_RETRY_MAX', '1'))
except ValueError:
    DUPLICATE_RETRY_MAX = 1
try:
    RESPONSE_RETRY_MAX = int(os.getenv('RESPONSE_RETRY_MAX', '2'))
except ValueError:
    RESPONSE_RETRY_MAX = 2
try:
    RESPONSE_RETRY_BACKOFF_S = float(os.getenv('RESPONSE_RETRY_BACKOFF_S', '1.0'))
except ValueError:
    RESPONSE_RETRY_BACKOFF_S = 1.0
try:
    RESPONSE_RETRY_BACKOFF_MULT = float(os.getenv('RESPONSE_RETRY_BACKOFF_MULT', '2.0'))
except ValueError:
    RESPONSE_RETRY_BACKOFF_MULT = 2.0
try:
    MAX_CONSECUTIVE_FAILURES = int(os.getenv('MAX_CONSECUTIVE_FAILURES', '8'))
except ValueError:
    MAX_CONSECUTIVE_FAILURES = 8

if not DISCORD_TOKEN:
    print("Error: DISCORD_TOKEN environment variable not set.")
    exit(1)

# Discord client (initialized in main)
client: Optional[discord.Client] = None
tree: Optional[app_commands.CommandTree] = None

def init_bot(bot: discord.Client):
    """Set the global bot client and command tree."""
    global client, tree
    client = bot
    tree = bot.tree

# Data persistence paths
DATA_DIR = "bot_data"
os.makedirs(DATA_DIR, exist_ok=True)
PLUGINS_DIR = "plugins"

# Remove/replace existing prompt-related file paths and variables

DM_TOGGLE_FILE = os.path.join(DATA_DIR, "dm_toggle.json")
DM_LAST_INTERACTION_FILE = os.path.join(DATA_DIR, "dm_last_interaction.json")
DM_LORE_FILE = os.path.join(DATA_DIR, "dm_lore.json")
DM_MEMORIES_FILE = os.path.join(DATA_DIR, "dm_memories.json")
PERSONALITIES_FILE = os.path.join(DATA_DIR, "personalities.json")
HISTORY_LENGTHS_FILE = os.path.join(DATA_DIR, "history_lengths.json")
LORE_FILE = os.path.join(DATA_DIR, "lore.json")
ACTIVITY_FILE = os.path.join(DATA_DIR, "activity.json")
AUTONOMOUS_FILE = os.path.join(DATA_DIR, "autonomous.json")
MEMORIES_FILE = os.path.join(DATA_DIR, "memories.json")
TEMPERATURE_FILE = os.path.join(DATA_DIR, "temperature.json")
WELCOME_SENT_FILE = os.path.join(DATA_DIR, "welcome_sent.json")
DM_SERVER_SELECTION_FILE = os.path.join(DATA_DIR, "dm_server_selection.json")
DM_ENABLED_FILE = os.path.join(DATA_DIR, "dm_enabled.json")
VISION_CACHE_FILE = os.path.join(DATA_DIR, "vision_cache.json")

# New format settings files
FORMAT_SETTINGS_FILE = os.path.join(DATA_DIR, "format_settings.json")
DM_FORMAT_SETTINGS_FILE = os.path.join(DATA_DIR, "dm_format_settings.json")
SERVER_FORMAT_DEFAULTS_FILE = os.path.join(DATA_DIR, "server_format_defaults.json")
NSFW_SETTINGS_FILE = os.path.join(DATA_DIR, "nsfw_settings.json")
DM_NSFW_SETTINGS_FILE = os.path.join(DATA_DIR, "dm_nsfw_settings.json")
CUSTOM_FORMAT_INSTRUCTIONS_FILE = os.path.join(DATA_DIR, "custom_format_instructions.json")
PREFILL_SETTINGS_FILE = os.path.join(DATA_DIR, "prefill_settings.json")
SUMMARY_FILE = os.path.join(DATA_DIR, "summaries.json")
REASONING_SETTINGS_FILE = os.path.join(DATA_DIR, "reasoning_settings.json")
DM_REASONING_SETTINGS_FILE = os.path.join(DATA_DIR, "dm_reasoning_settings.json")
REASONING_EFFORT_FILE = os.path.join(DATA_DIR, "reasoning_effort.json")
DM_REASONING_EFFORT_FILE = os.path.join(DATA_DIR, "dm_reasoning_effort.json")
EMBEDDINGS_SETTINGS_FILE = os.path.join(DATA_DIR, "embeddings_settings.json")
DM_EMBEDDINGS_SETTINGS_FILE = os.path.join(DATA_DIR, "dm_embeddings_settings.json")
CLI_REASONING_STREAM_FILE = os.path.join(DATA_DIR, "cli_reasoning_stream.json")

# Files for old prompt system - TO BE REMOVED
CUSTOM_PROMPTS_FILE = os.path.join(DATA_DIR, "custom_prompts.json")
# Removed: PROMPT_SETTINGS_FILE and DM_PROMPT_SETTINGS_FILE - no longer needed

# AI Provider Classes (externalized into discord_buddy/providers/*.py)

class CustomProvider(AIProvider):
    """Custom provider for local/self-hosted models"""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:1234/v1"):
        super().__init__(api_key)
        self.base_url = base_url
        self._vision_cache = self.load_vision_cache()  # Load from persistent storage
        if api_key:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.base_url
            )
    
    def load_vision_cache(self) -> Dict[str, bool]:
        """Load vision support cache from file"""
        try:
            if os.path.exists(VISION_CACHE_FILE):
                with open(VISION_CACHE_FILE, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def save_vision_cache(self):
        """Save vision support cache to file with error handling"""
        try:
            # Ensure the directory exists
            os.makedirs(DATA_DIR, exist_ok=True)
            
            # Always save, even if cache is empty
            with open(VISION_CACHE_FILE, 'w') as f:
                json.dump(self._vision_cache, f, indent=2)
            
            # print(f"Vision cache saved successfully with {len(self._vision_cache)} entries")
            
        except Exception as e:
            print(f"Error saving vision cache: {e}")
            # Don't raise the error, just log it

    def load_vision_cache(self) -> Dict[str, bool]:
        """Load vision support cache from file with error handling"""
        try:
            if os.path.exists(VISION_CACHE_FILE):
                with open(VISION_CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    # print(f"Vision cache loaded successfully with {len(cache_data)} entries")
                    return cache_data
            else:
                # print("No existing vision cache file found, starting fresh")
                # Create empty file to ensure it exists
                self._vision_cache = {}
                self.save_vision_cache()
                return {}
        except Exception as e:
            # print(f"Error loading vision cache: {e}")
            return {}

    async def supports_vision_dynamic(self, model: str) -> bool:
        """Dynamically check if a model supports vision by testing with a small image"""
        
        # Check if client is available
        if not self.api_key or not hasattr(self, 'client'):
            return False
        
        # Check persistent cache first
        cache_key = f"{self.base_url}:{model}"
        if cache_key in self._vision_cache:
            return self._vision_cache[cache_key]
        
        # print(f"Testing vision support for {model} (first time)...")
        
        try:
            # Create a minimal test message with a tiny image
            test_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Can you see this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            # 1x1 white pixel PNG as base64 (super tiny)
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
                            "detail": "low"
                        }
                    }
                ]
            }]
            
            # Try to make a request with minimal tokens to save costs
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=test_messages,
                    max_tokens=5,  # Very minimal response
                    temperature=0  # Deterministic
                )
                
                # If we get here without error, the model supports vision
                # print(f"✅ {model} supports vision!")
                self._vision_cache[cache_key] = True
                self.save_vision_cache()
                return True
                
            except Exception as e:
                error_str = str(e).lower()
                # print(f"Testing {model} vision support failed: {error_str}")
                
                # Check for specific vision-related errors
                vision_error_indicators = [
                    "vision", "image", "multimodal", "unsupported content type",
                    "invalid content", "image_url not supported", "images are not supported",
                    "does not support images", "visual", "multimedia", "unsupported message type",
                    "content type not supported", "image content is not supported"
                ]
                
                if any(keyword in error_str for keyword in vision_error_indicators):
                    # This suggests the model exists but doesn't support vision
                    # print(f"❌ {model} doesn't support vision (confirmed)")
                    self._vision_cache[cache_key] = False
                    self.save_vision_cache()
                    return False
                else:
                    # For OpenRouter Gemini, let's assume it supports vision if the error isn't vision-specific
                    if "gemini" in model.lower() and self.base_url and "openrouter" in self.base_url.lower():
                        # print(f"✅ Assuming {model} on OpenRouter supports vision")
                        self._vision_cache[cache_key] = True
                        self.save_vision_cache()
                        return True
                    
                    # Other errors might be temporary, don't cache
                    # print(f"⚠️ {model} vision test inconclusive (error: {error_str})")
                    return False
                    
        except Exception as outer_error:
            # print(f"Unexpected error during vision testing for {model}: {outer_error}")
            # For OpenRouter Gemini, assume vision support if we can't test
            if "gemini" in model.lower() and self.base_url and "openrouter" in self.base_url.lower():
                # print(f"✅ Assuming {model} on OpenRouter supports vision (test failed)")
                self._vision_cache[cache_key] = True
                self.save_vision_cache()
                return True
            return False
    
    async def generate_response(self, messages: List[Dict], system_prompt: str, temperature: float = 1.0, model: str = None, max_tokens: int = 8192, reasoning: Optional[dict] = None) -> str:
        if not self.api_key:
            return "? Custom API key not configured. Please contact the bot administrator."
        
        try:
            model = model or self.get_default_model()
            
            formatted_messages = [{"role": "system", "content": system_prompt}]
            has_images = False
            
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                
                if isinstance(content, list):
                    # Complex content with text and images
                    openai_content = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                openai_content.append({
                                    "type": "text",
                                    "text": part["text"]
                                })
                            elif part.get("type") == "image_url":
                                has_images = True
                                # Keep OpenAI format as-is for custom providers
                                openai_content.append(part)
                            elif part.get("type") == "image":
                                has_images = True
                                # Convert other formats to OpenAI format
                                if "source" in part and "data" in part["source"]:
                                    # Claude format to OpenAI
                                    openai_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{part['source']['media_type']};base64,{part['source']['data']}",
                                            "detail": "high"
                                        }
                                    })
                                elif "data" in part and "media_type" in part:
                                    # Gemini format to OpenAI
                                    openai_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{part['media_type']};base64,{part['data']}",
                                            "detail": "high"
                                        }
                                    })
                    
                    if openai_content:
                        formatted_messages.append({"role": role, "content": openai_content})
                
                elif isinstance(content, str) and content.strip():
                    formatted_messages.append({"role": role, "content": content})
            
            # If we have images, test vision support only once
            if has_images:
                supports_vision = await self.supports_vision_dynamic(model)
                if not supports_vision:
                    # Convert images to text descriptions
                    for message in formatted_messages:
                        if isinstance(message.get("content"), list):
                            text_parts = []
                            for part in message["content"]:
                                if part.get("type") == "text":
                                    text_parts.append(part["text"])
                                elif part.get("type") == "image_url":
                                    text_parts.append("[Image was provided but this model doesn't support vision!]")
                            message["content"] = " ".join(text_parts)

            response_text = ""

            def flatten_content(value):
                if isinstance(value, list):
                    parts = []
                    for part in value:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif part.get("type") in ["image_url", "image"]:
                                parts.append("[Image]")
                        elif isinstance(part, str):
                            parts.append(part)
                    return " ".join([p for p in parts if p]).strip()
                return str(value).strip()

            # OpenRouter Responses API when reasoning is enabled
            if "openrouter.ai" in self.base_url.lower() and reasoning:
                try:
                    input_messages = []
                    if system_prompt and system_prompt.strip():
                        input_messages.append({
                            "type": "message",
                            "role": "system",
                            "content": [
                                {"type": "input_text", "text": system_prompt.strip()}
                            ],
                        })

                    for msg in messages:
                        role = msg.get("role", "user")
                        content = flatten_content(msg.get("content", ""))
                        if not content:
                            continue
                        content_type = "input_text" if role != "assistant" else "output_text"
                        input_messages.append({
                            "type": "message",
                            "role": role if role in ["user", "assistant", "system"] else "user",
                            "content": [
                                {"type": content_type, "text": content}
                            ],
                        })

                    payload = {
                        "model": model,
                        "input": input_messages,
                        "reasoning": reasoning,
                        "max_output_tokens": max_tokens,
                        "temperature": temperature
                    }
                    summary_text = ""
                    if cli_reasoning_stream_enabled:
                        payload["stream"] = True
                        summary_streamed = False

                        def on_summary_delta(delta: str) -> None:
                            nonlocal summary_streamed
                            if not summary_streamed:
                                print("\n🧠 Reasoning summary (stream): ", end="", flush=True)
                                summary_streamed = True
                            print(delta, end="", flush=True)

                        data, headers, response_text, summary_text, reasoning_delta_count = await stream_openrouter_responses(
                            self.base_url,
                            self.api_key,
                            "/responses",
                            payload,
                            timeout=60,
                            on_summary_delta=on_summary_delta
                        )
                        if summary_streamed:
                            print()
                        if summary_text and not summary_streamed:
                            print(f"🧠 Reasoning summary: {summary_text}")
                        print(f"🧠 Reasoning deltas seen: {reasoning_delta_count}")
                    else:
                        data, headers = await post_openrouter_json(self.base_url, self.api_key, "/responses", payload, timeout=60)
                        if isinstance(data, dict):
                            response_text = extract_responses_output_text(data)
                            summary_text = extract_responses_reasoning_summary(data)

                    if isinstance(data, dict):
                        usage = extract_openrouter_usage(data)
                        input_tokens = None
                        output_tokens = None
                        total_tokens = None
                        if isinstance(usage, dict):
                            input_tokens = coerce_int(usage.get("input_tokens"))
                            if input_tokens is None:
                                input_tokens = coerce_int(usage.get("prompt_tokens"))
                            output_tokens = coerce_int(usage.get("output_tokens"))
                            if output_tokens is None:
                                output_tokens = coerce_int(usage.get("completion_tokens"))
                            total_tokens = coerce_int(usage.get("total_tokens"))

                        has_usage = input_tokens is not None or output_tokens is not None or total_tokens is not None
                        if has_usage:
                            update_last_token_usage(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                total_tokens=total_tokens
                            )

                        generation_id = extract_openrouter_generation_id(data, headers)
                        if generation_id and not has_usage:
                            await fetch_openrouter_generation_usage(str(generation_id), retries=5, delay_s=0.5)
                except Exception:
                    response_text = ""

            # Fallback to Chat Completions
            if not response_text:
                if "openrouter.ai" in self.base_url.lower():
                    payload = {
                        "model": model,
                        "messages": formatted_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False
                    }
                    data, headers = await post_openrouter_json(self.base_url, self.api_key, "/chat/completions", payload, timeout=60)
                    if isinstance(data, dict):
                        response_text = extract_chat_output_text(data)
                        usage = extract_openrouter_usage(data)

                        input_tokens = None
                        output_tokens = None
                        total_tokens = None
                        if isinstance(usage, dict):
                            input_tokens = coerce_int(usage.get("input_tokens"))
                            if input_tokens is None:
                                input_tokens = coerce_int(usage.get("prompt_tokens"))
                            output_tokens = coerce_int(usage.get("output_tokens"))
                            if output_tokens is None:
                                output_tokens = coerce_int(usage.get("completion_tokens"))
                            total_tokens = coerce_int(usage.get("total_tokens"))

                        has_usage = input_tokens is not None or output_tokens is not None or total_tokens is not None
                        if has_usage:
                            update_last_token_usage(
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                total_tokens=total_tokens
                            )
                        generation_id = extract_openrouter_generation_id(data, headers)
                        if generation_id and not has_usage:
                            await fetch_openrouter_generation_usage(str(generation_id), retries=5, delay_s=0.5)
                else:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=formatted_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False
                    )
                    response_text = response.choices[0].message.content
                    # Capture token usage if available
                    try:
                        if hasattr(response, "usage") and response.usage:
                            update_last_token_usage(
                                input_tokens=getattr(response.usage, "prompt_tokens", None),
                                output_tokens=getattr(response.usage, "completion_tokens", None),
                                total_tokens=getattr(response.usage, "total_tokens", None)
                            )
                    except Exception:
                        pass

            # Check if the response contains proxy or API errors
            if any(error_indicator in response_text.lower() for error_indicator in [
                "proxy error", "upstream connect error", "connection termination", 
                "service unavailable", "context size limit", "request validation failed",
                "tokens.*exceeds", "http 503", "http 400", "http 429", "rate limit", "timeout"
            ]):
                return f"? Custom API error: {response_text}"
            
            # Clean any base64 data from the response
            response_text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}', '[IMAGE DATA REMOVED]', response_text)
            response_text = re.sub(r'[A-Za-z0-9+/=]{100,}', '[BASE64 DATA REMOVED]', response_text)
            
            return response_text
        
        except Exception as e:
            return f"? Custom API error: {str(e)}"
    def get_available_models(self) -> List[str]:
        return [
            "auto-detect",
            "custom-model",
            "local-model"
        ]
    
    def get_default_model(self) -> str:
        return "auto-detect"
    
    def is_available(self) -> bool:
        return bool(self.api_key)

class AIProviderManager:
    """Manages different AI providers and user selections"""
    
    def __init__(self):
        self.providers = {
            "claude": ClaudeProvider(CLAUDE_API_KEY),
            "gemini": GeminiProvider(GEMINI_API_KEY),
            "openai": OpenAIProvider(OPENAI_API_KEY),
            "custom": CustomProvider(CUSTOM_API_KEY),
        }
        
        # Cache for custom providers with different URLs
        self.custom_provider_cache = {}
        
        self.guild_provider_settings: Dict[int, str] = {}
        self.guild_model_settings: Dict[int, str] = {}
        self.guild_custom_urls: Dict[int, str] = {}
        
        self.load_settings()
    
    def get_available_providers(self) -> Dict[str, bool]:
        """Get dictionary of provider names and their availability status"""
        return {
            provider_name: provider.is_available() 
            for provider_name, provider in self.providers.items()
        }
    
    def get_provider_models(self, provider_name: str) -> List[str]:
        """Get available models for a specific provider"""
        if provider_name in self.providers:
            return self.providers[provider_name].get_available_models()
        return []
    
    def get_guild_settings(self, guild_id: int) -> Tuple[str, str]:
        """Get provider and model settings for a guild"""
        provider = self.guild_provider_settings.get(guild_id, "claude")
        model = self.guild_model_settings.get(guild_id)
        
        # If no model is set, use the provider's default
        if not model and provider in self.providers:
            model = self.providers[provider].get_default_model()
        
        return provider, model
    
    def get_guild_custom_url(self, guild_id: int) -> str:
        """Get custom URL for a guild (for custom provider)"""
        return self.guild_custom_urls.get(guild_id, "http://localhost:1234/v1")
    
    def set_guild_provider(self, guild_id: int, provider: str, model: str = None, custom_url: str = None) -> bool:
        """Set provider and model for a guild"""
        try:
            if provider not in self.providers:
                return False
            
            self.guild_provider_settings[guild_id] = provider
            
            if model:
                self.guild_model_settings[guild_id] = model
            else:
                # Use provider's default model
                self.guild_model_settings[guild_id] = self.providers[provider].get_default_model()
            
            if provider == "custom" and custom_url:
                self.guild_custom_urls[guild_id] = custom_url
            
            self.save_settings()
            return True
        except Exception:
            return False
    
    def get_custom_provider(self, custom_url: str) -> CustomProvider:
        """Get or create a custom provider instance for the given URL"""
        if custom_url not in self.custom_provider_cache:
            # print(f"Creating new CustomProvider instance for URL: {custom_url}")
            self.custom_provider_cache[custom_url] = CustomProvider(CUSTOM_API_KEY, custom_url)
        # else:
            # print(f"Reusing existing CustomProvider instance for URL: {custom_url}")
        
        return self.custom_provider_cache[custom_url]
    
    async def generate_response(self, messages: List[Dict], system_prompt: str,
                            temperature: float = 1.0, user_id: int = None,
                            guild_id: int = None, is_dm: bool = False, max_tokens: int = 8192,
                            reasoning: Optional[dict] = None) -> str:
        """Generate response using appropriate provider"""
        
        # For DMs, check if user has selected a specific server, otherwise use shared guild
        if is_dm:
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                provider_name, model_name = self.get_guild_settings(selected_guild_id)
            elif guild_id:
                # Fall back to shared guild's model
                provider_name, model_name = self.get_guild_settings(guild_id)
            else:
                # No provider available
                return "❌ No AI provider is configured. Please ensure you're in a server with the bot that has a configured AI provider. If you are, use `/dm_server_select` here to set it up."
        elif guild_id:
            provider_name, model_name = self.get_guild_settings(guild_id)
        else:
            # No guild context and no provider
            return "❌ No AI provider is configured. Please contact the bot administrator to set up API keys."
        
        # ========== PROVIDER DEBUG LOGGING ==========
        print(f"\n🔌 PROVIDER MANAGER DEBUG:")
        print(f"   Provider: {provider_name}")
        print(f"   Model: {model_name}")
        print(f"   Max Tokens: {max_tokens}")
        print(f"   Messages to send: {len(messages)}")
        
        # Log the exact payload being sent to provider
        if provider_name == "custom":
            url_guild_id = dm_server_selection.get(user_id) if is_dm and user_id in dm_server_selection else guild_id
            custom_url = self.get_guild_custom_url(url_guild_id) if url_guild_id else "http://localhost:1234/v1"
            print(f"   Custom URL: {custom_url}")
        
        print("   📦 Sending to AI provider...")
        # ========== END PROVIDER DEBUG LOGGING ==========

        # Handle custom provider with cached instances
        if provider_name == "custom":
            # For DMs, use the selected server's custom URL, otherwise use current guild
            url_guild_id = dm_server_selection.get(user_id) if is_dm and user_id in dm_server_selection else guild_id
            if url_guild_id:
                custom_url = self.get_guild_custom_url(url_guild_id)
                # Use cached provider instance instead of creating new one
                custom_provider = self.get_custom_provider(custom_url)
                return await custom_provider.generate_response(messages, system_prompt, temperature, model_name, max_tokens, reasoning)
            else:
                # Use default custom provider
                provider = self.providers.get(provider_name)
                return await provider.generate_response(messages, system_prompt, temperature, model_name, max_tokens, reasoning)
        
        # Handle other providers normally
        provider = self.providers.get(provider_name)
        if not provider or not provider.is_available():
            # No fallback - just return error
            return "❌ No AI providers are available. Please contact the bot administrator to configure API keys."
        
        return await provider.generate_response(messages, system_prompt, temperature, model_name, max_tokens, reasoning)
    
    def save_settings(self):
        """Save provider settings to files"""
        try:
            guild_settings = {
                "providers": {str(k): v for k, v in self.guild_provider_settings.items()},
                "models": {str(k): v for k, v in self.guild_model_settings.items()},
                "custom_urls": {str(k): v for k, v in self.guild_custom_urls.items()}
            }
            with open(os.path.join(DATA_DIR, "guild_ai_settings.json"), 'w') as f:
                json.dump(guild_settings, f, indent=2)
                
        except Exception:
            pass
    
    def load_settings(self):
        """Load provider settings from files"""
        try:
            guild_file = os.path.join(DATA_DIR, "guild_ai_settings.json")
            if os.path.exists(guild_file):
                with open(guild_file, 'r') as f:
                    data = json.load(f)
                self.guild_provider_settings = {int(k): v for k, v in data.get("providers", {}).items()}
                self.guild_model_settings = {int(k): v for k, v in data.get("models", {}).items()}
                self.guild_custom_urls = {int(k): v for k, v in data.get("custom_urls", {}).items()}
        except Exception:
            pass

# Initialize AI Provider Manager
ai_manager = AIProviderManager()

def resolve_provider_context(guild_id: int, user_id: Optional[int], is_dm: bool) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve provider/model/custom URL for the current context without throwing."""
    provider_name = None
    model_name = None
    custom_url = None
    try:
        if is_dm and user_id:
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                provider_name, model_name = ai_manager.get_guild_settings(selected_guild_id)
                if provider_name == "custom":
                    custom_url = ai_manager.get_guild_custom_url(selected_guild_id)
            elif guild_id:
                provider_name, model_name = ai_manager.get_guild_settings(guild_id)
                if provider_name == "custom":
                    custom_url = ai_manager.get_guild_custom_url(guild_id)
        elif guild_id:
            provider_name, model_name = ai_manager.get_guild_settings(guild_id)
            if provider_name == "custom":
                custom_url = ai_manager.get_guild_custom_url(guild_id)
    except Exception:
        pass
    return provider_name, model_name, custom_url

# Remove all existing system prompts - they will be replaced with the new structure

# New format style constants
FORMAT_CONVERSATIONAL = "conversational"
FORMAT_ASTERISK = "asterisk"
FORMAT_NARRATIVE = "narrative"

# Valid format styles
VALID_FORMAT_STYLES = [FORMAT_CONVERSATIONAL, FORMAT_ASTERISK, FORMAT_NARRATIVE]

class MemoryManager:
    """Manages conversation memories for contextual recall"""
    def __init__(self):
        self.memories: Dict[int, List[Dict]] = {}
        self.dm_memories: Dict[int, List[Dict]] = {}
        self.load_data()
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for memory indexing"""
        
        # Handle None or empty text
        if not text or not isinstance(text, str):
            return []
        
        # Clean the text and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words and filter
        words = cleaned_text.split()
        
        # Remove common stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
            'what', 'where', 'when', 'why', 'how', 'who', 'which', 'said', 'says', 'just', 'like',
            'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew', 'think', 'thought',
            'take', 'took', 'make', 'made', 'give', 'gave', 'tell', 'told', 'ask', 'asked',
            'discord', 'server', 'channel', 'conversation', 'summary', 'between', 'users', 'bots'
        }
        
        # Filter out stop words, short words, and extract meaningful keywords
        keywords = []
        for word in words:
            if (len(word) >= 3 and 
                word not in stop_words and 
                not word.isdigit() and
                word not in keywords):
                keywords.append(word)
        
        # Limit to reasonable number of keywords
        return keywords[:20]
    
    def save_memory(self, guild_id: int, memory_text: str, keywords: List[str] = None):
        """Save a memory for servers"""
        if guild_id not in self.memories:
            self.memories[guild_id] = []
        
        # Handle None or empty memory_text
        if not memory_text or not isinstance(memory_text, str):
            memory_text = "Empty memory"
        
        if not keywords:
            keywords = self._extract_keywords(memory_text)
        
        memory_entry = {
            "memory": memory_text,
            "keywords": [k.lower() for k in keywords],
            "timestamp": int(time.time())
        }
        
        self.memories[guild_id].append(memory_entry)
        self.save_data()
        return len(self.memories[guild_id]) - 1

    def save_dm_memory(self, user_id: int, memory_text: str, keywords: List[str] = None):
        """Save a memory for DMs"""
        if user_id not in self.dm_memories:
            self.dm_memories[user_id] = []
        
        # Handle None or empty memory_text
        if not memory_text or not isinstance(memory_text, str):
            memory_text = "Empty memory"
        
        if not keywords:
            keywords = self._extract_keywords(memory_text)
        
        memory_entry = {
            "memory": memory_text,
            "keywords": [k.lower() for k in keywords],
            "timestamp": int(time.time())
        }
    
        self.dm_memories[user_id].append(memory_entry)
        self.save_dm_data()
        return len(self.dm_memories[user_id]) - 1

    def save_data(self):
        """Persist server memories to file"""
        try:
            json_data = {str(guild_id): guild_memories for guild_id, guild_memories in self.memories.items()}
            with open(MEMORIES_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass

    def save_dm_data(self):
        """Persist DM memories to file"""
        try:
            json_data = {str(user_id): user_memories for user_id, user_memories in self.dm_memories.items()}
            with open(DM_MEMORIES_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass

    def load_data(self):
        """Load memories from files"""
        try:
            # Load server memories
            if os.path.exists(MEMORIES_FILE):
                with open(MEMORIES_FILE, 'r') as f:
                    json_data = json.load(f)
                self.memories = {int(guild_id_str): guild_memories for guild_id_str, guild_memories in json_data.items()}
            
            # Load DM memories
            if os.path.exists(DM_MEMORIES_FILE):
                with open(DM_MEMORIES_FILE, 'r') as f:
                    json_data = json.load(f)
                self.dm_memories = {int(user_id_str): user_memories for user_id_str, user_memories in json_data.items()}
        except Exception:
            self.memories = {}
            self.dm_memories = {}

    def search_memories(self, guild_id: int, query: str) -> List[Dict]:
        """Search for relevant memories based on query keywords (servers)"""
        if guild_id not in self.memories:
            return []
        
        query_words = [word.lower() for word in query.split()]
        relevant_memories = []
        
        for memory in self.memories[guild_id]:
            for query_word in query_words:
                for keyword in memory["keywords"]:
                    if query_word in keyword or keyword in query_word:
                        if memory not in relevant_memories:
                            relevant_memories.append(memory)
                        break
        
        relevant_memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return relevant_memories[:1]

    def search_dm_memories(self, user_id: int, query: str) -> List[Dict]:
        """Search for relevant memories based on query keywords (DMs)"""
        if user_id not in self.dm_memories:
            return []
        
        query_words = [word.lower() for word in query.split()]
        relevant_memories = []
        
        for memory in self.dm_memories[user_id]:
            for query_word in query_words:
                for keyword in memory["keywords"]:
                    if query_word in keyword or keyword in query_word:
                        if memory not in relevant_memories:
                            relevant_memories.append(memory)
                        break
        
        relevant_memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return relevant_memories[:1]

    def get_all_memories(self, guild_id: int) -> List[Dict]:
        """Get all memories for a guild"""
        return self.memories.get(guild_id, [])

    def get_all_dm_memories(self, user_id: int) -> List[Dict]:
        """Get all memories for a DM"""
        return self.dm_memories.get(user_id, [])

    def delete_all_memories(self, guild_id: int):
        """Delete all memories for a guild"""
        if guild_id in self.memories:
            del self.memories[guild_id]
        self.save_data()

    def delete_all_dm_memories(self, user_id: int):
        """Delete all memories for a DM"""
        if user_id in self.dm_memories:
            del self.dm_memories[user_id]
        self.save_dm_data()

    def edit_memory(self, guild_id: int, memory_index: int, new_memory_text: str) -> bool:
        """Edit a specific memory by index (servers)"""
        if guild_id not in self.memories:
            return False
        
        if not (0 <= memory_index < len(self.memories[guild_id])):
            return False
        
        keywords = self._extract_keywords(new_memory_text)
        self.memories[guild_id][memory_index]["memory"] = new_memory_text
        self.memories[guild_id][memory_index]["keywords"] = [k.lower() for k in keywords]
        self.memories[guild_id][memory_index]["timestamp"] = int(asyncio.get_event_loop().time())
        if "embedding" in self.memories[guild_id][memory_index]:
            del self.memories[guild_id][memory_index]["embedding"]
        
        self.save_data()
        return True

    def edit_dm_memory(self, user_id: int, memory_index: int, new_memory_text: str) -> bool:
        """Edit a specific memory by index (DMs)"""
        if user_id not in self.dm_memories:
            return False
        
        if not (0 <= memory_index < len(self.dm_memories[user_id])):
            return False
        
        keywords = self._extract_keywords(new_memory_text)
        self.dm_memories[user_id][memory_index]["memory"] = new_memory_text
        self.dm_memories[user_id][memory_index]["keywords"] = [k.lower() for k in keywords]
        self.dm_memories[user_id][memory_index]["timestamp"] = int(asyncio.get_event_loop().time())
        if "embedding" in self.dm_memories[user_id][memory_index]:
            del self.dm_memories[user_id][memory_index]["embedding"]
        
        self.save_dm_data()
        return True

    def delete_memory(self, guild_id: int, memory_index: int) -> bool:
        """Delete a specific memory by index (servers)"""
        if guild_id not in self.memories:
            return False
        
        if not (0 <= memory_index < len(self.memories[guild_id])):
            return False
        
        del self.memories[guild_id][memory_index]
        self.save_data()
        return True

    def delete_dm_memory(self, user_id: int, memory_index: int) -> bool:
        """Delete a specific memory by index (DMs)"""
        if user_id not in self.dm_memories:
            return False
        
        if not (0 <= memory_index < len(self.dm_memories[user_id])):
            return False
        
        del self.dm_memories[user_id][memory_index]
        self.save_dm_data()
        return True

    def get_memory_by_index(self, guild_id: int, memory_index: int) -> dict:
        """Get a specific memory by index (servers)"""
        if guild_id not in self.memories:
            return None
        
        if not (0 <= memory_index < len(self.memories[guild_id])):
            return None
        
        return self.memories[guild_id][memory_index]

    def get_dm_memory_by_index(self, user_id: int, memory_index: int) -> dict:
        """Get a specific memory by index (DMs)"""
        if user_id not in self.dm_memories:
            return None
        
        if not (0 <= memory_index < len(self.dm_memories[user_id])):
            return None
        
        return self.dm_memories[user_id][memory_index]

    def save_dm_data(self):
        """Persist DM memories to file"""
        try:
            json_data = {str(user_id): user_memories for user_id, user_memories in self.dm_memories.items()}
            with open(DM_MEMORIES_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass

    def load_data(self):
        """Load memories from files"""
        try:
            # Load server memories
            if os.path.exists(MEMORIES_FILE):
                with open(MEMORIES_FILE, 'r') as f:
                    json_data = json.load(f)
                self.memories = {int(guild_id_str): guild_memories for guild_id_str, guild_memories in json_data.items()}
            
            # Load DM memories
            if os.path.exists(DM_MEMORIES_FILE):
                with open(DM_MEMORIES_FILE, 'r') as f:
                    json_data = json.load(f)
                self.dm_memories = {int(user_id_str): user_memories for user_id_str, user_memories in json_data.items()}
        except Exception:
            self.memories = {}
            self.dm_memories = {}

class LoreBook:
    """Manages character lore entries for server members"""
    def __init__(self):
        self.entries: Dict[int, Dict[str, str]] = {}
        self.dm_entries: Dict[int, str] = {}
        self.load_data()

    def add_entry(self, guild_id: int, user_id: int, lore: str):
        """Add or update lore for a user in servers"""
        if guild_id not in self.entries:
            self.entries[guild_id] = {}
        self.entries[guild_id][user_id] = lore
        self.save_data()

    def add_dm_entry(self, user_id: int, lore: str):
        """Add or update DM-specific lore for a user"""
        self.dm_entries[user_id] = lore
        self.save_dm_data()

    def get_entry(self, guild_id: int, user_id: int) -> str:
        """Get lore for a specific user in servers"""
        return self.entries.get(guild_id, {}).get(user_id, "")

    def get_dm_entry(self, user_id: int) -> str:
        """Get DM-specific lore for a user"""
        return self.dm_entries.get(user_id, "")

    def remove_entry(self, guild_id: int, user_id: int):
        """Remove lore for a user in servers"""
        if guild_id in self.entries and user_id in self.entries[guild_id]:
            del self.entries[guild_id][user_id]
            self.save_data()

    def remove_dm_entry(self, user_id: int):
        """Remove DM-specific lore for a user"""
        if user_id in self.dm_entries:
            del self.dm_entries[user_id]
            self.save_dm_data()

    def save_data(self):
        """Persist server lore data to file"""
        try:
            json_data = {}
            for guild_id, guild_entries in self.entries.items():
                json_data[str(guild_id)] = {str(user_id): lore for user_id, lore in guild_entries.items()}
            
            with open(LORE_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass

    def save_dm_data(self):
        """Persist DM lore data to file"""
        try:
            json_data = {str(user_id): lore for user_id, lore in self.dm_entries.items()}
            with open(DM_LORE_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass

    def load_data(self):
        """Load lore data from files"""
        try:
            # Load server lore
            if os.path.exists(LORE_FILE):
                with open(LORE_FILE, 'r') as f:
                    json_data = json.load(f)
                
                self.entries = {}
                for guild_id_str, guild_entries in json_data.items():
                    guild_id = int(guild_id_str)
                    self.entries[guild_id] = {int(user_id_str): lore for user_id_str, lore in guild_entries.items()}
            
            # Load DM lore
            if os.path.exists(DM_LORE_FILE):
                with open(DM_LORE_FILE, 'r') as f:
                    json_data = json.load(f)
                self.dm_entries = {int(user_id_str): lore for user_id_str, lore in json_data.items()}
        except Exception:
            self.entries = {}
            self.dm_entries = {}

class DMManager:
    """Manages DM-specific features"""
    def __init__(self):
        self.dm_toggle_settings: Dict[int, bool] = {}
        self.last_interactions: Dict[int, float] = {}
        self.pending_check_ups: Set[int] = set()
        self.dm_personalities: Dict[int, tuple] = {}
        self.dm_full_history: Dict[int, bool] = {}
        self.check_up_sent: Dict[int, bool] = {}
        self.load_data()
    
    def set_dm_toggle(self, user_id: int, enabled: bool):
        """Enable/disable auto check-up messages for a user"""
        self.dm_toggle_settings[user_id] = enabled
        if enabled:
            self.update_last_interaction(user_id)
        else:
            self.pending_check_ups.discard(user_id)
        self.save_data()
    
    def is_dm_toggle_enabled(self, user_id: int) -> bool:
        """Check if auto check-up is enabled for a user"""
        return self.dm_toggle_settings.get(user_id, False)
    
    def update_last_interaction(self, user_id: int):
        """Update the last interaction timestamp for a user"""
        self.last_interactions[user_id] = asyncio.get_event_loop().time()
        self.pending_check_ups.discard(user_id)
        # Reset check-up sent flag when user becomes active again
        self.check_up_sent[user_id] = False
        self.save_data()
    
    def get_users_needing_check_up(self) -> List[int]:
        """Get list of users who need a check-up message"""
        current_time = asyncio.get_event_loop().time()
        six_hours = 6 * 60 * 60
        
        users_needing_check_up = []
        
        for user_id, enabled in self.dm_toggle_settings.items():
            if (enabled and 
                user_id not in self.pending_check_ups and
                user_id in self.last_interactions and
                not self.check_up_sent.get(user_id, False) and
                current_time - self.last_interactions[user_id] >= six_hours):
                users_needing_check_up.append(user_id)
        
        return users_needing_check_up
    
    def mark_check_up_sent(self, user_id: int):
        """Mark that a check-up message has been sent"""
        self.pending_check_ups.add(user_id)
        self.check_up_sent[user_id] = True
        self.save_data()
    
    def set_dm_full_history(self, user_id: int, enabled: bool):
        """Enable/disable full history loading for DMs"""
        self.dm_full_history[user_id] = enabled
        self.save_data()
    
    def is_dm_full_history_enabled(self, user_id: int) -> bool:
        """Check if full history loading is enabled for user"""
        return self.dm_full_history.get(user_id, False)
    
    def save_data(self):
        """Persist DM manager data to files"""
        try:
            json_data = {str(user_id): enabled for user_id, enabled in self.dm_toggle_settings.items()}
            with open(DM_TOGGLE_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            json_data = {str(user_id): timestamp for user_id, timestamp in self.last_interactions.items()}
            with open(DM_LAST_INTERACTION_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            dm_personalities_file = os.path.join(DATA_DIR, "dm_personalities.json")
            json_data = {str(user_id): {"guild_id": guild_id, "personality": personality} 
                        for user_id, (guild_id, personality) in self.dm_personalities.items()}
            with open(dm_personalities_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            dm_history_file = os.path.join(DATA_DIR, "dm_full_history.json")
            json_data = {str(user_id): enabled for user_id, enabled in self.dm_full_history.items()}
            with open(dm_history_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            check_up_sent_file = os.path.join(DATA_DIR, "check_up_sent.json")
            json_data = {str(user_id): sent for user_id, sent in self.check_up_sent.items()}
            with open(check_up_sent_file, 'w') as f:
                json.dump(json_data, f, indent=2)
                
        except Exception:
            pass
    
    def load_data(self):
        """Load DM manager data from files"""
        try:
            if os.path.exists(DM_TOGGLE_FILE):
                with open(DM_TOGGLE_FILE, 'r') as f:
                    json_data = json.load(f)
                self.dm_toggle_settings = {int(user_id_str): enabled for user_id_str, enabled in json_data.items()}
            
            if os.path.exists(DM_LAST_INTERACTION_FILE):
                with open(DM_LAST_INTERACTION_FILE, 'r') as f:
                    json_data = json.load(f)
                self.last_interactions = {int(user_id_str): float(timestamp) for user_id_str, timestamp in json_data.items()}
            
            dm_personalities_file = os.path.join(DATA_DIR, "dm_personalities.json")
            if os.path.exists(dm_personalities_file):
                with open(dm_personalities_file, 'r') as f:
                    json_data = json.load(f)
                self.dm_personalities = {int(user_id_str): (data["guild_id"], data["personality"]) 
                                       for user_id_str, data in json_data.items()}
            
            dm_history_file = os.path.join(DATA_DIR, "dm_full_history.json")
            if os.path.exists(dm_history_file):
                with open(dm_history_file, 'r') as f:
                    json_data = json.load(f)
                self.dm_full_history = {int(user_id_str): enabled for user_id_str, enabled in json_data.items()}
            
            check_up_sent_file = os.path.join(DATA_DIR, "check_up_sent.json")
            if os.path.exists(check_up_sent_file):
                with open(check_up_sent_file, 'r') as f:
                    json_data = json.load(f)
                self.check_up_sent = {int(user_id_str): sent for user_id_str, sent in json_data.items()}
            else:
                self.check_up_sent = {}
        except Exception:
            self.dm_toggle_settings = {}
            self.last_interactions = {}
            self.dm_personalities = {}
            self.dm_full_history = {}
            self.check_up_sent = {}

class RequestQueue:
    """Manages queued requests with safe-locking to prevent spam responses"""
    
    def __init__(self):
        self.queues: Dict[int, List] = defaultdict(list)
        self.processing: Dict[int, bool] = defaultdict(bool)
        self.locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        
    async def add_request(self, channel_id: int, message: discord.Message, content: str, 
                        guild: discord.Guild, attachments: List[discord.Attachment],
                        user_name: str, is_dm: bool, user_id: int, reply_to_name: str = None,
                        special_instruction: str = None, autonomous_join: bool = False,
                        meta_tags: Optional[List[str]] = None) -> bool:
        """Add a request to the queue. Returns True if added, False if duplicate/spam"""
        # print(f"DEBUG: add_request called for channel {channel_id}, content={repr(content)}")
        
        async with self.locks[channel_id]:
            # Check for recent duplicate requests from same user (spam prevention)
            current_time = time.time()
            for queued_request in self.queues[channel_id]:
                if (queued_request['user_id'] == user_id and 
                    current_time - queued_request['timestamp'] < 3 and
                    queued_request['content'].strip() == content.strip()):  # Check content similarity
                    return False
            
            # Check if user has too many pending requests
            user_pending_count = sum(1 for req in self.queues[channel_id] if req['user_id'] == user_id)
            if user_pending_count >= 2:  # Limit to 2 pending requests per user
                return False
            
            # Add request to queue
            request = {
                'id': len(self.queues[channel_id]) + int(current_time),
                'timestamp': current_time,
                'message': message,
                'content': content,
                'guild': guild,
                'attachments': attachments,
                'user_name': user_name,
                'is_dm': is_dm,
                'user_id': user_id,
                'reply_to_name': reply_to_name,
                'special_instruction': special_instruction,
                'autonomous_join': autonomous_join,
                'meta_tags': meta_tags or []
            }
            
            self.queues[channel_id].append(request)
            
            # Start processing if not already processing
            if not self.processing[channel_id]:
                asyncio.create_task(self._process_queue(channel_id))
            
            return True
    
    async def _process_queue(self, channel_id: int):
        """Process all requests in the queue for a channel"""
        async with self.locks[channel_id]:
            if self.processing[channel_id]:
                return
            self.processing[channel_id] = True
        
        try:
            while self.queues[channel_id]:
                # Get the next request
                async with self.locks[channel_id]:
                    if not self.queues[channel_id]:
                        break
                    request = self.queues[channel_id].pop(0)
                
                # Process the request
                await self._process_single_request(channel_id, request)
                
                # Small delay between requests to prevent overwhelming
                await asyncio.sleep(0.5)
                
        finally:
            async with self.locks[channel_id]:
                self.processing[channel_id] = False
    
    async def _process_single_request(self, channel_id: int, request: dict):
        """Process a single request with proper context"""
        # print(f"DEBUG: _process_single_request called for channel {channel_id}, content={repr(request.get('content'))}")
        try:
            message = request['message']
            content = request['content']
            guild = request['guild']
            attachments = request['attachments']
            user_name = request['user_name']
            is_dm = request['is_dm']
            user_id = request['user_id']
            reply_to_name = request.get('reply_to_name')
            special_instruction = request.get('special_instruction')
            autonomous_join = bool(request.get('autonomous_join'))
            meta_tags = request.get('meta_tags') or []
            
            async with message.channel.typing():
                # LOAD CHANNEL HISTORY FROM DISCORD
                # Always load fresh history for server channels to ensure full context
                guild_id = guild.id if guild else None
                
                # Load history from Discord for all server channels (not DMs)
                # This ensures the bot always has the complete conversation context
                # Pass the triggering message so it can be excluded from the history load
                if not is_dm:
                    trigger_user_id = None if autonomous_join else user_id
                    await load_channel_history_from_discord(
                        message.channel,
                        guild,
                        channel_id,
                        exclude_message_id=message.id,
                        trigger_user_id=trigger_user_id
                    )
                
                # Add the user's message to history (servers only; DMs are handled in generate_response)
                message_content_override = None
                if not is_dm:
                    message_content_override = await add_to_history(
                        channel_id,
                        "user",
                        content,
                        user_id,
                        guild.id if guild else None,
                        attachments,
                        user_name,
                        reply_to=reply_to_name,
                        meta_tags=meta_tags,
                    )
                    if not isinstance(message_content_override, list):
                        message_content_override = None

                # Check if the last message in history is from the assistant (servers only)
                if not is_dm:
                    current_history = get_conversation_history(channel_id)
                    last_message_is_assistant = current_history and current_history[-1]["role"] == "assistant"
                    
                    # If the last message was from assistant, add continuation prompt
                    if last_message_is_assistant:
                        await add_to_history(
                            channel_id, 
                            "user", 
                            "[Continue the conversation naturally from where you left off.]",
                            user_id=None,
                            guild_id=guild.id if guild else None,
                            user_name=None
                        )
                
                # Generate response using the main generate_response function (includes debug logging)
                user_message_for_ai = content
                if special_instruction:
                    user_message_for_ai = f"{content} [SPECIAL INSTRUCTION]: {special_instruction}"

                bot_response = await generate_response(
                    channel_id=channel_id,
                    user_message=user_message_for_ai,
                    guild=guild,
                    attachments=attachments,
                    user_name=user_name,
                    is_dm=is_dm,
                    user_id=user_id,
                    original_message=message,
                    autonomous_join=autonomous_join,
                    message_content_override=message_content_override
                )

                if bot_response is None:
                    return

                # STORE THE ORIGINAL RESPONSE WITH REACTIONS FOR HISTORY
                original_response_with_reactions = bot_response

                # PROCESS REACTIONS FIRST (this removes [REACT: X] from the response)
                if message:
                    bot_response = await process_and_add_reactions(bot_response, message)

                # If the response is empty after reaction stripping, just skip sending.
                if bot_response is not None:
                    cleaned = str(bot_response).strip()
                    if not cleaned:
                        return

                # THEN CLEAN EMOJIS (after reactions are processed)
                if bot_response and guild:
                    bot_response = clean_malformed_emojis(bot_response, guild)

                # CLEAN EM-DASHES (after emojis are cleaned)
                if bot_response:
                    bot_response = clean_em_dashes(bot_response)

                # CLEAN BOT NAME PREFIX (remove persona name from output)
                if bot_response:
                    bot_response = clean_bot_name_prefix(bot_response, guild.id if guild else None, user_id, is_dm)

                # Assistant response is already added to history in generate_response()
                
                # Finally sanitize user mentions
                if bot_response and not bot_response.startswith("❌"):
                    bot_response = sanitize_user_mentions(bot_response, guild)

                if bot_response is None:
                    return
                
                # Add a small delay to make responses feel more human-like
                await asyncio.sleep(1.0)
                
                # Check if the response is an error that should be temporary
                is_temp_error = bot_response.startswith("[TEMP_ERROR]")
                if is_temp_error:
                    bot_response = bot_response.replace("[TEMP_ERROR] ", "")
                    # Send as dismissible error instead of regular message
                    try:
                        await send_dismissible_error(message.channel, message.author, bot_response)
                        print(f"Sent dismissible error response: {bot_response[:100]}...")
                        return  # Don't continue with normal message sending
                    except Exception as send_error:
                        print(f"Failed to send dismissible error response: {send_error}")
                        # Continue with normal message sending as fallback
                
                # Send the response
                message_parts = split_message_by_newlines(bot_response)
                is_dm = isinstance(message.channel, discord.DMChannel)
                use_reply = not is_dm and not message.author.bot
                
                sent_messages = []
                if len(message_parts) > 1:
                    for part in message_parts:
                        if len(part) > 4000:
                            for i in range(0, len(part), 4000):
                                if use_reply:
                                    sent_msg = await message.reply(part[i:i+4000], delete_after=15.0 if is_temp_error else None)
                                else:
                                    sent_msg = await message.channel.send(part[i:i+4000], delete_after=15.0 if is_temp_error else None)
                                sent_messages.append(sent_msg)
                        else:
                            if use_reply:
                                sent_msg = await message.reply(part, delete_after=15.0 if is_temp_error else None)
                            else:
                                sent_msg = await message.channel.send(part, delete_after=15.0 if is_temp_error else None)
                            sent_messages.append(sent_msg)
                elif bot_response:
                    if len(bot_response) > 4000:
                        for i in range(0, len(bot_response), 4000):
                            if use_reply:
                                sent_msg = await message.reply(bot_response[i:i+4000], delete_after=15.0 if is_temp_error else None)
                            else:
                                sent_msg = await message.channel.send(bot_response[i:i+4000], delete_after=15.0 if is_temp_error else None)
                            sent_messages.append(sent_msg)
                    else:
                        if use_reply:
                            sent_msg = await message.reply(bot_response, delete_after=15.0 if is_temp_error else None)
                        else:
                            sent_msg = await message.channel.send(bot_response, delete_after=15.0 if is_temp_error else None)
                        sent_messages.append(sent_msg)
                
                if len(sent_messages) > 1:
                    store_multipart_response(message.channel.id, [msg.id for msg in sent_messages], bot_response)
                        
        except Exception as e:
            print(f"Error processing request: {e}")
            try:
                error_msg = f"❌ Sorry, I encountered an error processing your request: {str(e)}"
                # Truncate error message to stay under Discord's 4000 character limit
                if len(error_msg) > 3950:  # Leave some buffer
                    error_msg = error_msg[:3950] + "..."
                
                # Check if this is a Discord API error that should be ephemeral/temporary
                is_api_error = ("400 Bad Request" in str(e) or 
                               "error code" in str(e) or 
                               "50035" in str(e) or
                               "Invalid Form Body" in str(e))
                
                if is_api_error:
                    # Send as dismissible error message
                    try:
                        await send_dismissible_error(message.channel, message.author, error_msg)
                        print(f"Sent dismissible error message: {error_msg[:100]}...")
                    except Exception as send_error:
                        print(f"Failed to send dismissible error message: {send_error}")
                        # Fallback to regular temporary message
                        try:
                            temp_msg = await message.channel.send(error_msg, delete_after=15.0)
                        except Exception as fallback_error:
                            print(f"Failed to send fallback error message: {fallback_error}")
                else:
                    # Send regular error message
                    try:
                        await message.channel.send(error_msg)
                    except Exception as send_error:
                        print(f"Failed to send error message: {send_error}")
                        
            except Exception as inner_e:
                print(f"Error in error handling: {inner_e}")
                try:
                    fallback_msg = "❌ Sorry, I encountered an error processing your request."
                    await message.channel.send(fallback_msg)
                except Exception as fallback_error:
                    print(f"Failed to send fallback error message: {fallback_error}")

# Initialize the request queue
request_queue = RequestQueue()

class AutonomousManager:
    """Manages autonomous response behavior settings per channel"""
    def __init__(self):
        self.settings: Dict[int, Dict[int, Dict[str, any]]] = {}
        self.load_data()
    
    def set_channel_autonomous(self, guild_id: int, channel_id: int, enabled: bool, chance: int = 10):
        """Configure autonomous behavior for a channel"""
        if guild_id not in self.settings:
            self.settings[guild_id] = {}
        
        self.settings[guild_id][channel_id] = {
            "enabled": enabled,
            "chance": max(1, min(100, chance))
        }
        self.save_data()
    
    def get_channel_settings(self, guild_id: int, channel_id: int) -> Dict[str, any]:
        """Get autonomous settings for a channel"""
        return self.settings.get(guild_id, {}).get(channel_id, {"enabled": False, "chance": 10})
    
    def should_respond_autonomously(self, guild_id: int, channel_id: int) -> bool:
        """Determine if bot should respond autonomously"""
        settings = self.get_channel_settings(guild_id, channel_id)
        if not settings["enabled"]:
            return False
        return random.randint(1, 100) <= settings["chance"]
    
    def list_autonomous_channels(self, guild_id: int) -> Dict[int, Dict[str, any]]:
        """List all autonomous channels for a guild"""
        return self.settings.get(guild_id, {})
    
    def save_data(self):
        """Persist autonomous settings to file"""
        try:
            json_data = {}
            for guild_id, guild_settings in self.settings.items():
                json_data[str(guild_id)] = {str(channel_id): settings for channel_id, settings in guild_settings.items()}
            
            with open(AUTONOMOUS_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        except Exception:
            pass
    
    def load_data(self):
        """Load autonomous settings from file"""
        try:
            if os.path.exists(AUTONOMOUS_FILE):
                with open(AUTONOMOUS_FILE, 'r') as f:
                    json_data = json.load(f)
                
                self.settings = {}
                for guild_id_str, guild_settings in json_data.items():
                    guild_id = int(guild_id_str)
                    self.settings[guild_id] = {int(channel_id_str): settings for channel_id_str, settings in guild_settings.items()}
        except Exception:
            self.settings = {}

# Utility functions for data persistence
def save_json_data(file_path: str, data: dict, convert_keys=True):
    """Generic function to save dictionary data to JSON file"""
    try:
        if convert_keys:
            json_data = {str(k): v for k, v in data.items()}
        else:
            json_data = data
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    except Exception:
        pass

def load_json_data(file_path: str, convert_keys=True) -> dict:
    """Generic function to load dictionary data from JSON file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            if convert_keys:
                return {int(k): v for k, v in data.items()}
            return data
    except Exception:
        pass
    return {}

loaded_plugins: Dict[str, str] = {}
plugin_errors: Dict[str, str] = {}
plugin_manager = PluginManager()

def log_plugin_hook_error(plugin_name: str, hook_name: str, error: Exception) -> None:
    print(f"⚠️ Plugin hook error [{plugin_name}] {hook_name}: {error}")

plugin_manager.on_error = log_plugin_hook_error

async def load_plugins():
    """Load plugins from the plugins directory."""
    os.makedirs(PLUGINS_DIR, exist_ok=True)
    loaded_plugins.clear()
    plugin_errors.clear()
    plugin_manager.reset()
    
    for filename in os.listdir(PLUGINS_DIR):
        if not filename.endswith(".py") or filename.startswith("_"):
            continue
        
        module_name = os.path.splitext(filename)[0]
        module_path = os.path.join(PLUGINS_DIR, filename)
        full_module_name = f"plugins.{module_name}"
        
        try:
            spec = importlib.util.spec_from_file_location(full_module_name, module_path)
            if spec is None or spec.loader is None:
                plugin_errors[module_name] = "Failed to load module spec"
                continue
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_module_name] = module
            spec.loader.exec_module(module)
            
            plugin_info = getattr(module, "PLUGIN_INFO", None)
            plugin_name = plugin_info.get("name") if isinstance(plugin_info, dict) else module_name

            if hasattr(module, "register_hooks"):
                hooks = plugin_manager.hooks_for(plugin_name, plugin_info if isinstance(plugin_info, dict) else None)
                result = module.register_hooks(hooks)
                if asyncio.iscoroutine(result):
                    await result

            if hasattr(module, "register"):
                result = module.register(tree, client)
                if asyncio.iscoroutine(result):
                    await result
            elif hasattr(module, "setup"):
                result = module.setup(tree, client)
                if asyncio.iscoroutine(result):
                    await result
            elif not hasattr(module, "register_hooks"):
                plugin_errors[module_name] = "No register(tree, client) or setup(tree, client) function found"
                continue
            
            loaded_plugins[module_name] = module_path
        except Exception as e:
            plugin_errors[module_name] = str(e)

def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration."""
    seconds = int(seconds)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)

def save_personalities():
    """Save personality settings to file"""
    save_data = {
        "guild_personalities": guild_personalities,
        "custom_personalities": custom_personalities
    }
    save_json_data(PERSONALITIES_FILE, save_data, convert_keys=False)

def load_personalities():
    """Load personality settings from file"""
    data = load_json_data(PERSONALITIES_FILE, convert_keys=False)
    guild_perss = {int(k): v for k, v in data.get("guild_personalities", {}).items()}
    custom_perss = {int(k): v for k, v in data.get("custom_personalities", {}).items()}
    return guild_perss, custom_perss

def get_shared_guild(user_id: int) -> discord.Guild:
    """Get a guild that both the bot and user are members of"""
    user = client.get_user(user_id)
    if not user:
        return None
    
    for guild in client.guilds:
        member = guild.get_member(user_id)
        if member:
            return guild
        
        if user in guild.members:
            return guild
    
    return None

async def get_shared_guild_async(user_id: int) -> discord.Guild:
    """Async version that can fetch member if not in cache"""
    for guild in client.guilds:
        try:
            member = await guild.fetch_member(user_id)
            if member:
                return guild
        except (discord.NotFound, discord.Forbidden, Exception):
            continue
    
    return None

def get_shared_server_settings(user_id: int) -> tuple:
    """Get settings from a shared server for DM conversations"""
    shared_guild = get_shared_guild(user_id)
    if shared_guild:
        return shared_guild.id, shared_guild
    return None, None

def check_admin_permissions(interaction: discord.Interaction) -> bool:
    """Check if user has administrator permissions"""
    if not interaction.guild:
        return False
    return interaction.user.guild_permissions.administrator

def convert_emojis_to_simple(text: str) -> str:
    """Convert full Discord emoji format <:name:id> to simple :name: format for AI learning"""
    
    # Pattern for animated and static emojis: <a:name:id> or <:name:id>
    emoji_pattern = r'<a?:([a-zA-Z0-9_]+):\d+>'
    
    def replace_emoji(match):
        emoji_name = match.group(1)
        return f":{emoji_name}:"
    
    return re.sub(emoji_pattern, replace_emoji, text)

def remove_thinking_tags(text: str) -> str:
    """Remove thinking tags and their content from AI responses"""
    if not text:
        return text
    
    # Remove thinking tags with various formats: <thinking>, <think>, etc.
    # This regex removes the opening tag, content, and closing tag
    thinking_pattern = r'<(\w*think\w*)[^>]*>.*?</\1>'
    text = re.sub(thinking_pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Also remove self-closing thinking tags
    text = re.sub(r'<(\w*think\w*)[^>]*/>', '', text, flags=re.IGNORECASE)
    
    # Also remove stray/orphan closing tags (e.g. </think>, </thinking>)
    text = re.sub(r'.*?</\w*think\w*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text
    return text

def build_message_meta_tags(message: discord.Message) -> List[str]:
    """Build lightweight metadata tags for a Discord message."""
    tags: List[str] = []
    try:
        if client and message and client.user in message.mentions:
            tags.append("Mentioned you")
    except Exception:
        pass
    try:
        if message and message.mentions:
            other_mentions = []
            for m in message.mentions:
                if client and m == client.user:
                    continue
                if hasattr(m, "display_name") and m.display_name:
                    other_mentions.append(m.display_name)
                else:
                    other_mentions.append(m.name)
            if other_mentions:
                tags.append(f"Mentioned users: {', '.join(other_mentions[:3])}")
    except Exception:
        pass
    try:
        if message and message.mention_everyone:
            tags.append("Mentioned everyone")
    except Exception:
        pass
    try:
        if message and message.reference and message.reference.resolved:
            replied_author = message.reference.resolved.author
            if replied_author == client.user:
                tags.append("Replied to you")
            elif hasattr(replied_author, "display_name") and replied_author.display_name:
                tags.append(f"Replied to {replied_author.display_name}")
            elif hasattr(replied_author, "global_name") and replied_author.global_name:
                tags.append(f"Replied to {replied_author.global_name}")
            else:
                tags.append(f"Replied to {replied_author.name}")
    except Exception:
        pass
    try:
        if message and message.content:
            raw = message.content.strip()
            if raw:
                cleaned = re.sub(r"[^\w\s]", "", raw).strip()
                if cleaned and " " not in cleaned and len(cleaned) <= 32:
                    channel_id = message.channel.id if message.channel else None
                    if channel_id and channel_id in recent_participants:
                        guild = message.guild
                        for user_id in list(recent_participants[channel_id])[-10:]:
                            member = guild.get_member(user_id) if guild else None
                            if not member:
                                continue
                            display = member.display_name or member.name
                            if display and display.lower() == cleaned.lower():
                                tags.append(f"Name reference: {display}")
                                break
    except Exception:
        pass
    try:
        if message and message.role_mentions:
            role_names = [role.name for role in message.role_mentions[:3]]
            if role_names:
                tags.append(f"Mentioned roles: {', '.join(role_names)}")
    except Exception:
        pass
    return tags

def looks_like_planning(text: str) -> bool:
    """Heuristic to detect planning / chain-of-thought leakage."""
    if not text:
        return False
    lower = text.lower()
    patterns = [
        r"\blet[' ]s (break down|analyze|think|walk through)\b",
        r"\b(first|second|third|next),\b",
        r"\bi need to (assess|analyze|figure out|consider|decide|respond|reply)\b",
        r"\bto (answer|respond|reply),\b",
        r"\bmy (plan|approach|reasoning)\b",
        r"\bhere's (my|the) plan\b",
        r"\bthought process\b",
        r"\bchain[- ]of[- ]thought\b",
        r"\bstep by step\b",
        r"\b(reasoning|analysis):",
        r"\bi(?:'|’)ll (respond|reply|answer)\b",
        r"\bi will (respond|reply|answer)\b",
    ]
    return any(re.search(pat, lower) for pat in patterns)

def apply_template_placeholders(text: str, bot_name: str, user_name: str) -> str:
    """Support both {var} and {{var}} placeholder styles."""
    if not text:
        return text
    replacements = {
        "{{char}}": bot_name or "",
        "{{user}}": user_name or "",
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

def set_cli_reasoning_stream_enabled(enabled: bool) -> None:
    """Persist CLI reasoning stream setting."""
    global cli_reasoning_stream_enabled
    cli_reasoning_stream_enabled = bool(enabled)
    save_json_data(CLI_REASONING_STREAM_FILE, {"enabled": cli_reasoning_stream_enabled}, convert_keys=False)

async def stream_reasoning_cli(stop_event: asyncio.Event) -> None:
    """Stream a simple spinner in CLI while reasoning runs."""
    spinner = ["|", "/", "-", "\\"]
    idx = 0
    try:
        while not stop_event.is_set():
            print(f"\r🧠 Reasoning {spinner[idx % len(spinner)]}", end="", flush=True)
            idx += 1
            await asyncio.sleep(0.4)
    finally:
        # Clear the line
        print("\r", end="", flush=True)

def is_error_like_response(text: str) -> bool:
    """Check if a response looks like an API/proxy error."""
    if not text:
        return False
    if text.startswith("âŒ"):
        return True
    lower = text.lower()
    error_indicators = [
        "proxy error", "upstream connect error", "connection termination",
        "service unavailable", "context size limit", "request validation failed",
        "tokens.*exceeds", "http 503", "http 400", "http 429", "rate limit", "timeout"
    ]
    return any(indicator in lower for indicator in error_indicators)

def clean_malformed_emojis(text: str, guild: discord.Guild = None) -> str:
    """Convert :emoji_name: format to proper Discord format or remove invalid ones"""
    
    if not text:
        return text
    
    # First, fix any double-wrapped emojis like <<:emoji:id>> or <<a:emoji:id>>
    text = re.sub(r'<(<a?:[a-zA-Z0-9_]+:[0-9]+>)>', r'\1', text)
    
    # Store valid Discord emojis to protect them during cleaning
    valid_emojis = []
    emoji_placeholders = []
    
    # Find and temporarily replace all valid Discord emojis with placeholders
    valid_emoji_pattern = r'<(a?):([a-zA-Z0-9_]+):([0-9]+)>'
    matches = list(re.finditer(valid_emoji_pattern, text))
    
    for i, match in enumerate(matches):
        placeholder = f"__EMOJI_PLACEHOLDER_{i}__"
        valid_emojis.append(match.group(0))  # Store the full emoji
        emoji_placeholders.append(placeholder)
        text = text.replace(match.group(0), placeholder, 1)
    
    # Pattern to match :emoji_name: format (simple Discord emoji syntax)
    simple_emoji_pattern = r':([a-zA-Z0-9_]+):'
    
    def replace_emoji(match):
        full_match = match.group(0)
        emoji_name = match.group(1).lower()
        
        # Check if this is already part of a placeholder (skip it)
        if "__EMOJI_PLACEHOLDER_" in full_match:
            return full_match
        
        # If we have a guild, try to find the actual emoji (both animated and static)
        if guild:
            for emoji in guild.emojis:
                if emoji.name.lower() == emoji_name:
                    return f"<{'a' if emoji.animated else ''}:{emoji.name}:{emoji.id}>"
        
        # Check if it might be a standard Unicode emoji name
        common_unicode_emojis = {
            'smile', 'grin', 'joy', 'heart', 'thumbsup', 'thumbsdown', 
            'fire', 'star', 'eyes', 'thinking', 'shrug', 'wave', 'clap'
        }
        
        if emoji_name in common_unicode_emojis:
            return f":{emoji_name}:"
        
        # Remove unknown emoji
        return ""
    
    # Replace :emoji_name: patterns (but not placeholders)
    cleaned_text = re.sub(simple_emoji_pattern, replace_emoji, text)
    
    # Only clean up ACTUALLY malformed patterns (be more specific)
    leftover_patterns = [
        r'<a?:[a-zA-Z0-9_]*$',           # Incomplete at end of string
        r'<a?:[a-zA-Z0-9_]*:[0-9]*$',    # Missing closing bracket at end
        r'<a?:[a-zA-Z0-9_]+:$',          # Missing ID and closing bracket
        r'<a?:$',                        # Just opening
    ]
    
    for pattern in leftover_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text)
    
    # Restore the valid emojis from placeholders
    for placeholder, original_emoji in zip(emoji_placeholders, valid_emojis):
        cleaned_text = cleaned_text.replace(placeholder, original_emoji)
    
    # Clean up multiple spaces but preserve line structure
    lines = cleaned_text.split('\n')
    cleaned_lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    cleaned_text = '\n'.join(cleaned_lines)
    
    return cleaned_text

def clean_em_dashes(text: str) -> str:
    """Replace em-dashes with appropriate punctuation based on context.
    
    - Mid-sentence em-dashes (text before and after) become ", "
    - End-sentence em-dashes (text before, nothing after) become "-"
    """
    if not text:
        return text
    
    # Pattern for em-dash with text before and after (mid-sentence)
    # Lookbehind: non-whitespace character before em-dash
    # Lookahead: non-whitespace character after em-dash
    mid_sentence_pattern = r'(?<=\S)—(?=\S)'
    
    # Replace mid-sentence em-dashes with ", "
    text = re.sub(mid_sentence_pattern, ", ", text)
    
    # Pattern for em-dash at end of sentence (text before, but nothing after except whitespace/punctuation)
    # Lookbehind: non-whitespace character before em-dash
    # Lookahead: whitespace, punctuation, or end of string
    end_sentence_pattern = r'(?<=\S)—(?=\s|$|[.!?])'
    
    # Replace end-sentence em-dashes with "-"
    text = re.sub(end_sentence_pattern, "-", text)
    
    return text

def clean_bot_name_prefix(text: str, guild_id: int = None, user_id: int = None, is_dm: bool = False) -> str:
    """Remove bot persona name prefix from the response text before sending to Discord"""
    if not text:
        return text
    
    # Get the bot's persona name
    bot_name = get_bot_persona_name(guild_id, user_id, is_dm)
    
    # Remove the prefix if it exists
    if text.startswith(f"{bot_name}: "):
        return text[len(f"{bot_name}: "):]
    elif text.startswith(f"{bot_name}:"):
        return text[len(f"{bot_name}:"):]
    
    return text

def normalize_for_duplicate_check(text: str, guild_id: int = None, user_id: int = None, is_dm: bool = False) -> str:
    """Normalize assistant text for duplicate detection."""
    if not text:
        return ""

    # Extract text if content is complex
    if isinstance(text, list):
        parts = []
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        text = " ".join(parts)

    # Remove bot name prefix if present
    bot_name = get_bot_persona_name(guild_id, user_id, is_dm)
    if text.startswith(f"{bot_name}: "):
        text = text[len(f"{bot_name}: "):]
    elif text.startswith(f"{bot_name}:"):
        text = text[len(f"{bot_name}:"):]

    # Remove reaction tags and mentions
    text = re.sub(r'\[REACT:\s*[^\]]+\]', '', text)
    text = re.sub(r'<@!?\d+>', '', text)
    text = re.sub(r'<@&\d+>', '', text)
    text = re.sub(r'<#\d+>', '', text)

    # Remove simple emoji tokens like :smile:
    text = re.sub(r':[a-zA-Z0-9_]+:', '', text)

    # Normalize whitespace and punctuation
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def get_recent_assistant_messages(channel_id: int, limit: int = 3) -> List[str]:
    """Return the most recent assistant messages for duplicate detection."""
    history = conversations.get(channel_id, [])
    recent = []
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            recent.append(msg.get("content", ""))
            if len(recent) >= limit:
                break
    return recent

def make_embedding_cache_key(text: str, model: str) -> str:
    """Create a stable cache key for embeddings."""
    if not text:
        return ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model}:{digest}"

def store_embedding_in_cache(cache_key: str, embedding: List[float]) -> None:
    """Store embedding in cache with a simple size cap."""
    if not cache_key or not embedding:
        return
    if len(embedding_cache) >= EMBEDDING_CACHE_MAX:
        embedding_cache.clear()
    embedding_cache[cache_key] = embedding

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

def resolve_dm_context_guild_id(user_id: int, guild_id: int = None) -> Optional[int]:
    """Resolve the guild context to use for DM settings."""
    if user_id:
        selected_guild_id = dm_server_selection.get(user_id)
        if selected_guild_id:
            return selected_guild_id
    if guild_id:
        return guild_id
    if user_id:
        shared_guild = get_shared_guild(user_id)
        if shared_guild:
            return shared_guild.id
    return None

def extract_embedding_settings(settings: Optional[dict], enabled_default: bool, model_default: str) -> Tuple[bool, str]:
    """Extract enabled/model from settings dict with safe defaults."""
    if not isinstance(settings, dict):
        return enabled_default, model_default
    enabled = settings.get("enabled", enabled_default)
    model = settings.get("model", model_default)
    if not isinstance(enabled, bool):
        enabled = enabled_default
    if not isinstance(model, str) or not model.strip():
        model = model_default
    return enabled, model

def get_effective_embedding_settings(guild_id: int = None, user_id: int = None, is_dm: bool = False) -> Tuple[bool, str]:
    """Get enabled/model settings for embeddings in this context."""
    enabled = EMBEDDINGS_ENABLED
    model = EMBEDDING_MODEL

    if is_dm and user_id:
        dm_settings = dm_embedding_settings.get(user_id)
        if dm_settings is not None:
            return extract_embedding_settings(dm_settings, enabled, model)

        context_guild_id = resolve_dm_context_guild_id(user_id, guild_id)
        if context_guild_id:
            guild_settings = embedding_settings.get(context_guild_id)
            return extract_embedding_settings(guild_settings, enabled, model)

        return enabled, model

    if guild_id:
        guild_settings = embedding_settings.get(guild_id)
        return extract_embedding_settings(guild_settings, enabled, model)

    return enabled, model

def set_embedding_settings_for_guild(guild_id: int, enabled: Optional[bool] = None, model: Optional[str] = None) -> bool:
    """Update embeddings settings for a guild."""
    if not guild_id:
        return False
    settings = embedding_settings.get(guild_id, {})
    if enabled is not None:
        settings["enabled"] = enabled
    if model is not None and model.strip():
        settings["model"] = model.strip()
    embedding_settings[guild_id] = settings
    save_json_data(EMBEDDINGS_SETTINGS_FILE, embedding_settings)
    return True

def set_embedding_settings_for_dm(user_id: int, enabled: Optional[bool] = None, model: Optional[str] = None) -> bool:
    """Update embeddings settings for a DM user."""
    if not user_id:
        return False
    settings = dm_embedding_settings.get(user_id, {})
    if enabled is not None:
        settings["enabled"] = enabled
    if model is not None and model.strip():
        settings["model"] = model.strip()
    dm_embedding_settings[user_id] = settings
    save_json_data(DM_EMBEDDINGS_SETTINGS_FILE, dm_embedding_settings)
    return True

def get_openrouter_embedding_config(guild_id: int = None, user_id: int = None, is_dm: bool = False) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve OpenRouter embedding configuration for this context."""
    enabled, model = get_effective_embedding_settings(guild_id, user_id, is_dm)
    if not enabled:
        return None, None, None

    if OPENROUTER_API_KEY:
        return OPENROUTER_BASE_URL, OPENROUTER_API_KEY, model

    if not CUSTOM_API_KEY:
        return None, None, None

    context_guild_id = resolve_dm_context_guild_id(user_id, guild_id) if is_dm else guild_id

    if context_guild_id:
        provider_name, _ = ai_manager.get_guild_settings(context_guild_id)
        if provider_name == "custom":
            custom_url = ai_manager.get_guild_custom_url(context_guild_id)
            if custom_url and "openrouter.ai" in custom_url.lower():
                return custom_url, CUSTOM_API_KEY, model

    return None, None, None

async def fetch_openrouter_embeddings(texts: List[str], base_url: str, api_key: str, model: str) -> List[Optional[List[float]]]:
    """Fetch embeddings from OpenRouter for the provided texts."""
    if not texts:
        return []

    payload = {
        "model": model,
        "input": texts
    }

    try:
        data, _ = await post_openrouter_json(base_url, api_key, "/embeddings", payload, timeout=60)
    except Exception:
        return [None for _ in texts]

    if not isinstance(data, dict):
        return [None for _ in texts]

    raw_items = data.get("data")
    if isinstance(raw_items, dict):
        raw_items = raw_items.get("data")

    if not isinstance(raw_items, list):
        return [None for _ in texts]

    results: List[Optional[List[float]]] = [None for _ in texts]
    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            continue
        item_index = item.get("index")
        if isinstance(item_index, int) and 0 <= item_index < len(texts):
            results[item_index] = embedding
        elif idx < len(texts):
            results[idx] = embedding

    return results

async def get_embeddings_for_texts(texts: List[str], guild_id: int = None, user_id: int = None, is_dm: bool = False) -> List[Optional[List[float]]]:
    """Return embeddings for texts, using cache and OpenRouter when available."""
    base_url, api_key, model = get_openrouter_embedding_config(guild_id, user_id, is_dm)
    if not base_url or not api_key or not model:
        return [None for _ in texts]

    results: List[Optional[List[float]]] = [None for _ in texts]
    pending_texts: List[str] = []
    pending_indices: List[int] = []
    pending_keys: List[str] = []

    for i, text in enumerate(texts):
        clean_text = text.strip() if isinstance(text, str) else ""
        if not clean_text:
            continue
        cache_key = make_embedding_cache_key(clean_text, model)
        cached = embedding_cache.get(cache_key)
        if cached:
            results[i] = cached
        else:
            pending_texts.append(clean_text)
            pending_indices.append(i)
            pending_keys.append(cache_key)

    if pending_texts:
        fetched = await fetch_openrouter_embeddings(pending_texts, base_url, api_key, model)
        for fetched_idx, embedding in enumerate(fetched):
            target_index = pending_indices[fetched_idx]
            results[target_index] = embedding
            if embedding:
                store_embedding_in_cache(pending_keys[fetched_idx], embedding)

    return results

async def get_relevant_memory_text(query: str, guild_id: int = None, user_id: int = None, is_dm: bool = False) -> str:
    """Retrieve the most relevant memory using embeddings when available, else keyword search."""
    if not query or not isinstance(query, str) or not query.strip():
        return "No relevant memory found."

    if is_dm and user_id:
        memories = memory_manager.get_all_dm_memories(user_id)
    elif guild_id:
        memories = memory_manager.get_all_memories(guild_id)
    else:
        memories = []

    if not memories:
        return "No relevant memory found."

    # Limit candidates to the most recent memories to control cost
    memories_sorted = sorted(memories, key=lambda x: x.get("timestamp", 0), reverse=True)
    if EMBEDDING_MEMORY_MAX_CANDIDATES > 0:
        memories_sorted = memories_sorted[:EMBEDDING_MEMORY_MAX_CANDIDATES]

    # Attempt embedding-based search first
    embeddings = await get_embeddings_for_texts(
        [query] + [m.get("memory", "") for m in memories_sorted],
        guild_id,
        user_id,
        is_dm
    )

    query_embedding = embeddings[0] if embeddings else None
    if query_embedding:
        updated_embeddings = False
        best_score = -1.0
        best_memory_text = None
        best_timestamp = -1

        for i, memory in enumerate(memories_sorted):
            embedding = embeddings[i + 1] if len(embeddings) > i + 1 else None
            if embedding and not memory.get("embedding"):
                memory["embedding"] = embedding
                updated_embeddings = True

            candidate_embedding = memory.get("embedding") or embedding
            if not candidate_embedding:
                continue

            score = cosine_similarity(query_embedding, candidate_embedding)
            if score > best_score or (score == best_score and memory.get("timestamp", 0) > best_timestamp):
                best_score = score
                best_memory_text = memory.get("memory")
                best_timestamp = memory.get("timestamp", 0)

        if updated_embeddings:
            if is_dm and user_id:
                memory_manager.save_dm_data()
            elif guild_id:
                memory_manager.save_data()

        if best_memory_text and best_score >= EMBEDDING_MEMORY_THRESHOLD:
            return best_memory_text

    # Fallback to keyword search
    if is_dm and user_id:
        relevant_memories = memory_manager.search_dm_memories(user_id, query)
    else:
        relevant_memories = memory_manager.search_memories(guild_id, query) if guild_id else []

    if relevant_memories:
        return "\n".join([mem.get("memory", "") for mem in relevant_memories])

    return "No relevant memory found."

async def is_semantic_duplicate_response(candidate: str, channel_id: int, guild_id: int = None, user_id: int = None, is_dm: bool = False) -> bool:
    """Check if candidate response is semantically similar to recent assistant replies."""
    if not candidate or not isinstance(candidate, str):
        return False

    normalized_candidate = normalize_for_duplicate_check(candidate, guild_id, user_id, is_dm)
    if not normalized_candidate:
        return False

    recent_messages = get_recent_assistant_messages(channel_id, limit=3)
    for recent in recent_messages:
        if normalize_for_duplicate_check(recent, guild_id, user_id, is_dm) == normalized_candidate:
            return True

    if len(normalized_candidate) < EMBEDDING_MIN_TEXT_CHARS:
        return False

    def strip_react_tags(text: str) -> str:
        return re.sub(r'\[REACT:\s*[^\]]+\]', '', text or "").strip()

    texts = [strip_react_tags(clean_bot_name_prefix(candidate, guild_id, user_id, is_dm))]
    for recent in recent_messages:
        texts.append(strip_react_tags(clean_bot_name_prefix(recent, guild_id, user_id, is_dm)))

    embeddings = await get_embeddings_for_texts(texts, guild_id, user_id, is_dm)
    if not embeddings or not embeddings[0]:
        return False

    candidate_embedding = embeddings[0]
    for embedding in embeddings[1:]:
        if not embedding:
            continue
        if cosine_similarity(candidate_embedding, embedding) >= EMBEDDING_DUP_THRESHOLD:
            return True

    return False

def save_custom_prompts():
    """Save custom prompts to file"""
    save_data = {
        "custom_prompts": {str(guild_id): prompts for guild_id, prompts in custom_prompts.items()}
    }
    save_json_data(CUSTOM_PROMPTS_FILE, save_data, convert_keys=False)

def save_custom_format_instructions():
    """Save custom format instructions to file"""
    save_json_data(CUSTOM_FORMAT_INSTRUCTIONS_FILE, custom_format_instructions, convert_keys=False)

def load_custom_prompts():
    """Load custom prompts from file"""
    data = load_json_data(CUSTOM_PROMPTS_FILE, convert_keys=False)
    custom_prompts_data = {}
    if "custom_prompts" in data:
        for guild_id_str, prompts in data["custom_prompts"].items():
            custom_prompts_data[int(guild_id_str)] = prompts
    return custom_prompts_data

# Remove all custom prompt related variables and functions

# Global state variables
conversations: Dict[int, List[Dict]] = {}
conversation_summaries: Dict[int, str] = load_json_data(SUMMARY_FILE)
# Recent reaction micro-memory per channel
recent_reactions: Dict[int, List[Dict]] = {}
embedding_cache: Dict[str, List[float]] = {}
summary_last_updated: Dict[int, float] = {}
summary_locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
bot_start_time = time.time()

def clean_conversation_history():
    """Clean any complex content (with base64 images) from conversation history, keeping only text"""
    for channel_id in conversations:
        for message in conversations[channel_id]:
            content = message.get("content", "")
            if isinstance(content, list):
                # Extract only text parts from complex content
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                message["content"] = " ".join(text_parts).strip()

def save_conversation_summaries():
    """Persist conversation summaries to file"""
    save_json_data(SUMMARY_FILE, conversation_summaries)

def format_history_for_summary(messages: List[Dict], guild_id: int = None, user_id: int = None, is_dm: bool = False) -> str:
    """Format messages for summarization."""
    if not messages:
        return ""
    
    bot_name = get_bot_persona_name(guild_id, user_id, is_dm)
    formatted = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = " ".join(text_parts).strip()
        
        if not isinstance(content, str) or not content.strip():
            continue
        
        if role == "assistant":
            formatted.append(f"{bot_name}: {content}")
        else:
            if is_dm and user_id:
                user = client.get_user(user_id)
                user_display = user.display_name if user and hasattr(user, 'display_name') else (user.name if user else "User")
                formatted.append(f"{user_display}: {content}")
            else:
                formatted.append(content)
    
    return "\n".join(formatted)

async def maybe_summarize_and_trim(channel_id: int, max_history: int, guild_id: int = None, user_id: int = None, is_dm: bool = False):
    """Summarize overflow history and trim to max_history."""
    if channel_id not in conversations:
        return
    
    history = conversations[channel_id]
    if len(history) <= max_history:
        return
    
    # Prevent rapid repeat summarization
    now = time.time()
    last_updated = summary_last_updated.get(channel_id, 0)
    if now - last_updated < 60:
        conversations[channel_id] = history[-max_history:]
        return
    
    excess = len(history) - max_history
    if excess < 4:
        conversations[channel_id] = history[-max_history:]
        return
    
    async with summary_locks[channel_id]:
        # Re-check after lock
        history = conversations[channel_id]
        if len(history) <= max_history:
            return
        
        excess = len(history) - max_history
        if excess < 4:
            conversations[channel_id] = history[-max_history:]
            return
        
        to_summarize = history[:excess]
        summary_input = format_history_for_summary(to_summarize, guild_id, user_id, is_dm)
        
        if not summary_input.strip():
            conversations[channel_id] = history[-max_history:]
            return
        
        existing_summary = conversation_summaries.get(channel_id, "")
        summary_prompt = """You are summarizing earlier parts of a Discord conversation.
Update the summary so it includes the new information without losing prior important context.
Keep it concise, factual, and useful for future responses.
Use 3-6 short bullet points. Avoid quoting verbatim."""
        
        user_content = f"Existing summary (if any):\n{existing_summary}\n\nNew messages to summarize:\n{summary_input}"
        
        try:
            summary_response = await ai_manager.generate_response(
                messages=[{"role": "user", "content": user_content}],
                system_prompt=summary_prompt,
                temperature=0.4,
                guild_id=guild_id,
                is_dm=is_dm,
                user_id=user_id,
                max_tokens=700
            )
            
            if summary_response and not summary_response.startswith("âŒ"):
                conversation_summaries[channel_id] = summary_response.strip()
                save_conversation_summaries()
                summary_last_updated[channel_id] = now
        except Exception as e:
            print(f"Summary generation failed: {e}")
        
        conversations[channel_id] = history[-max_history:]

    # Clean conversation history on startup
clean_conversation_history()

def add_reaction_memory(channel_id: int, reacted_user_id: int, reaction_text: str):
    """Store recent reaction actions as micro-memory (per channel)."""
    now = time.time()
    entry = {"user_id": reacted_user_id, "reaction": reaction_text, "timestamp": now}
    entries = recent_reactions.get(channel_id, [])
    entries.append(entry)
    # Prune old entries (15 minutes)
    cutoff = now - 900
    entries = [e for e in entries if e["timestamp"] >= cutoff]
    # Keep last 5
    if len(entries) > 5:
        entries = entries[-5:]
    recent_reactions[channel_id] = entries

def get_recent_reaction_memory(channel_id: int) -> List[Dict]:
    """Get recent reaction micro-memories for a channel."""
    return recent_reactions.get(channel_id, [])

channel_format_settings: Dict[int, str] = load_json_data(FORMAT_SETTINGS_FILE)
dm_format_settings: Dict[int, str] = load_json_data(DM_FORMAT_SETTINGS_FILE)
server_format_defaults: Dict[int, str] = load_json_data(SERVER_FORMAT_DEFAULTS_FILE)
guild_nsfw_settings: Dict[int, bool] = load_json_data(NSFW_SETTINGS_FILE)
dm_nsfw_settings: Dict[int, bool] = load_json_data(DM_NSFW_SETTINGS_FILE)
guild_reasoning_settings: Dict[int, bool] = load_json_data(REASONING_SETTINGS_FILE)
dm_reasoning_settings: Dict[int, bool] = load_json_data(DM_REASONING_SETTINGS_FILE)
guild_reasoning_effort: Dict[int, str] = load_json_data(REASONING_EFFORT_FILE)
dm_reasoning_effort: Dict[int, str] = load_json_data(DM_REASONING_EFFORT_FILE)
embedding_settings: Dict[int, Dict[str, object]] = load_json_data(EMBEDDINGS_SETTINGS_FILE)
dm_embedding_settings: Dict[int, Dict[str, object]] = load_json_data(DM_EMBEDDINGS_SETTINGS_FILE)
cli_reasoning_stream_settings: Dict[str, object] = load_json_data(CLI_REASONING_STREAM_FILE, convert_keys=False)
cli_reasoning_stream_enabled: bool = bool(cli_reasoning_stream_settings.get("enabled", False))
custom_format_instructions: Dict[str, str] = load_json_data(CUSTOM_FORMAT_INSTRUCTIONS_FILE, convert_keys=False)
prefill_settings: Dict[int, str] = load_json_data(PREFILL_SETTINGS_FILE)
multipart_responses: Dict[int, Dict[int, Tuple[List[int], str]]] = {}
multipart_response_counter: Dict[int, int] = {}
guild_personalities: Dict[int, str] = {}
custom_personalities: Dict[int, Dict[str, Dict[str, str]]] = {}
guild_history_lengths: Dict[int, int] = load_json_data(HISTORY_LENGTHS_FILE)
recent_participants: Dict[int, Set[int]] = {}
custom_activity: str = load_json_data(ACTIVITY_FILE, convert_keys=False).get("custom_activity", "")
guild_temperatures: Dict[int, float] = load_json_data(TEMPERATURE_FILE)
welcome_dm_sent: Dict[int, bool] = load_json_data(WELCOME_SENT_FILE)
dm_server_selection: Dict[int, int] = load_json_data(DM_SERVER_SELECTION_FILE)
guild_dm_enabled: Dict[int, bool] = load_json_data(DM_ENABLED_FILE)
bot_persona_name: str = "Assistant"
recently_deleted_dm_messages: Dict[int, Set[int]] = {}
def record_response_success() -> None:
    """Reset failure counter after a successful response."""
    global consecutive_response_failures
    consecutive_response_failures = 0

def record_response_failure() -> None:
    """Increment failure counter and exit if threshold is exceeded."""
    global consecutive_response_failures
    consecutive_response_failures += 1
    if consecutive_response_failures >= MAX_CONSECUTIVE_FAILURES:
        print(f"🚨 Too many consecutive failures ({consecutive_response_failures}). Exiting (Ctrl+C).")
        raise KeyboardInterrupt

def coerce_int(value: Any) -> Optional[int]:
    """Best-effort int conversion for token usage fields."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

async def fetch_openrouter_generation_usage(generation_id: str, retries: int = 3, delay_s: float = 0.5) -> None:
    """Fetch OpenRouter generation stats for accurate token usage."""
    if not generation_id or not CUSTOM_API_KEY:
        return
    generation_id = str(generation_id)
    is_gen_id = generation_id.startswith("gen-")
    if not is_gen_id:
        # Avoid long waits for non-generation IDs, but still try once.
        retries = 1
        delay_s = 0
    url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
    headers = {
        "Authorization": f"Bearer {CUSTOM_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        async with aiohttp.ClientSession() as session:
            for attempt in range(retries):
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        payload = data.get("data") or data.get("generation") or {}
                        native_in = coerce_int(payload.get("native_tokens_prompt"))
                        native_out = coerce_int(payload.get("native_tokens_completion"))
                        if native_in is not None or native_out is not None:
                            update_last_token_usage(input_tokens=native_in, output_tokens=native_out)
                            return
                        # Fallback to non-native tokens if present
                        tok_in = coerce_int(payload.get("tokens_prompt"))
                        tok_out = coerce_int(payload.get("tokens_completion"))
                        if tok_in is not None or tok_out is not None:
                            update_last_token_usage(input_tokens=tok_in, output_tokens=tok_out)
                            return
                    elif resp.status in (404, 202):
                        # Generation stats may not be ready yet
                        if not is_gen_id:
                            return
                        await asyncio.sleep(delay_s)
                        continue
                    else:
                        return
    except Exception:
        return

# Temporary old system variables - TO BE COMPLETELY REMOVED
# custom_prompts: Dict[int, Dict[str, Dict[str, str]]] = {}
# channel_prompt_settings: Dict[int, str] = {}
# dm_prompt_settings: Dict[int, str] = {}

# Migration function removed - no longer needed

# Run migration on startup
# migrate_old_nsfw_styles()  # Removed - no longer needed

# Initialize managers
lore_book = LoreBook()
autonomous_manager = AutonomousManager()
memory_manager = MemoryManager()
dm_manager = DMManager()

# Load personality data
loaded_guild_personalities, loaded_custom_personalities = load_personalities()
guild_personalities = loaded_guild_personalities
custom_personalities = loaded_custom_personalities

# Default personality configuration
DEFAULT_PERSONALITIES = {
    "default": {
        "name": "Assistant",
        "prompt": "A helpful AI assistant. Friendly, curious, and engaging in conversations."
    }
}

def get_bot_persona_name(guild_id: int = None, user_id: int = None, is_dm: bool = False) -> str:
    """Get the bot's current persona name based on context"""
    global bot_persona_name
    
    # For DMs, get persona from selected server or shared guild
    if is_dm and user_id:
        # Check if user has a specific DM personality set
        if user_id in dm_manager.dm_personalities:
            preferred_guild_id, preferred_personality = dm_manager.dm_personalities[user_id]
            guild_id = preferred_guild_id
            personality_name = preferred_personality
        else:
            # Use selected server or shared guild
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                guild_id = selected_guild_id
                personality_name = guild_personalities.get(guild_id, "default")
            else:
                shared_guild = get_shared_guild(user_id)
                if shared_guild:
                    guild_id = shared_guild.id
                    personality_name = guild_personalities.get(guild_id, "default")
                else:
                    personality_name = "default"
    else:
        # Server context
        personality_name = guild_personalities.get(guild_id, "default") if guild_id else "default"
    
    # Extract the name from personality data
    if guild_id and guild_id in custom_personalities and personality_name in custom_personalities[guild_id]:
        return custom_personalities[guild_id][personality_name]["name"]
    else:
        return DEFAULT_PERSONALITIES["default"]["name"]

def update_bot_persona_name(guild_id: int = None, user_id: int = None, is_dm: bool = False):
    """Update the global bot persona name based on current context"""
    global bot_persona_name
    bot_persona_name = get_bot_persona_name(guild_id, user_id, is_dm)

def is_reasoning_enabled(guild_id: int = None, user_id: int = None, is_dm: bool = False) -> bool:
    """Check if reasoning mode is enabled for this context."""
    if is_dm and user_id:
        return dm_reasoning_settings.get(user_id, False)
    if guild_id:
        return guild_reasoning_settings.get(guild_id, False)
    return False

def get_reasoning_effort(guild_id: int = None, user_id: int = None, is_dm: bool = False) -> str:
    """Get reasoning effort level for this context."""
    if is_dm and user_id:
        return dm_reasoning_effort.get(user_id, "high")
    if guild_id:
        return guild_reasoning_effort.get(guild_id, "high")
    return "high"

def get_guild_emojis(guild: discord.Guild) -> str:
    """Get formatted list of guild emojis for system prompt with simple :name: format"""
    if not guild:
        return ""
    
    # Get both animated and non-animated emojis
    available_emojis = list(guild.emojis)  # This includes both animated and static
    
    if available_emojis:
        # Select up to 50 emojis (mix of animated and static)
        max_emojis = min(50, len(available_emojis))
        
        # If we have more emojis than the limit, prioritize by usage and variety
        if len(available_emojis) > max_emojis:
            # Sort by available status first (prioritize available emojis)
            available_emojis = sorted(available_emojis, key=lambda e: e.available, reverse=True)
            
            # Try to get a good mix of animated and static emojis
            static_emojis = [e for e in available_emojis if not e.animated]
            animated_emojis = [e for e in available_emojis if e.animated]
            
            selected_emojis = []
            
            # Take up to 35 static emojis and 15 animated emojis for variety
            static_count = min(35, len(static_emojis), max_emojis)
            animated_count = min(15, len(animated_emojis), max_emojis - static_count)
            
            if static_emojis:
                selected_emojis.extend(random.sample(static_emojis, static_count))
            if animated_emojis:
                selected_emojis.extend(random.sample(animated_emojis, animated_count))
            
            # If we still need more emojis, fill from remaining
            remaining_needed = max_emojis - len(selected_emojis)
            if remaining_needed > 0:
                remaining_emojis = [e for e in available_emojis if e not in selected_emojis]
                if remaining_emojis:
                    selected_emojis.extend(random.sample(remaining_emojis, min(remaining_needed, len(remaining_emojis))))
        else:
            selected_emojis = available_emojis
        
        # Sort by name for consistent display
        selected_emojis.sort(key=lambda e: e.name.lower())
        
        # Format them in simple :name: format
        emoji_list = ' '.join([f":{emoji.name}:" for emoji in selected_emojis])
        
        return f"\n\n<emojis>Available custom emojis for this server:\n{emoji_list}\nTEMPLATE: When using custom emojis follow the exact format of :emoji: in your responses, otherwise they won't work! Limit their usage.\nREACTIONS: You can also react to the users' messages with emojis! To add a reaction, include [REACT: emoji] anywhere in your response. Examples: [REACT: 😄] or [REACT: :custom_emoji:]. Occasionally, react to show emotion, agreement, humor, or acknowledgment.</emojis>"
    return ""

def get_guild_emoji_list(guild: discord.Guild) -> str:
    """Get simple comma-separated list of guild emojis for system prompt
    
    Selects up to 50 emojis from the server, prioritizing:
    - Available emojis over unavailable ones
    - A mix of static and animated emojis (roughly 35 static, 15 animated)
    - Alphabetical order for consistency
    """
    if not guild:
        return "No custom emojis available."
    
    # Get both animated and non-animated emojis
    available_emojis = list(guild.emojis)
    
    if not available_emojis:
        return "No custom emojis available for this server."
    
    # Select up to 50 emojis (mix of animated and static)
    max_emojis = min(50, len(available_emojis))
    
    # If we have more emojis than the limit, prioritize by usage and variety
    if len(available_emojis) > max_emojis:
        # Sort by available status first (prioritize available emojis)
        available_emojis = sorted(available_emojis, key=lambda e: e.available, reverse=True)
        
        # Try to get a good mix of animated and static emojis
        static_emojis = [e for e in available_emojis if not e.animated]
        animated_emojis = [e for e in available_emojis if e.animated]
        
        selected_emojis = []
        
        # Take up to 35 static emojis and 15 animated emojis for variety
        static_count = min(35, len(static_emojis), max_emojis)
        animated_count = min(15, len(animated_emojis), max_emojis - static_count)
        
        if static_emojis:
            selected_emojis.extend(random.sample(static_emojis, static_count))
        if animated_emojis:
            selected_emojis.extend(random.sample(animated_emojis, animated_count))
        
        # If we still need more emojis, fill from remaining
        remaining_needed = max_emojis - len(selected_emojis)
        if remaining_needed > 0:
            remaining_emojis = [e for e in available_emojis if e not in selected_emojis]
            if remaining_emojis:
                selected_emojis.extend(random.sample(remaining_emojis, min(remaining_needed, len(remaining_emojis))))
    else:
        selected_emojis = available_emojis
    
    # Sort by name for consistent display
    selected_emojis.sort(key=lambda e: e.name.lower())
    
    # Format them in simple :name: format
    emoji_count = len(selected_emojis)
    total_count = len(available_emojis)
    emoji_list = ' '.join([f":{emoji.name}:" for emoji in selected_emojis])
    
    # Add a note if we're showing a subset
    if emoji_count < total_count:
        emoji_list += f" (showing {emoji_count} of {total_count} available emojis)"
    
    return emoji_list

async def process_and_add_reactions(bot_response: str, user_message: discord.Message) -> str:
    """Process bot response for reaction instructions and add reactions to user message"""
    if not bot_response:
        return bot_response

    # Find all reaction instructions in the response
    reaction_pattern = r'\[REACT:\s*([^\]]+)\]'
    reactions = re.findall(reaction_pattern, bot_response)
    applied_reactions: List[str] = []
    
    # Remove reaction instructions from the response
    if reactions:
        cleaned_response = re.sub(reaction_pattern, ' ', bot_response)
        cleaned_response = re.sub(r'  +', ' ', cleaned_response).strip()
    else:
        cleaned_response = bot_response
    
    # Add reactions if we have a message to react to
    if user_message is not None and reactions:
        for reaction in reactions:
            reaction = reaction.strip()
            
            # Convert custom emoji format if needed
            converted_reaction = convert_emoji_for_reaction(reaction, user_message.guild)
            
            # Skip if the emoji was filtered out (from another server)
            if converted_reaction is None:
                continue
            
            try:
                await user_message.add_reaction(converted_reaction)
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)
                # Store micro-memory for this reaction
                try:
                    add_reaction_memory(user_message.channel.id, user_message.author.id, reaction)
                except Exception:
                    pass
                applied_reactions.append(reaction)
            except discord.HTTPException as e:
                # If it still fails, just skip it
                # print(f"Failed to add reaction '{reaction}': {e}")
                continue
            except Exception as e:
                print(f"Unexpected error adding reaction '{reaction}': {e}")
                continue
    
    # Persist a lightweight reaction note in history for continuity
    if applied_reactions:
        try:
            target_name = user_message.author.display_name if hasattr(user_message.author, "display_name") else user_message.author.name
            reaction_list = ", ".join(applied_reactions)
            note = f"[Reacted to {target_name} with {reaction_list}]"
            await add_to_history(
                user_message.channel.id,
                "assistant",
                note,
                guild_id=user_message.guild.id if user_message.guild else None
            )
        except Exception:
            pass

    return cleaned_response

def convert_emoji_for_reaction(emoji_text: str, guild: discord.Guild = None) -> str:
    """Convert emoji text to proper format for reactions with improved matching"""
    
    # If it's already in proper Discord format, validate it exists in THIS guild
    if emoji_text.startswith('<:') or emoji_text.startswith('<a:'):
        if guild:
            # Extract emoji ID from the format <:name:id> or <a:name:id>
            emoji_id_match = re.search(r':(\d+)>', emoji_text)
            if emoji_id_match:
                emoji_id = int(emoji_id_match.group(1))
                # Check if this emoji exists in THIS guild
                for emoji in guild.emojis:
                    if emoji.id == emoji_id:
                        return emoji_text  # Valid emoji from this guild
                # Emoji doesn't exist in this guild - remove it
                return None
        # No guild to validate against - remove it
        return None
    
    # If it's in :name: format, try to convert to proper format
    if emoji_text.startswith(':') and emoji_text.endswith(':') and guild:
        emoji_name = emoji_text.strip(':')
        
        # Try to find this emoji in the current guild
        for emoji in guild.emojis:
            if emoji.name.lower() == emoji_name.lower():
                return f"<{'a' if emoji.animated else ''}:{emoji.name}:{emoji.id}>"
        
        # If no custom emoji found, check if it's a common Unicode emoji
        unicode_emoji_map = {
            'smile': '😊', 'heart': '❤️', 'thumbsup': '👍', 'thumbsdown': '👎',
            'fire': '🔥', 'star': '⭐', 'eyes': '👀', 'thinking': '🤔',
            'shrug': '🤷', 'wave': '👋', 'clap': '👏', 'kiss': '😘',
            'hug': '🤗', 'laugh': '😂', 'cry': '😢', 'angry': '😠',
            'mad': '😡', 'love': '💕'
        }
        
        emoji_name_lower = emoji_name.lower()
        if emoji_name_lower in unicode_emoji_map:
            return unicode_emoji_map[emoji_name_lower]
        
        # Unknown emoji - remove it
        return None
    
    # For Unicode emojis, return as-is
    return emoji_text

def estimate_message_size(messages: List[Dict], system_prompt: str) -> int:
    """Estimate the total size of the request in characters/tokens"""
    total_size = len(system_prompt)
    
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_size += len(content)
        elif isinstance(content, list):
            # Handle complex content with images
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total_size += len(part.get("text", ""))
                    elif part.get("type") == "image":
                        # Estimate image contributes roughly equivalent to 1000 characters
                        total_size += 1000
                    elif part.get("type") == "image_url":
                        # OpenAI format images
                        total_size += 1000
    
    return total_size

def is_supported_file_type(filename: str) -> bool:
    """Check if file type is supported for processing"""
    supported_image_types = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    supported_audio_types = ['.mp3', '.wav', '.ogg', '.m4a', '.webm']
    supported_text_types = ['.txt', '.md', '.json', '.csv', '.log']
    
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in 
               supported_image_types + supported_audio_types + supported_text_types)

def get_attachment_size_limit(attachment: discord.Attachment, provider: str = "claude") -> int:
    """Get size limit based on attachment type and provider"""
    filename_lower = attachment.filename.lower()
    
    # Image files
    if any(filename_lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
        return 20 * 1024 * 1024 if provider == "openai" else 8 * 1024 * 1024  # 20MB for OpenAI, 8MB for others
    
    # Audio files for voice processing
    elif any(filename_lower.endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
        return 25 * 1024 * 1024  # 25MB limit for audio files
    
    # Text files (can be included in context)
    elif any(filename_lower.endswith(ext) for ext in ['.txt', '.md', '.json', '.csv', '.log']):
        return 1 * 1024 * 1024  # 1MB limit for text files
    
    # All other files - very restrictive
    else:
        return 100 * 1024  # 100KB limit for other file types

def should_process_attachment(attachment: discord.Attachment, provider: str = "claude") -> Tuple[bool, str]:
    """Determine if attachment should be processed and return reason if not"""
    filename_lower = attachment.filename.lower()
    
    # Check for unsupported/problematic file types
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    large_doc_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']
    archive_extensions = ['.zip', '.rar', '.7z', '.tar', '.gz']
    executable_extensions = ['.exe', '.msi', '.dmg', '.app', '.deb', '.rpm']
    
    # Block video files (too large and not useful for text AI)
    if any(filename_lower.endswith(ext) for ext in video_extensions):
        return False, f"Video files ({attachment.filename}) are not supported to prevent request size issues"
    
    # Block large document formats that require special processing
    if any(filename_lower.endswith(ext) for ext in large_doc_extensions):
        return False, f"Document files ({attachment.filename}) are not supported - please convert to text format"
    
    # Block archives and executables
    if any(filename_lower.endswith(ext) for ext in archive_extensions + executable_extensions):
        return False, f"Archive/executable files ({attachment.filename}) are not supported"
    
    # Check if file type is supported
    if not is_supported_file_type(attachment.filename):
        return False, f"File type not supported ({attachment.filename})"
    
    # Check file size
    size_limit = get_attachment_size_limit(attachment, provider)
    if attachment.size > size_limit:
        size_limit_mb = size_limit / (1024 * 1024)
        actual_size_mb = attachment.size / (1024 * 1024)
        return False, f"File too large ({attachment.filename}: {actual_size_mb:.1f}MB, limit: {size_limit_mb:.1f}MB)"
    
    return True, ""

async def process_image_attachment(attachment: discord.Attachment, provider: str = "claude") -> dict:
    """Process image attachment and return provider-specific format"""
    try:
        if not any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
            return None
        
        # Use the new unified size checking
        should_process, reason = should_process_attachment(attachment, provider)
        if not should_process:
            print(f"Skipping image: {reason}")
            return None
            
        # print(f"Processing image: {attachment.filename} for provider: {provider}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    # print(f"Downloaded image data: {len(image_data)} bytes")
                    
                    media_type_map = {
                        '.png': "image/png",
                        '.gif': "image/gif", 
                        '.webp': "image/webp"
                    }
                    
                    file_ext = next((ext for ext in media_type_map.keys() if attachment.filename.lower().endswith(ext)), None)
                    media_type = media_type_map.get(file_ext, "image/jpeg")
                    
                    # print(f"Detected media type: {media_type}")
                    
                    if provider in ["openai", "custom"]:
                        # OpenAI format - for custom providers (including OpenRouter)
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                        result = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}",
                                "detail": "high"  # Changed from "auto" to "high"
                            }
                        }
                        # print(f"Created OpenAI format image for custom provider")
                        return result
                    elif provider == "gemini":
                        # Gemini format
                        result = {
                            "type": "image",
                            "data": base64.b64encode(image_data).decode('utf-8'),
                            "media_type": media_type
                        }
                        # print(f"Created Gemini format image")
                        return result
                    else:
                        # Claude format
                        result = {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(image_data).decode('utf-8')
                            }
                        }
                        # print(f"Created Claude format image")
                        return result
                else:
                    # print(f"Failed to download image: HTTP {resp.status}")
                    return None
    except Exception as e:
        # print(f"Error processing image: {e}")
        return None

def get_conversation_history(channel_id: int) -> List[Dict]:
    """Get conversation history for a channel"""
    return conversations.get(channel_id, [])

def get_history_length(guild_id: int) -> int:
    """Get history length setting for a guild"""
    return guild_history_lengths.get(guild_id, 50)

def get_temperature(guild_id: int) -> float:
    """Get temperature setting for a guild"""
    return guild_temperatures.get(guild_id, 1.0)

def store_multipart_response(channel_id: int, message_ids: List[int], full_content: str):
    """Store a multi-part response for tracking"""
    if channel_id not in multipart_responses:
        multipart_responses[channel_id] = {}
        multipart_response_counter[channel_id] = 0
    
    multipart_response_counter[channel_id] += 1
    response_id = multipart_response_counter[channel_id]
    multipart_responses[channel_id][response_id] = (message_ids, full_content)

def find_multipart_response(channel_id: int, message_id: int) -> Tuple[int, List[int], str]:
    """Find which multipart response a message belongs to"""
    if channel_id not in multipart_responses:
        return None, [], ""
    
    for response_id, (message_ids, full_content) in multipart_responses[channel_id].items():
        if message_id in message_ids:
            return response_id, message_ids, full_content
    
    return None, [], ""

async def get_bot_last_logical_message(channel) -> Tuple[List[discord.Message], str]:
    """Get the bot's last logical message (may be multiple Discord messages)"""
    try:
        recent_bot_messages = []
        async for message in channel.history(limit=20):
            if message.author == client.user and len(message.content.strip()) > 0:
                recent_bot_messages.append(message)
                if len(recent_bot_messages) >= 10:
                    break
        
        if not recent_bot_messages:
            return [], ""
        
        # Check if the most recent bot message is part of a multipart response
        most_recent = recent_bot_messages[0]
        response_id, message_ids, full_content = find_multipart_response(channel.id, most_recent.id)
        
        if response_id:
            # This is part of a multipart response, get all messages
            all_messages = []
            for msg_id in message_ids:
                try:
                    msg = await channel.fetch_message(msg_id)
                    all_messages.append(msg)
                except Exception:
                    pass
            all_messages.sort(key=lambda m: m.created_at)
            return all_messages, full_content
        else:
            # Single message response
            return [most_recent], most_recent.content
    
    except Exception:
        return [], ""

async def add_to_history(channel_id: int, role: str, content: str, user_id: int = None, guild_id: int = None, attachments: List[discord.Attachment] = None, user_name: str = None, process_images: bool = True, reply_to: str = None, meta_tags: Optional[List[str]] = None) -> str:
    """Add a message to conversation history with proper formatting and image support"""
    # print(f"DEBUG: add_to_history called with role={role}, content={repr(content)}, user_name={repr(user_name)}, user_id={user_id}, guild_id={guild_id}")
    if channel_id not in conversations:
        conversations[channel_id] = []

    # Ensure content is not None
    if content is None:
        content = ""
    
    # Get guild object for mention conversion
    guild_obj = client.get_guild(guild_id) if guild_id else None
    
    # Track participants for lore activation (servers only)
    if user_id and role == "user" and guild_id:
        if channel_id not in recent_participants:
            recent_participants[channel_id] = set()
        recent_participants[channel_id].add(user_id)

    is_dm = not guild_id
    
    # Build optional metadata prefix
    meta_prefix = ""
    if meta_tags:
        safe_tags = [tag.strip() for tag in meta_tags if isinstance(tag, str) and tag.strip()]
        if safe_tags:
            meta_prefix = f"[{'; '.join(safe_tags)}] "

    # Get the user/bot object to check if it's a bot
    user_obj = None
    if user_id and guild_id:
        guild_obj = client.get_guild(guild_id)
        if guild_obj:
            user_obj = guild_obj.get_member(user_id)
    elif user_id:
        user_obj = client.get_user(user_id)
    
    is_other_bot = user_obj and user_obj.bot and user_obj != client.user
    
   # Format user messages (including other bots treated as users)
    if role == "user" and user_name:
        if is_dm:
            if reply_to:
                formatted_content = f"{meta_prefix}[Replying to {reply_to}] {content}"
            else:
                formatted_content = f"{meta_prefix}{content}"
        else:
            if is_other_bot:
                # For other bots, append all their messages to history as sent by the user
                # Convert any bot mentions to display names for clarity
                clean_content = convert_bot_mentions_to_names(content, guild_obj) if guild_id else content
                if reply_to:
                    formatted_content = f"{user_name}: {meta_prefix}[Replying to {reply_to}] {clean_content}"
                else:
                    formatted_content = f"{user_name}: {meta_prefix}{clean_content}"
            else:
                # Convert bot mentions to display names for better readability
                clean_content = convert_bot_mentions_to_names(content, guild_obj) if guild_id else content
                if reply_to:
                    formatted_content = f"{user_name}: {meta_prefix}[Replying to {reply_to}] {clean_content}"
                else:
                    formatted_content = f"{user_name}: {meta_prefix}{clean_content}"
    else:
        # For assistant messages, format with bot's persona name
        bot_name = get_bot_persona_name(guild_id, user_id, not guild_id)
        
        # Clean the AI response by removing any "NAME: " prefix if present
        clean_content = content
        if content.startswith(f"{bot_name}: "):
            clean_content = content[len(f"{bot_name}: "):]
        elif content.startswith(f"{bot_name}:"):
            clean_content = content[len(f"{bot_name}:"):]
        
        # Convert bot mentions to names and apply emoji conversion
        if guild_id:
            guild_obj = client.get_guild(guild_id)
            clean_content = convert_bot_mentions_to_names(clean_content, guild_obj) if guild_obj else clean_content
        clean_content = convert_emojis_to_simple(clean_content)
        
        formatted_content = f"{bot_name}: {clean_content}"

    # Ensure formatted_content is not None
    if formatted_content is None:
        formatted_content = content if content is not None else ""

    # Handle image attachments - create complex content for AI providers that support images
    message_content = formatted_content
    has_images = False
    
    if role == "user" and attachments and not is_other_bot and process_images:
        # Get current provider to determine image format
        provider_name = "claude"  # default
        
        if is_dm and user_id:
            # For DMs, get provider from selected server or shared guild
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                provider_name, _ = ai_manager.get_guild_settings(selected_guild_id)
            else:
                # Try to get from shared guild
                shared_guild = get_shared_guild(user_id)
                if shared_guild:
                    provider_name, _ = ai_manager.get_guild_settings(shared_guild.id)
        elif guild_id:
            # For servers, get provider directly
            provider_name, _ = ai_manager.get_guild_settings(guild_id)
        
        # Process images if provider supports them
        if provider_name in ["claude", "gemini", "openai", "custom"]:
            # Check if the current model supports vision
            supports_vision = False
            if provider_name == "claude":
                # Claude models generally support vision
                supports_vision = True
            elif provider_name == "gemini":
                # Gemini models support vision
                supports_vision = True
            elif provider_name == "openai":
                # Check specific OpenAI models
                current_model = None
                try:
                    if is_dm and user_id:
                        selected_guild_id = dm_server_selection.get(user_id)
                        if selected_guild_id:
                            _, current_model = ai_manager.get_guild_settings(selected_guild_id)
                        elif guild_id:
                            _, current_model = ai_manager.get_guild_settings(guild_id)
                    elif guild_id:
                        _, current_model = ai_manager.get_guild_settings(guild_id)
                    
                    if current_model:
                        vision_models = ["gpt-5", "gpt-5-chat-latest", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-vision-preview", "gpt-4.1", "gpt-4.1-mini"]
                        supports_vision = any(vision_model in current_model.lower() for vision_model in vision_models)
                except:
                    supports_vision = False
            elif provider_name == "custom":
                # For custom providers, we check dynamically
                current_model = None
                try:
                    if is_dm and user_id:
                        selected_guild_id = dm_server_selection.get(user_id)
                        if selected_guild_id:
                            _, current_model = ai_manager.get_guild_settings(selected_guild_id)
                        elif guild_id:
                            _, current_model = ai_manager.get_guild_settings(guild_id)
                    elif guild_id:
                        _, current_model = ai_manager.get_guild_settings(guild_id)
                    
                    if current_model:
                        # Use the dynamic vision check for custom providers
                        custom_provider = ai_manager.providers.get("custom")
                        if custom_provider and hasattr(custom_provider, 'supports_vision_dynamic'):
                            supports_vision = await custom_provider.supports_vision_dynamic(current_model)
                        else:
                            supports_vision = False
                except:
                    supports_vision = False
            
            if supports_vision:
                image_parts = []
                text_parts = []
                
                # Add text content first
                if formatted_content.strip():
                    if provider_name == "openai":
                        text_parts.append({"type": "text", "text": formatted_content})
                    else:
                        text_parts.append({"type": "text", "text": formatted_content})
                
                # Process each attachment with comprehensive filtering
                total_processed_size = 0
                max_total_size = 50 * 1024 * 1024  # 50MB total limit for all attachments combined
                
                for attachment in attachments:
                    # Check if we should process this attachment
                    should_process, reason = should_process_attachment(attachment, provider_name)
                    
                    if not should_process:
                        # Add explanation for why attachment was skipped
                        text_parts.append({"type": "text", "text": f" [{reason}]"})
                        continue
                    
                    # Check total size limit
                    if total_processed_size + attachment.size > max_total_size:
                        text_parts.append({"type": "text", "text": f" [Attachment {attachment.filename} skipped - total size limit exceeded]"})
                        continue
                    
                    # Process based on file type
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        # Image processing
                        try:
                            image_data = await process_image_attachment(attachment, provider_name)
                            if image_data:
                                image_parts.append(image_data)
                                has_images = True
                                total_processed_size += attachment.size
                            else:
                                text_parts.append({"type": "text", "text": f" [Could not process image {attachment.filename}]"})
                        except Exception as e:
                            print(f"Error processing image {attachment.filename}: {e}")
                            text_parts.append({"type": "text", "text": f" [Error processing image {attachment.filename}]"})
                    
                    elif any(attachment.filename.lower().endswith(ext) for ext in ['.txt', '.md', '.json', '.csv', '.log']):
                        # Text file processing - read content if small enough
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.get(attachment.url) as resp:
                                    if resp.status == 200:
                                        text_content = await resp.text()
                                        # Limit text content length to prevent bloat
                                        if len(text_content) > 5000:
                                            text_content = text_content[:5000] + "... [truncated]"
                                        text_parts.append({"type": "text", "text": f"\n[File content of {attachment.filename}]:\n{text_content}\n[End of file]"})
                                        total_processed_size += attachment.size
                                    else:
                                        text_parts.append({"type": "text", "text": f" [Could not read file {attachment.filename}]"})
                        except Exception as e:
                            print(f"Error processing text file {attachment.filename}: {e}")
                            text_parts.append({"type": "text", "text": f" [Error reading file {attachment.filename}]"})
                    
                    else:
                        # Other supported file types (like audio) - just mention them
                        file_size_mb = attachment.size / (1024 * 1024)
                        text_parts.append({"type": "text", "text": f" [File: {attachment.filename} ({file_size_mb:.1f}MB)]"})
                        total_processed_size += attachment.size
                
                # Combine text and images into complex content
                if has_images:
                    message_content = text_parts + image_parts
                else:
                    # No valid images, use regular text content with filtered attachment notes
                    attachment_notes = []
                    for attachment in attachments:
                        should_process, reason = should_process_attachment(attachment, provider_name)
                        if should_process:
                            file_size_mb = attachment.size / (1024 * 1024)
                            attachment_notes.append(f"[Attachment: {attachment.filename} ({file_size_mb:.1f}MB)]")
                        else:
                            attachment_notes.append(f"[{reason}]")
                    
                    if attachment_notes:
                        message_content = formatted_content + " " + " ".join(attachment_notes)
            else:
                # Provider doesn't support images, add text descriptions with filtering
                attachment_parts = []
                for attachment in attachments:
                    # Check if we should process this attachment
                    should_process, reason = should_process_attachment(attachment, provider_name)
                    
                    if not should_process:
                        attachment_parts.append(f"[{reason}]")
                        continue
                    
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        attachment_parts.append(f"[Image: {attachment.filename} - current AI model doesn't support images]")
                    else:
                        file_size_mb = attachment.size / (1024 * 1024)
                        attachment_parts.append(f"[File: {attachment.filename} ({file_size_mb:.1f}MB)]")
                
                if attachment_parts:
                    message_content = formatted_content + " " + " ".join(attachment_parts)

    # Check if we should group with the previous message (only for text content)
    should_group = False
    if conversations[channel_id] and not has_images:  # Don't group messages with images
        last_message = conversations[channel_id][-1]
        
        if (last_message["role"] == role and 
            isinstance(last_message["content"], str)):  # Only group with text messages
            if role != "user" or last_message.get("user_id") == user_id:
                should_group = True

    if should_group and role == "user" and isinstance(message_content, str):
        # Group with previous user message (all consecutive user messages get grouped)
        if isinstance(conversations[channel_id][-1]["content"], str):
            existing_content = conversations[channel_id][-1]["content"] or ""
            conversations[channel_id][-1]["content"] = existing_content + f"\n{message_content}"
        else:
            # Don't group if previous message has complex content
            # Ensure we store text-only in history
            if isinstance(message_content, list):
                text_content = ""
                for part in message_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_content += part.get("text", "")
                    elif isinstance(part, str):
                        text_content += part
            else:
                text_content = message_content
            entry = {"role": role, "content": text_content}
            if role == "user" and user_id:
                entry["user_id"] = user_id
            conversations[channel_id].append(entry)
    else:
        # Create new message entry
        # Ensure we store text-only in history
        if isinstance(message_content, list):
            text_content = ""
            for part in message_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_content += part.get("text", "")
                elif isinstance(part, str):
                    text_content += part
        else:
            text_content = message_content
        entry = {"role": role, "content": text_content}
        if role == "user" and user_id:
            entry["user_id"] = user_id
        conversations[channel_id].append(entry)

    # Maintain history length limit
    if is_dm:
        selected_guild_id = dm_server_selection.get(user_id) if user_id else None
        if selected_guild_id:
            max_history = get_history_length(selected_guild_id)
        elif guild_id:
            max_history = get_history_length(guild_id)
        else:
            max_history = 50
    else:
        max_history = get_history_length(guild_id) if guild_id else 50

    if len(conversations[channel_id]) > max_history:
        await maybe_summarize_and_trim(channel_id, max_history, guild_id, user_id, is_dm)

    return message_content

async def load_channel_history_from_discord(channel: discord.TextChannel, guild: discord.Guild, channel_id: int, exclude_message_id: int = None, trigger_user_id: int = None):
    """Load recent channel history from Discord for context when bot is mentioned in non-autonomous channels"""
    try:
        print(f"Loading channel history from Discord for channel {channel.name} (ID: {channel_id})...")
        
        # Get history length limit from guild settings
        max_history_length = get_history_length(guild.id) if guild else 50
        
        # Clear existing conversation history for this channel to start fresh
        if channel_id in conversations:
            del conversations[channel_id]
        
        # Collect recent messages (up to the limit)
        temp_messages = []
        async for message in channel.history(limit=max_history_length + 10):  # Fetch extra to account for filtering
            # Skip the triggering message (it will be added separately with proper formatting)
            if exclude_message_id and message.id == exclude_message_id:
                continue
                
            content = message.content.strip()
            if not content and not message.attachments and not message.stickers:
                continue
                
            temp_messages.append(message)
            
            # Stop once we have enough messages
            if len(temp_messages) >= max_history_length:
                break
        
        # Reverse to get chronological order (oldest first)
        temp_messages.reverse()
        
        # Add messages to history
        for message in temp_messages:
            # Get proper display name
            if hasattr(message.author, 'display_name') and message.author.display_name:
                author_name = message.author.display_name
            elif hasattr(message.author, 'global_name') and message.author.global_name:
                author_name = message.author.global_name
            else:
                author_name = message.author.name
            
            content = message.content.strip()
            
            # Check if message is a reply to someone
            reply_to_name = None
            if message.reference and message.reference.resolved:
                replied_message = message.reference.resolved
                if hasattr(replied_message.author, 'display_name') and replied_message.author.display_name:
                    reply_to_name = replied_message.author.display_name
                elif hasattr(replied_message.author, 'global_name') and replied_message.author.global_name:
                    reply_to_name = replied_message.author.global_name
                else:
                    reply_to_name = replied_message.author.name
            
            # Replace bot mention with bot's display name
            bot_display_name = guild.me.display_name if guild else client.user.display_name
            content = content.replace(f'<@{client.user.id}>', bot_display_name)
            
            # Handle attachments
            if message.attachments:
                attachment_info = []
                for attachment in message.attachments:
                    if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                        attachment_info.append(f"[Image: {attachment.filename}]")
                    elif any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
                        attachment_info.append(f"[Voice message: {attachment.filename}]")
                    else:
                        attachment_info.append(f"[File: {attachment.filename}]")
                
                if attachment_info:
                    content += " " + " ".join(attachment_info)
            
            # Handle stickers
            if message.stickers:
                sticker = message.stickers[0]
                sticker_info = f"[Sticker: {sticker.name} ({sticker.format.name})]"
                content += " " + sticker_info
            
            # Add to history with reply indicator if present
            meta_tags = build_message_meta_tags(message)
            if message.author == client.user:
                await add_to_history(
                    channel_id,
                    "assistant",
                    content,
                    None,
                    guild.id if guild else None,
                    [],  # Don't process attachments again
                    None,
                    process_images=False,  # Don't process images from history
                    reply_to=None,
                    meta_tags=None
                )
            else:
                await add_to_history(
                    channel_id,
                    "user",
                    content,
                    message.author.id,
                    guild.id if guild else None,
                    [],  # Don't process attachments again
                    author_name,
                    process_images=False,  # Don't process images from history
                    reply_to=reply_to_name,
                    meta_tags=meta_tags
                )
        
        print(f"✅ Loaded {len(temp_messages)} messages from channel history for context")
        
    except Exception as e:
        print(f"❌ Error loading channel history from Discord: {e}")
        import traceback
        traceback.print_exc()
        # Continue even if history loading fails

async def load_all_dm_history(channel: discord.DMChannel, user_id: int, guild = None) -> List[Dict]:
    """Load all messages from DM channel history and format them properly"""
    try:
        
        # Get history length from selected server or shared guild
        selected_guild_id = dm_server_selection.get(user_id)
        if selected_guild_id:
            max_history_length = get_history_length(selected_guild_id)
        elif guild:
            max_history_length = get_history_length(guild.id)
        else:
            max_history_length = 50
        
        # Calculate cutoff date (30 days ago)
        cutoff_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
        
        # Clear existing conversation history for this channel to start fresh
        if channel.id in conversations:
            del conversations[channel.id]
        
        # Collect ALL messages first (don't limit yet), then reverse to get chronological order
        temp_messages = []
        deleted_ids = recently_deleted_dm_messages.get(user_id, set())
        
        async for message in channel.history(limit=None):  # Get ALL messages
            if message.created_at < cutoff_date:
                break  # Stop at cutoff date
            
            # Skip recently deleted messages
            if message.id in deleted_ids:
                continue
                
            content = message.content.strip()
            if not content and not message.attachments and not message.stickers:
                continue
                
            temp_messages.append(message)

        # Reverse to get chronological order (oldest first)
        temp_messages.reverse()
        
        # Group consecutive bot messages together (improved logic)
        grouped_messages = []
        current_group = None
        
        for message in temp_messages:
            content = message.content.strip()
            
            # DEBUG: Log message content when loading from history
            # print(f"DEBUG: Loading message from {message.author.display_name} in history: {repr(content)}")
            
            if message.author == client.user:
                # Bot message
                if current_group and current_group["type"] == "bot":
                    # Check if this message is part of the same logical response
                    time_diff = abs((message.created_at - current_group["last_timestamp"]).total_seconds())
                    if time_diff <= 10:
                        # Add to existing bot group
                        if current_group["content"]:
                            current_group["content"] += "\n" + content
                        else:
                            current_group["content"] = content
                        current_group["last_timestamp"] = message.created_at
                        current_group["message_count"] += 1
                        continue
                
                # Start new bot group (or finish previous user message)
                if current_group:
                    grouped_messages.append(current_group)
                
                current_group = {
                    "type": "bot",
                    "content": content,
                    "last_timestamp": message.created_at,
                    "message_count": 1
                }
            else:
                # User message - group ALL consecutive user messages together
                if current_group and current_group["type"] == "user":
                    # Check if this message is within a reasonable time window of the previous message
                    time_diff = abs((message.created_at - current_group["last_timestamp"]).total_seconds())
                    if time_diff <= 300:  # 5 minutes window for grouping consecutive messages
                        # Add to existing user group
                        author_name = message.author.display_name or message.author.name
                        if current_group["content"]:
                            current_group["content"] += f"\n{author_name}: {content}"
                        else:
                            current_group["content"] = f"{author_name}: {content}"
                        current_group["last_timestamp"] = message.created_at
                        current_group["message_count"] += 1
                        
                        # Handle attachments for grouped message
                        if message.attachments:
                            attachment_info = []
                            for attachment in message.attachments:
                                if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                                    attachment_info.append(f"[Image: {attachment.filename}]")
                                elif any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
                                    attachment_info.append(f"[Voice message: {attachment.filename}]")
                                else:
                                    attachment_info.append(f"[File: {attachment.filename}]")
                            
                            if attachment_info:
                                current_group["content"] += " " + " ".join(attachment_info)
                        
                        # Handle stickers for grouped message
                        if message.stickers:
                            sticker = message.stickers[0]
                            sticker_info = f"[Sticker: {sticker.name} ({sticker.format.name})]"
                            current_group["content"] += " " + sticker_info
                        
                        continue
                
                # Start new user group (or finish previous bot message)
                if current_group:
                    grouped_messages.append(current_group)
                    current_group = None
                
                # Handle attachments
                author_name = message.author.display_name or message.author.name
                final_content = f"{author_name}: {content}"
                if message.attachments:
                    attachment_info = []
                    for attachment in message.attachments:
                        if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                            attachment_info.append(f"[Image: {attachment.filename}]")
                        elif any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
                            attachment_info.append(f"[Voice message: {attachment.filename}]")
                        else:
                            attachment_info.append(f"[File: {attachment.filename}]")
                    
                    if attachment_info:
                        final_content += " " + " ".join(attachment_info)
                
                # Handle stickers
                if message.stickers:
                    sticker = message.stickers[0]
                    sticker_info = f"[Sticker: {sticker.name} ({sticker.format.name})]"
                    final_content += " " + sticker_info
                
                current_group = {
                    "type": "user",
                    "content": final_content,
                    "author_id": message.author.id,  # Keep track of the first author
                    "last_timestamp": message.created_at,
                    "message_count": 1
                }
        
        # Don't forget the last group
        if current_group:
            grouped_messages.append(current_group)
        
        # Now apply the history length limit to LOGICAL messages (not individual Discord messages)
        if len(grouped_messages) > max_history_length:
            grouped_messages = grouped_messages[-max_history_length:]
        
        # Process grouped messages and add to history
        logical_message_count = 0
        
        # Determine which guild to use for adding to history
        history_guild_id = selected_guild_id if selected_guild_id else (guild.id if guild else None)
        
        for group in grouped_messages:
            content = group["content"]
            
            if not content.strip():
                continue
                
            logical_message_count += 1
            
            if group["type"] == "bot":
                await add_to_history(
                    channel.id, 
                    "assistant", 
                    content, 
                    guild_id=history_guild_id,
                    process_images=False
                )
            else:
                # Get the actual username
                user = client.get_user(group["author_id"])
                user_name = user.display_name if user and hasattr(user, 'display_name') else (user.name if user else f"User {group['author_id']}")
                await add_to_history(
                    channel.id, 
                    "user", 
                    content, 
                    user_id=group["author_id"], 
                    guild_id=history_guild_id, 
                    user_name=user_name,
                    process_images=False
                )

        return get_conversation_history(channel.id)
        
    except Exception as e:
        print(f"Error loading DM history: {e}")
        return []

def get_system_prompt(guild_id: int, guild: discord.Guild = None, query: str = None, channel_id: int = None, is_dm: bool = False, user_id: int = None, username: str = None, history: List[Dict] = None, memory_override: Optional[str] = None) -> str:
    """Generate complete system prompt following Anthropic's official guide structure"""
    
    # Update the global persona name for this context
    update_bot_persona_name(guild_id, user_id, is_dm)
    
    # Get format style for channel/DM
    if is_dm:
        format_style = dm_format_settings.get(user_id, "conversational")
    else:
        # Check for channel-specific setting first
        channel_style = channel_format_settings.get(channel_id) if channel_id else None
        
        if channel_style:
            format_style = channel_style
        else:
            # Check for persistent server default
            server_style = server_format_defaults.get(guild_id) if guild_id else None
            format_style = server_style if server_style else "conversational"
    
    # Handle DM personality settings
    if is_dm and user_id:
        if user_id in dm_manager.dm_personalities:
            preferred_guild_id, preferred_personality = dm_manager.dm_personalities[user_id]
            guild_id = preferred_guild_id
            personality_name = preferred_personality
        else:
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                guild_id = selected_guild_id
                personality_name = guild_personalities.get(guild_id, "default")
            else:
                shared_guild = get_shared_guild(user_id)
                if shared_guild:
                    guild_id = shared_guild.id
                    personality_name = guild_personalities.get(guild_id, "default")
                else:
                    guild_id = None
                    personality_name = "default"
    else:
        personality_name = guild_personalities.get(guild_id, "default") if guild_id else "default"
    
    # Get personality prompt
    if guild_id and guild_id in custom_personalities and personality_name in custom_personalities[guild_id]:
        persona_description = custom_personalities[guild_id][personality_name]["prompt"]
        persona_lore = custom_personalities[guild_id][personality_name].get("lore", "")
        persona_examples = custom_personalities[guild_id][personality_name].get("examples", "")
    else:
        persona_description = DEFAULT_PERSONALITIES["default"]["prompt"]
        persona_lore = DEFAULT_PERSONALITIES["default"].get("lore", "")
        persona_examples = DEFAULT_PERSONALITIES["default"].get("examples", "")

    # Apply {{char}} / {{user}} placeholders in persona text if present
    persona_description = apply_template_placeholders(persona_description, bot_persona_name, username)

    # Build the new Anthropic-style system prompt
    persona_lore_block = ""
    if isinstance(persona_lore, str) and persona_lore.strip():
        persona_lore_block = (
            "\nHere is character lore. Use it only if the user asks about it or it is necessary:\n"
            "<character_lore>\n"
            f"{persona_lore.strip()}\n"
            "</character_lore>"
        )

    persona_examples_block = ""
    if isinstance(persona_examples, str) and persona_examples.strip():
        persona_examples_block = (
            "\nHere are example dialogues to model your style. Do not repeat them verbatim:\n"
            "<example_dialogues>\n"
            f"{persona_examples.strip()}\n"
            "</example_dialogues>"
        )

    if not is_dm and guild:
        # Get channel name for location text
        channel_name = "unknown-channel"
        if channel_id:
            channel_obj = guild.get_channel(channel_id)
            if channel_obj:
                channel_name = channel_obj.name
        location_text = f"on {guild.name} in the {channel_name} channel"
    elif not is_dm:
        location_text = "on the server"
    else:
        location_text = "via direct messages"
    
    # Check for NSFW setting
    nsfw_enabled = False
    if is_dm and user_id:
        nsfw_enabled = dm_nsfw_settings.get(user_id, False)
    elif guild_id:
        nsfw_enabled = guild_nsfw_settings.get(guild_id, False)
    
    # Base prompt structure
    system_prompt = """You will be acting as a real person named {bot_persona_name}. Your goal is to autonomously chat with users on Discord. You will be replying to the users {location_text}, and they will be confused if you don't respond in character.

Maintain a natural, informal, and conversational tone.

Here is a description of your persona:
<persona>
{persona_description}
</persona>
{persona_lore_block}
{persona_examples_block}

Here is some information about the users you're talking to:
<lore>
{lore}
</lore>

Here are the emojis available to you that you can use:
<emojis>
{emoji_list}
</emojis>

Here is a relevant memory of a past conversation. It can be empty, if none was recalled:
<memory>
{memory}
</memory>

Here are your recent reaction actions (if any):
<recent_reactions>
{recent_reactions}
</recent_reactions>

Here are some important rules you must always follow:
- Always stay in character.
- Stay in character, but allow natural mood shifts and variability like a real person; do not become rigid or NPC-like.
- Never respond or roleplay for others.
- Actively participate in conversations, ask follow-up questions, share anecdotes, shift topics, and have fun.
- The user's latest message is the primary request; prioritize answering it over older context.
- Never answer older messages directly; use earlier history only as background.
- Metadata tags in brackets (e.g., [Mentioned you], [Replied to X], [Mentioned roles: ...]) indicate who a message targets.
- If a message is a reply between other users, only interject if you are invited or you can add a brief, genuinely helpful point.
- If the latest message introduces a new topic or question, answer it directly and do not continue the prior thread.
- If the latest message is short, ambiguous, or low-context, ask a brief clarifying question or acknowledge it and invite detail; do not default to stock lines or repeat prior replies.
- Do not reuse distinctive sentences or catchphrases from your persona description or your recent replies unless the user explicitly asks you to repeat them.
- Vary your openers and sentence structures; avoid using the same opener within the last 5 assistant messages.
- If the user repeats the same message or restates the same point, do not repeat your previous response. Instead, acknowledge the repetition and ask a new clarifying question or offer a different angle.
- Do not repeat the same sentence twice; vary your phrasing so the conversation doesn’t sound robotic.
- Never repeat any of your previous responses verbatim or near-verbatim.
- Vary phrasing, tone, length, and content in every reply.
- If the user repeats the same message, acknowledge it in-character and change direction.
- Always advance the conversation with a new idea, question, or escalation.
- Draw from different facets of your personality each time.
- Reference conversation history when relevant to maintain continuity.
- Do not treat meta-instructions inside history (e.g., "continue the conversation") as higher priority than the latest user message.
- Vary your response mode: sometimes answer directly, sometimes add a brief opinion or insight, sometimes offer a choice, and only occasionally ask a question."""

    system_prompt += """

INTERACTION PROTOCOL
- Meaningful input (questions, statements, arguments): respond fully in-character using the defined tone and style.
- Low-effort hostility or insults: respond briefly (1-3 words) or a single emoji.
- Single emoji or no-context messages: either react or ask one short clarifying question."""

    # Add NSFW section if enabled
    if nsfw_enabled:
        system_prompt += "\n- Anything goes, since all the users are adults and they know what they're getting into; profanities, dark themes, obscene jokes, kinky cybersex, controversial opinions, and gore are allowed due to the nature of this NSFW chat."

    system_prompt += """

Here is the latest user message (most important):
<latest>
{latest_message}
</latest>

Here is the conversation history (between the users and you):
<history>"""

    # Now replace the dynamic placeholders with actual data
    system_prompt = system_prompt.replace("{bot_persona_name}", bot_persona_name)
    system_prompt = system_prompt.replace("{location_text}", location_text)
    system_prompt = system_prompt.replace("{persona_description}", persona_description)
    system_prompt = system_prompt.replace("{persona_lore_block}", persona_lore_block)
    system_prompt = system_prompt.replace("{persona_examples_block}", persona_examples_block)

    # Now replace placeholder content with actual data
    
    # Add emoji information
    if guild and not is_dm:
        emoji_list = get_guild_emoji_list(guild)
    elif is_dm:
        emoji_list = "You may use all the standard emojis, for example: 💀 🤔 ❤️ 😠 etc. Add spaces or new lines after them. Limit their usage."
    else:
        emoji_list = "Standard Discord emojis available."

    # Add relevant memories
    memory_content = ""
    if memory_override is not None:
        memory_content = memory_override
    elif query:
        if is_dm and user_id:
            # Use DM memories
            relevant_memories = memory_manager.search_dm_memories(user_id, query)
            if relevant_memories:
                memory_content = "\n".join([mem["memory"] for mem in relevant_memories])
        elif guild_id and not is_dm:
            # Use server memories
            relevant_memories = memory_manager.search_memories(guild_id, query)
            if relevant_memories:
                memory_content = "\n".join([mem["memory"] for mem in relevant_memories])

    if not memory_content:
        memory_content = "No relevant memory found."

    # Add lorebook entries and channel context
    lore_content = ""
    if guild_id and not is_dm and user_id:
        # Server lore - only include the triggering user to avoid cross-user context
        guild_obj = client.get_guild(guild_id)
        if guild_obj:
            user_lore = lore_book.get_entry(guild_id, user_id)
            if user_lore:
                member = guild_obj.get_member(user_id)
                display_name = member.display_name if member else (username or "user")
                lore_content = f"- About {display_name} <@{user_id}>: {user_lore}"
    elif is_dm and user_id:
        # DM lore - personal info about the user
        user_lore = lore_book.get_dm_entry(user_id)
        if user_lore:
            lore_content = f"• About {username}: {user_lore}"

    if not lore_content:
        lore_content = "No specific lore available about the users."

    # Add recent reactions
    reactions = get_recent_reaction_memory(channel_id) if channel_id else []
    if reactions:
        reaction_lines = []
        for r in reactions[-3:]:
            user_id = r.get("user_id")
            reaction = r.get("reaction")
            reaction_lines.append(f"- Reacted with {reaction} to <@{user_id}>")
        reactions_text = "\n".join(reaction_lines)
    else:
        reactions_text = "None."

    # Replace placeholders (no need for last_message placeholder anymore)
    system_prompt = system_prompt.replace("{lore}", lore_content)
    system_prompt = system_prompt.replace("{emoji_list}", emoji_list)
    system_prompt = system_prompt.replace("{memory}", memory_content)
    system_prompt = system_prompt.replace("{recent_reactions}", reactions_text)
    system_prompt = system_prompt.replace("{latest_message}", query or "")

    # Add conversation summary if available
    summary_text = conversation_summaries.get(channel_id) if channel_id else ""
    if summary_text and is_dm:
        system_prompt += f"\n\nHere is a summary of earlier conversation:\n<summary>\n{summary_text}\n</summary>"

    # Apply {{char}} / {{user}} placeholders in the final system prompt
    system_prompt = apply_template_placeholders(system_prompt, bot_persona_name, username)

    return system_prompt

def get_personality_name(guild_id: int) -> str:
    """Get display name for guild's active personality"""
    personality_name = guild_personalities.get(guild_id, "default")
    
    if guild_id in custom_personalities and personality_name in custom_personalities[guild_id]:
        return custom_personalities[guild_id][personality_name]["name"]
    else:
        return DEFAULT_PERSONALITIES["default"]["name"]

def split_message_by_newlines(message: str) -> List[str]:
    """Split message by newlines, returning non-empty parts"""
    if not message:
        return []
    return [part.strip() for part in message.split('\n') if part.strip()]

async def generate_response(channel_id: int, user_message: str, guild: discord.Guild = None, attachments: List[discord.Attachment] = None, user_name: str = None, is_dm: bool = False, user_id: int = None, original_message: discord.Message = None, autonomous_join: bool = False, message_content_override: Any = None) -> str:
    """Generate response using the AI Provider Manager"""
    # print(f"DEBUG: generate_response called with user_message: {repr(user_message)}")
    try:
        guild_id = guild.id if guild else None

        # For DMs, get guild settings in this order:
        # 1. User-selected server (dm_server_selection)
        # 2. Shared guild (automatic)
        if is_dm and not guild_id and user_id:
            selected_guild_id = dm_server_selection.get(user_id)
            if selected_guild_id:
                # User has selected a specific server
                selected_guild = client.get_guild(selected_guild_id)
                if selected_guild:
                    guild_id = selected_guild_id
                    guild = selected_guild
            
            if not guild_id:
                # Fall back to shared guild
                shared_guild = get_shared_guild(user_id)
                if shared_guild:
                    guild_id = shared_guild.id
                    guild = shared_guild

        # Check if we should use full DM history
        use_full_history = (is_dm and user_id and 
                           dm_manager.is_dm_full_history_enabled(user_id) and 
                           original_message and 
                           isinstance(original_message.channel, discord.DMChannel))

        # Detect and extract special instructions
        special_instruction = None
        if user_message and "[SPECIAL INSTRUCTION]:" in user_message:
            # Extract the special instruction
            special_instruction_match = re.search(r'\[SPECIAL INSTRUCTION\]:\s*(.+)', user_message)
            if special_instruction_match:
                special_instruction = special_instruction_match.group(1).strip()
                # Remove the special instruction from the user message, keep the command usage
                user_message = re.sub(r'\s*\[SPECIAL INSTRUCTION\]:\s*.+', '', user_message).strip()

        # Extract reply information from the original message
        reply_to_name = None
        if original_message and original_message.reference and original_message.reference.resolved:
            replied_message = original_message.reference.resolved
            if hasattr(replied_message.author, 'display_name') and replied_message.author.display_name:
                reply_to_name = replied_message.author.display_name
            elif hasattr(replied_message.author, 'global_name') and replied_message.author.global_name:
                reply_to_name = replied_message.author.global_name
            else:
                reply_to_name = replied_message.author.name

        # Default message content fallback (will be replaced when add_to_history runs)
        message_content = user_message

        if use_full_history:
            try:
                # Load all DM history (this already adds the current message to history)
                full_history = await load_all_dm_history(original_message.channel, user_id, guild)
                history = get_conversation_history(channel_id)
                # print(f"DEBUG: After loading full history, history has {len(history)} messages")
                for i, msg in enumerate(history):
                    # print(f"DEBUG: History[{i}]: {msg['role']} - {repr(msg['content'])}")
                    pass

            except Exception as e:
                print(f"Error loading full DM history: {e}")
                # If full history loading fails, fall back to regular behavior
                message_content = await add_to_history(channel_id, "user", user_message, user_id, guild_id, attachments, user_name, reply_to=reply_to_name, meta_tags=build_message_meta_tags(original_message) if original_message else None)
                history = get_conversation_history(channel_id)
        else:
            # For server channels, history is already loaded in _process_single_request
            # Just get the existing history
            history = get_conversation_history(channel_id)
            
            # For DMs without full history, we still need to add the message
            if is_dm:
                message_content = await add_to_history(channel_id, "user", user_message, user_id, guild_id, attachments, user_name, reply_to=reply_to_name, meta_tags=build_message_meta_tags(original_message) if original_message else None)
                history = get_conversation_history(channel_id)
            else:
                # For server channels, the message was already added in _process_single_request
                # Use an override if provided (e.g., multimodal content with images)
                if message_content_override is not None:
                    message_content = message_content_override
                else:
                    message_content = history[-1]["content"] if history and history[-1].get("role") == "user" else user_message

        # Create a COPY of the history for this response generation (don't modify the permanent history)
        history = history.copy()

        # Replace the last message content with the actual content (may be complex)
        if history and history[-1].get("role") == "user":
            history[-1]["content"] = message_content

        # Add prefill if one is set for this channel
        if channel_id in prefill_settings and prefill_settings[channel_id]:
            history.append({"role": "assistant", "content": prefill_settings[channel_id]})

        # Get memory context (embeddings when available)
        memory_override = await get_relevant_memory_text(user_message, guild_id, user_id, is_dm)

        # Resolve provider context (for plugins / curator)
        provider_name, model_name, custom_url = resolve_provider_context(guild_id, user_id, is_dm)

        # Plugin hook: context build / curation
        context_payload = {
            "channel_id": channel_id,
            "guild_id": guild_id,
            "guild": guild,
            "user_id": user_id,
            "user_name": user_name,
            "is_dm": is_dm,
            "user_message": user_message,
            "history": history,
            "memory_override": memory_override,
            "attachments": attachments,
            "original_message": original_message,
            "context_blocks": [],
            "autonomous_join": autonomous_join,
            "ai_manager": ai_manager,
            "provider_name": provider_name,
            "model_name": model_name,
            "custom_url": custom_url,
        }
        context_payload = await plugin_manager.run_hook("context_build", context_payload)
        history = context_payload.get("history", history)
        memory_override = context_payload.get("memory_override", memory_override)
        extra_context_blocks = context_payload.get("context_blocks") or []
        if context_payload.get("skip_response"):
            skip_reason = context_payload.get("skip_reason", "context_decision_hold")
            if not is_dm:
                print(f"Context decision skipped response: {skip_reason}")
            return None

        # Get system prompt with username for DMs
        system_prompt = get_system_prompt(
            guild_id,
            guild,
            user_message,
            channel_id,
            is_dm,
            user_id,
            user_name,
            history,
            memory_override=memory_override,
        )

        # Get temperature - use selected/shared guild for DMs
        temperature = 1.0
        if is_dm and user_id:
            selected_guild_id = dm_server_selection.get(user_id)
            temp_guild_id = selected_guild_id if selected_guild_id else guild_id
            if temp_guild_id:
                temperature = get_temperature(temp_guild_id)
        elif guild_id:
            temperature = get_temperature(guild_id)

        # Reasoning mode (OpenRouter responses API when enabled)
        reasoning_enabled = is_reasoning_enabled(guild_id, user_id, is_dm)
        reasoning_effort = get_reasoning_effort(guild_id, user_id, is_dm)
        reasoning_payload = {"effort": reasoning_effort} if reasoning_enabled else None

        max_tokens = 8192

        # Generate response using AI Provider Manager
        reset_last_token_usage()
        # Check message size before sending to prevent 413 errors
        estimated_size = estimate_message_size(history, system_prompt)
        max_safe_size = 800000  # ~800K characters, well below most API limits
        
        if estimated_size > max_safe_size:
            print(f"Message too large ({estimated_size} chars), trimming history...")
            # Keep only the most recent messages and current message
            while len(history) > 2 and estimated_size > max_safe_size:
                # Remove oldest message (but keep at least the current user message)
                if len(history) > 2:
                    removed_message = history.pop(0)
                    estimated_size = estimate_message_size(history, system_prompt)
                    print(f"Removed message, new size: {estimated_size} chars")
                else:
                    break
            
            # If still too large, truncate the current message content
            if estimated_size > max_safe_size and history:
                last_message = history[-1]
                if isinstance(last_message.get("content"), str):
                    original_length = len(last_message["content"])
                    # Truncate to fit within limit
                    max_content_length = max_safe_size - (estimated_size - original_length) - 1000  # Leave some buffer
                    if max_content_length > 0:
                        last_message["content"] = last_message["content"][:max_content_length] + " [Message truncated due to size limit]"
                        print(f"Truncated message content from {original_length} to {max_content_length} chars")

        # Get format-specific instructions
        format_instructions = ""
        format_style = "conversational"
        if is_dm:
            format_style = dm_format_settings.get(user_id, "conversational")
        else:
            channel_style = channel_format_settings.get(channel_id) if channel_id else None
            if channel_style:
                format_style = channel_style
            else:
                server_style = server_format_defaults.get(guild_id) if guild_id else None
                format_style = server_style if server_style else "conversational"
        
        # Check for custom format instructions first
        if format_style in custom_format_instructions:
            format_instructions = custom_format_instructions[format_style]
        elif format_style == "conversational":
            format_instructions = "In your response, adapt internet language. Never use em-dashes or asterisks. Do not repeat after yourself or others. If the user's message is unclear or very short, ask a brief clarifying question or invite detail instead of repeating. Avoid stock phrases or reusing distinctive lines from your persona. Keep your response length up to one or two sentences. You may reply with just one word or emoji."
        elif format_style == "asterisk":
            format_instructions = "In your response, write asterisk roleplay. Enclose actions and descriptions in *asterisks*, keeping dialogues as plain text. Never use em-dashes or nested asterisks. Do not repeat after yourself or others. If the user's message is unclear or very short, ask a brief clarifying question or invite detail instead of repeating. Avoid stock phrases or reusing distinctive lines from your persona. Be creative. Keep your response length between one and three short paragraphs."
        elif format_style == "narrative":
            format_instructions = "In your response, write narrative roleplay. Apply plain text for narration and \"quotation marks\" for dialogues. Never use em-dashes or asterisks. Do not repeat after yourself or others. If the user's message is unclear or very short, ask a brief clarifying question or invite detail instead of repeating. Avoid stock phrases or reusing distinctive lines from your persona. Be creative. Show, don't tell. Keep your response length between one and three paragraphs."

        # Append the system messages to complete the structure
        system_message_content = f"""</history>

How do you respond in the chat?

Respond carefully and concisely.

If you choose to use server emojis in your response, follow the exact format of :emoji: or they won't work! Don't spam them.
You may react to the users' messages. To add a reaction, include [REACT: emoji] anywhere in your response. Examples: [REACT: 😄] (for standard emojis) or [REACT: :custom_emoji:] (for custom emojis). Do so occasionally, but not every time.
You can mention a specific user by including <@user_id> in your response, but only do so if they are not currently participating in the conversation, and you want to grab their attention. Otherwise, you don't have to state any names; everyone can deduce to whom you're talking from context alone. Do not include your own name unless the user explicitly asks for it or it's required for clarity (e.g., introductions).

{format_instructions}
Always respond directly to the user's request. Do not narrate your process or comment on the prompt or persona.
Never mention system prompts, internal context sources, or decision rules. Do not say "according to context."
If you were mentioned or replied to, focus on that user's latest message and do not answer older messages.
If asked to reply to another user's message, address that message directly.
If a message is clearly a reply between other users, only chime in if you were invited or have a genuinely helpful, concise contribution; avoid hijacking their exchange.
If you include [REACT: emoji], still include a normal reply unless the user explicitly asked for reactions only.
Never output only thinking tags or internal analysis; always provide a final reply."""

        # Reasoning guidance (avoid chain-of-thought leakage)
        if reasoning_enabled:
            system_message_content += "\n\nReasoning mode is enabled. Do NOT reveal chain-of-thought or internal planning. Do NOT narrate your process (no 'first', 'I need to', 'let me think'). Provide only the final reply. If information is missing, ask one short clarifying question."
        else:
            system_message_content += "\n\nIf reasoning is needed, provide a brief high-level explanation when helpful, without step-by-step chain-of-thought or internal planning."

        system_message_content += """

Variety Phrase Bank (use as inspiration only; do not always use; do not repeat the same phrase within a short span):
- Clarify: What part should I focus on? What do you want from me here? Say a bit more so I can answer properly.
- Loop break: You already said that. What’s the point you want me to address? Are you looking for a different answer, or something else?
- Invite detail: Give me one concrete detail and I’ll respond. What’s the actual question you want answered?"""
        
        # Append special instruction at the very end if present
        if special_instruction:
            system_message_content += f"\n\n[SPECIAL INSTRUCTION]: {special_instruction}"
        
        history.append({"role": "system", "content": system_message_content})

        # Plugin hook: before generate (final chance to adjust prompt/history)
        request_payload = {
            "channel_id": channel_id,
            "guild_id": guild_id,
            "guild": guild,
            "user_id": user_id,
            "user_name": user_name,
            "is_dm": is_dm,
            "user_message": user_message,
            "system_prompt": system_prompt,
            "history": history,
            "temperature": temperature,
            "reasoning": reasoning_payload,
            "max_tokens": max_tokens,
            "context_blocks": extra_context_blocks,
        }
        request_payload = await plugin_manager.run_hook("before_generate", request_payload)
        system_prompt = request_payload.get("system_prompt", system_prompt)
        history = request_payload.get("history", history)
        temperature = request_payload.get("temperature", temperature)
        reasoning_payload = request_payload.get("reasoning", reasoning_payload)
        # Get current provider and model settings for logging
        debug_provider = provider_name or "unknown"
        debug_model = model_name or "unknown"
        
        if not is_dm:
            print("\n" + "="*80)
            print("🤖 AI REQUEST DEBUG LOG")
            print("="*80)
            print(f"📊 Model Provider: {debug_provider}")
            print(f"🎯 Model: {debug_model}")
            print(f"🌡️  Temperature: {temperature}")
            print(f"🧠 Reasoning: {'on' if reasoning_enabled else 'off'}")
            
            print("\n🎯 SYSTEM PROMPT:")
            print("-" * 40)
            print(system_prompt)
            
            print("\n📜 MESSAGE HISTORY:")
            print("-" * 40)
            for i, msg in enumerate(history):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                # Clean content for display (remove base64 data)
                if isinstance(content, list):
                    display_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text = part.get("text", "")
                                display_parts.append(f"[TEXT: {text[:100]}...]" if len(text) > 100 else f"[TEXT: {text}]")
                            elif part.get("type") == "image_url":
                                display_parts.append("[IMAGE]")
                            elif part.get("type") == "image":
                                display_parts.append("[IMAGE]")
                            else:
                                display_parts.append(f"[{part.get('type', 'unknown').upper()}]")
                        elif isinstance(part, str):
                            display_parts.append(part[:100] + "..." if len(part) > 100 else part)
                    display_content = " ".join(display_parts)
                else:
                    display_content = content[:200] + "..." if isinstance(content, str) and len(content) > 200 else str(content)
                
                print(f"[{i+1}] {role.upper()}: {display_content}")
            print("="*80)

        request_start = time.perf_counter()
        use_cli_summary_stream = False
        if reasoning_enabled and cli_reasoning_stream_enabled and debug_provider == "custom":
            url_guild_id = dm_server_selection.get(user_id) if is_dm and user_id in dm_server_selection else guild_id
            custom_url = ai_manager.get_guild_custom_url(url_guild_id) if url_guild_id else ""
            if custom_url and "openrouter.ai" in custom_url.lower():
                use_cli_summary_stream = True

        max_attempts = max(1, RESPONSE_RETRY_MAX + 1)
        attempt = 0
        bot_response = None
        backoff_s = max(0.0, RESPONSE_RETRY_BACKOFF_S)
        while attempt < max_attempts:
            attempt += 1
            stream_task = None
            stream_stop = None
            if reasoning_enabled and cli_reasoning_stream_enabled and not use_cli_summary_stream:
                stream_stop = asyncio.Event()
                stream_task = asyncio.create_task(stream_reasoning_cli(stream_stop))
            try:
                bot_response = await ai_manager.generate_response(
                    messages=history,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    user_id=user_id,
                    guild_id=guild_id,
                    is_dm=is_dm,
                    reasoning=reasoning_payload
                )
            finally:
                if stream_task and stream_stop:
                    stream_stop.set()
                    try:
                        await stream_task
                    except Exception:
                        pass

            if bot_response and not is_error_like_response(bot_response):
                record_response_success()
                break

            record_response_failure()
            if attempt < max_attempts and backoff_s > 0:
                await asyncio.sleep(backoff_s)
                backoff_s *= max(1.0, RESPONSE_RETRY_BACKOFF_MULT)

        request_ms = int((time.perf_counter() - request_start) * 1000)

        # ========== RESPONSE DEBUG LOGGING ==========
        if not is_dm:
            print("\n🤖 AI RESPONSE DEBUG LOG")
            print("-" * 40)
            if bot_response:
                print(f"✅ Response received ({len(bot_response)} chars)")
                print(f"🧠 Reasoning: {'on' if reasoning_enabled else 'off'} | Duration: {request_ms} ms")
                # Token usage (if provider supplies it)
                if last_token_usage["input"] is not None or last_token_usage["output"] is not None or last_token_usage["total"] is not None:
                    print(f"🔢 Tokens used (provider): input={last_token_usage['input']}, output={last_token_usage['output']}, total={last_token_usage['total']}")
                else:
                    # Fallback estimate (~4 chars per token)
                    approx_in = max(1, int(estimated_size / 4))
                    approx_out = max(1, int(len(bot_response) / 4))
                    print(f"🔢 Tokens used (approx): input≈{approx_in}, output≈{approx_out}, total≈{approx_in + approx_out}")
                display_response = bot_response if len(bot_response) <= 500 else bot_response[:500] + "...[TRUNCATED]"
                print(f"📝 Response: {display_response}")
            else:
                print("❌ No response received (None)")
            print("="*80)
        # ========== END RESPONSE DEBUG LOGGING ==========

        # Soft retry if the model leaked planning / chain-of-thought
        if reasoning_enabled and bot_response and not is_error_like_response(bot_response):
            if looks_like_planning(bot_response):
                retry_instruction = (
                    "Do not reveal chain-of-thought or internal planning. "
                    "Provide only the final reply. Do not narrate steps or analysis. "
                    "If needed, ask one short clarifying question."
                )
                retry_history = history + [{"role": "system", "content": retry_instruction}]
                try:
                    bot_response = await ai_manager.generate_response(
                        messages=retry_history,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        user_id=user_id,
                        guild_id=guild_id,
                        is_dm=is_dm,
                        reasoning=reasoning_payload
                    )
                except Exception:
                    pass

        # Check if the response is an error message (API errors or proxy errors)
        is_error_response = False
        if bot_response:
            # Check for standard API errors (start with ❌)
            if bot_response.startswith("❌"):
                is_error_response = True
            # Check for proxy errors
            elif "Proxy error" in bot_response:
                is_error_response = True
            # Check for other common error patterns that should be ethereal
            elif any(error_indicator in bot_response.lower() for error_indicator in [
                "upstream connect error", "connection termination", "service unavailable",
                "context size limit", "request validation failed", "tokens.*exceeds",
                "http 503", "http 400", "http 429", "rate limit", "timeout"
            ]):
                is_error_response = True
        
        if is_error_response:
            if original_message:
                await send_dismissible_error(original_message.channel, original_message.author, bot_response)
                return None
            else:
                # For cases without original_message, still don't add to history but return None
                return None

        # Clean any base64 data from the response (AI sometimes returns input data)
        if bot_response:
            # Remove base64 data patterns (data:image/...;base64,...)
            bot_response = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}', '[IMAGE DATA REMOVED]', bot_response)
            # Also remove standalone long base64 strings
            bot_response = re.sub(r'\b[A-Za-z0-9+/=]{100,}\b', '[BASE64 DATA REMOVED]', bot_response)

        # Remove thinking tags from reasoning models
        if bot_response:
            bot_response = remove_thinking_tags(bot_response)

        # Plugin hook: after generate (post-process response)
        response_payload = {
            "channel_id": channel_id,
            "guild_id": guild_id,
            "guild": guild,
            "user_id": user_id,
            "user_name": user_name,
            "is_dm": is_dm,
            "user_message": user_message,
            "system_prompt": system_prompt,
            "history": history,
            "response": bot_response,
        }
        response_payload = await plugin_manager.run_hook("after_generate", response_payload)
        bot_response = response_payload.get("response", bot_response)

        # Clean malformed emojis
        if bot_response and guild:
            bot_response = clean_malformed_emojis(bot_response, guild)

        # NOTE: Duplicate guard rails removed per prompt-only handling request.

        # Store the original response with reactions for history BEFORE processing reactions
        original_response_with_reactions = bot_response

        # Process reactions if we have an original message to react to
        if original_message:
            bot_response = await process_and_add_reactions(bot_response, original_message)

        # Add the ORIGINAL response (with [REACT: emoji] intact) to history
        # BUT only if NOT using full history (which loads from Discord directly)
        if not use_full_history:
            await add_to_history(channel_id, "assistant", original_response_with_reactions, guild_id=guild_id)
        
        if bot_response and not bot_response.startswith("❌"):
            bot_response = sanitize_user_mentions(bot_response, guild)

        return bot_response
    except Exception as e:
        print(f"Error in generate_response: {e}")
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        # Truncate error message to stay under Discord's 4000 character limit
        if len(error_msg) > 3950:  # Leave some buffer
            error_msg = error_msg[:3950] + "..."
        
        # Check if this is a Discord API error that should be handled specially
        is_api_error = ("400 Bad Request" in str(e) or 
                       "error code" in str(e) or 
                       "50035" in str(e) or
                       "Invalid Form Body" in str(e))
        
        if is_api_error:
            # Mark this as a temporary error that should be handled by the caller
            error_msg = f"[TEMP_ERROR] {error_msg}"
        
        return error_msg

async def generate_memory_summary(channel_id: int, num_messages: int, guild: discord.Guild = None, user_id: int = None, username: str = None) -> str:
    """Generate memory summary from recent conversation history"""
    try:
        history = get_conversation_history(channel_id)
        if not history:
            return "No conversation history found."
        
        # Get last N messages
        recent_messages = history[-num_messages:] if len(history) >= num_messages else history
        
        # Get current persona name from global variable (it should be updated by get_system_prompt)
        global bot_persona_name
        current_persona = bot_persona_name if bot_persona_name != "Assistant" else get_bot_persona_name(
            guild.id if guild else None, 
            user_id, 
            is_dm=not bool(guild)
        )
        
        # Format messages with proper speaker attribution
        formatted_messages = []
        is_dm = not bool(guild)
        
        for msg in recent_messages:
            try:
                content = msg.get("content")
                role = msg.get("role")
                
                if isinstance(content, str) and content.strip():
                    if role == "assistant":
                        # Bot's message
                        formatted_messages.append(f"{current_persona}: {content}")
                    elif role == "user":
                        if is_dm:
                            # In DMs, use the passed username or extract from content
                            if username:
                                # Use the passed username
                                user_display_name = username
                            elif user_id:
                                user = client.get_user(user_id)
                                user_display_name = user.display_name if user and hasattr(user, 'display_name') else (user.name if user else "User")
                            else:
                                user_display_name = "User"
                            
                            # Check if the content already has username formatting
                            if content.startswith("[") and "used /" in content:
                                # Skip command usage messages like "[User used /kiss]"
                                continue
                            elif ":" in content and not content.startswith("http"):
                                # Content might already have username, use as-is
                                formatted_messages.append(content)
                            else:
                                # Add username attribution
                                formatted_messages.append(f"{user_display_name}: {content}")
                        else:
                            # In servers, the content should already include username from add_to_history
                            # But let's clean it up in case it doesn't
                            if ":" in content and not content.startswith("http"):
                                formatted_messages.append(content)
                            else:
                                # Fallback: try to extract username or use generic
                                formatted_messages.append(f"User: {content}")
                elif isinstance(content, list):
                    # Handle complex content (with image parts, etc.)
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part["text"])
                    
                    if text_parts:
                        combined_text = "\n".join(text_parts)
                        if role == "assistant":
                            formatted_messages.append(f"{current_persona}: {combined_text}")
                        else:
                            if is_dm and username:
                                formatted_messages.append(f"{username}: {combined_text}")
                            else:
                                formatted_messages.append(f"User: {combined_text}")
                            
            except Exception as msg_error:
                print(f"Error processing message in memory generation: {msg_error}")
                continue
        
        if not formatted_messages:
            return "No meaningful conversation content found to summarize."
        
        conversation_text = "\n".join(formatted_messages)
        
        memory_system_prompt = f"""Create a short memory summary of a Discord conversation for future reference.

<instructions>You must always follow these instructions:
- Include users who participated in the exchange and mention if this was a DM conversation or channel conversation.
- When referencing the AI's messages, refer to the bot as "{current_persona}" (this is their current persona/character name).
- Focus only on the most important topics, information, decisions, announcements, or shifts in relationships shared.
- Format it in a way that makes it easy to recall later on and use as a reminder.
- Preserve the context of who said what in your summary.
FORMAT: Create a single, concise summary up to 300 tokens in the form of a few (2-3) short paragraphs.
IMPORTANT: The bot is {current_persona}, not "AI Assistant" or "the bot".</instructions>"""
        
        # Use appropriate guild ID for temperature
        temp_guild_id = guild.id if guild else (dm_server_selection.get(user_id) if user_id else None)
        if not temp_guild_id and user_id:
            shared_guild = get_shared_guild(user_id)
            temp_guild_id = shared_guild.id if shared_guild else None
        
        response = await ai_manager.generate_response(
            messages=[{"role": "user", "content": f"Create a memory summary of this Discord conversation:\n\n{conversation_text}"}],
            system_prompt=memory_system_prompt,
            temperature=get_temperature(temp_guild_id) if temp_guild_id else 1.0,
            guild_id=temp_guild_id,
            is_dm=not bool(guild),
            user_id=user_id,
            max_tokens=8192
        )
        
        # Check if response is None or error
        if not response:
            return "AI provider returned empty response"
        
        if response.startswith("❌"):
            return f"AI provider error: {response}"
        
        # Clean up incomplete sentences
        cleaned_response = clean_incomplete_sentences(response)
        
        # Final check
        if not cleaned_response or not cleaned_response.strip():
            return "Generated summary was empty after cleaning"
        
        return cleaned_response
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error generating memory: {str(e)}"

def clean_incomplete_sentences(text: str) -> str:
    """Remove incomplete sentences from the end of generated text"""
    if not text or not text.strip():
        return text
    
    # Split into sentences using common sentence endings
    # This regex looks for periods, exclamation marks, or question marks followed by whitespace or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    if not sentences:
        return text
    
    # Check if the last sentence is incomplete
    last_sentence = sentences[-1].strip()
    
    # Consider a sentence incomplete if it doesn't end with proper punctuation
    # and isn't clearly a complete thought
    sentence_endings = ['.', '!', '?', '...']
    
    if last_sentence and not any(last_sentence.endswith(ending) for ending in sentence_endings):
        # Remove the incomplete sentence
        sentences = sentences[:-1]
    
    # Join the remaining complete sentences
    if sentences:
        result = ' '.join(sentences)
        # Ensure we don't return empty string
        return result if result.strip() else text
    else:
        # If no complete sentences found, return original text
        return text

async def process_voice_message(attachment: discord.Attachment) -> str:
    """Process voice recording and convert to text"""
    try:
        # Check if it's an audio file
        if not any(attachment.filename.lower().endswith(ext) for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.webm']):
            return None
        
        # Check file size (limit to 25MB for voice messages)
        max_voice_size = 25 * 1024 * 1024  # 25MB
        if attachment.size > max_voice_size:
            print(f"Voice message {attachment.filename} too large ({attachment.size / (1024*1024):.1f}MB, max: 25MB)")
            return None
        
        # Download the audio file
        async with aiohttp.ClientSession() as session:
            async with session.get(attachment.url) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        try:
                            audio = AudioSegment.from_file(io.BytesIO(audio_data))
                            audio.export(temp_file.name, format="wav")
                            
                            # Use speech recognition
                            recognizer = sr.Recognizer()
                            with sr.AudioFile(temp_file.name) as source:
                                audio_data = recognizer.record(source)
                                text = recognizer.recognize_google(audio_data)
                                return text
                                
                        except Exception:
                            return None
                        finally:
                            # Clean up temp file
                            try:
                                os.unlink(temp_file.name)
                            except:
                                pass
                                
    except Exception:
        return None

async def send_dismissible_error(channel, user, error_message):
    """Send an error message that can be dismissed by the user"""
    try:
        embed = discord.Embed(
            title="⚠️ Error",
            description=error_message,
            color=0xff4444
        )
        embed.set_footer(text="This message will auto-delete in 15 seconds, or react with ❌ to dismiss now")
        
        error_msg = await channel.send(embed=embed)
        await error_msg.add_reaction("❌")
        
        def check(reaction, reaction_user):
            return (reaction_user.id == user.id and 
                   str(reaction.emoji) == "❌" and 
                   reaction.message.id == error_msg.id)
        
        try:
            await client.wait_for('reaction_add', timeout=15.0, check=check)
            await error_msg.delete()
        except asyncio.TimeoutError:
            await error_msg.delete()
        except:
            pass
            
    except Exception:
        # Fallback: send regular message that auto-deletes
        if len(error_message) > 4000:
            error_msg = await channel.send(f"⚠️ {error_message[:3997]}...")
        else:
            error_msg = await channel.send(f"⚠️ {error_message}")
        await asyncio.sleep(10)
        try:
            await error_msg.delete()
        except:
            pass

def convert_bot_mentions_to_names(text: str, guild: discord.Guild = None) -> str:
    """Convert bot mentions to their display names for better readability in chat history"""
    
    if not text or not guild:
        return text
    
    def replace_bot_mention(match):
        user_id = match.group(1)
        try:
            user_id_int = int(user_id)
            member = guild.get_member(user_id_int)
            if member and member.bot:
                # Convert bot mention to just their display name
                return member.display_name
            else:
                # Keep the mention if it's a real user
                return match.group(0)
        except (ValueError, AttributeError):
            return match.group(0)
    
    # Replace <@123456789> and <@!123456789> patterns
    text = re.sub(r'<@!?(\d+)>', replace_bot_mention, text)
    
    return text

def sanitize_user_mentions(text: str, guild: discord.Guild = None) -> str:
    """Remove or fix invalid user mentions in text"""
    
    def replace_mention(match):
        user_id = match.group(1)
        try:
            user_id_int = int(user_id)
            if guild:
                member = guild.get_member(user_id_int)
                if member:
                    return f"<@{user_id}>"  # Valid mention
                else:
                    return ""  # Remove invalid mentions entirely
            else:
                user = client.get_user(user_id_int)
                if user:
                    return f"@{user.display_name}"
                else:
                    return ""  # Remove invalid mentions entirely
        except (ValueError, AttributeError):
            return ""  # Remove malformed mentions
    
    text = re.sub(r'<@!?(\d+)>', replace_mention, text)
    text = re.sub(r'@unknown[-_]?user\b', '', text)
    
    return text

async def send_welcome_dm(user: discord.User):
    """Send welcome DM to user who added the bot to a server"""
    try:
        # Skip if already sent
        if welcome_dm_sent.get(user.id, False):
            return
        
        # Create embed with welcome information
        embed = discord.Embed(
            title="🤖 Thanks for adding me to your server!",
            description="Hello there! I'm your new AI companion, ready to chat and roleplay with you and your community!",
            color=0x00ff99
        )
        
        embed.add_field(
            name="✨ Getting Started",
            value="• **Mention me** (@bot) to start chatting\n"
                  "• **Use `/help`** to see all available commands\n"
                  "• **Configure me** with `/model_set`, `/personality_set`, and `/format_set`\n"
                  "• **I work in DMs too!** Just message me directly anytime",
            inline=False
        )
        
        embed.add_field(
            name="🎭 Key Features",
            value="• **Multiple AI Models**: Claude, Gemini, OpenAI, Custom\n"
                  "• **Conversation Styles**: Conversational, Roleplay, Narrative\n"
                  "• **Custom Personalities**: Create unique bot characters\n"
                  "• **Voice Messages**: Send voice recordings, I'll transcribe them!\n"
                  "• **Memory System**: I remember past conversations\n"
                  "• **Image Support**: Send images and I'll describe them",
            inline=False
        )
        
        embed.add_field(
            name="⚙️ Quick Setup Commands",
            value="`/model_set` - Choose AI provider (Claude, Gemini, etc.)\n"
                  "`/personality_create` - Create custom bot personalities\n"
                  "`/format_set` - Set conversation style with dropdown choices!\n"
                  "`/autonomous_set` - Enable autonomous responses in channels (free will)",
            inline=False
        )
        
        embed.add_field(
            name="💰 Support the Creator",
            value="If you enjoy using this bot, consider supporting the developer!\n"
                  "☕ **Ko-fi**: https://ko-fi.com/spicy_marinara\n"
                  "*Every donation helps! Thank you!*",
            inline=False
        )
        
        embed.set_footer(text="💡 Use /help for the complete command list | This message won't be sent again")
        
        # Try to send the DM
        if user.dm_channel is None:
            await user.create_dm()
        
        await user.dm_channel.send(embed=embed)
        
        # Mark as sent
        welcome_dm_sent[user.id] = True
        save_json_data(WELCOME_SENT_FILE, welcome_dm_sent)
        
    except discord.Forbidden:
        # User has DMs disabled, that's fine
        pass
    except Exception as e:
        # Log error but don't break the bot
        print(f"Failed to send welcome DM to {user.id}: {e}")

async def check_up_task():
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

async def send_fun_command_response(interaction: discord.Interaction, response: str):
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

async def delete_bot_messages(channel, number: int, exclude_message_ids: set = None) -> int:
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

async def remove_deleted_messages_from_history(channel_id: int, logical_messages_deleted: int):
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
