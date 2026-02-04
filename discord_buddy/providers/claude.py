from typing import Dict, List, Optional
import asyncio
import re
import anthropic

from discord_buddy.providers.base import AIProvider
from discord_buddy.providers.token_usage import update_last_token_usage


class ClaudeProvider(AIProvider):
    """Claude AI provider using Anthropic API."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

    async def generate_response(
        self,
        messages: List[Dict],
        system_prompt: str,
        temperature: float = 1.0,
        model: str = None,
        max_tokens: int = 8192,
        reasoning: Optional[dict] = None,
    ) -> str:
        if not self.api_key:
            return "❌ Claude API key not configured. Please contact the bot administrator."

        try:
            model = model or self.get_default_model()

            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
                stream=False,
            )

            response_text = response.content[0].text
            try:
                if hasattr(response, "usage") and response.usage:
                    update_last_token_usage(
                        input_tokens=getattr(response.usage, "input_tokens", None),
                        output_tokens=getattr(response.usage, "output_tokens", None),
                    )
            except Exception:
                pass

            if any(
                error_indicator in response_text.lower()
                for error_indicator in [
                    "proxy error",
                    "upstream connect error",
                    "connection termination",
                    "service unavailable",
                    "context size limit",
                    "request validation failed",
                    "tokens.*exceeds",
                    "http 503",
                    "http 400",
                    "http 429",
                    "rate limit",
                    "timeout",
                ]
            ):
                return f"❌ Claude API error: {response_text}"

            response_text = re.sub(
                r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}",
                "[IMAGE DATA REMOVED]",
                response_text,
            )
            response_text = re.sub(r"\b[A-Za-z0-9+/=]{100,}\b", "[BASE64 DATA REMOVED]", response_text)
            return response_text
        except Exception as e:
            return f"❌ Claude API error: {str(e)}"

    def get_available_models(self) -> List[str]:
        return [
            "claude-opus-4-1",
            "claude-opus-4",
            "claude-opus-4-0",
            "claude-sonnet-4-0",
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
        ]

    def get_default_model(self) -> str:
        return "claude-3-7-sonnet-latest"

    def is_available(self) -> bool:
        return bool(self.api_key)
