from typing import Dict, List, Optional
import re
from openai import AsyncOpenAI

from discord_buddy.providers.base import AIProvider
from discord_buddy.providers.token_usage import update_last_token_usage


class OpenAIProvider(AIProvider):
    """OpenAI provider supporting ChatGPT models."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        if api_key:
            self.client = AsyncOpenAI(api_key=api_key)

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
            return "? OpenAI API key not configured. Please contact the bot administrator."

        try:
            model = model or self.get_default_model()

            formatted_messages = [{"role": "system", "content": system_prompt}]

            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                if isinstance(content, list):
                    openai_content = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                openai_content.append({"type": "text", "text": part["text"]})
                            elif part.get("type") == "image_url":
                                openai_content.append(part)
                            elif part.get("type") == "image":
                                if "data" in part and "media_type" in part:
                                    openai_content.append(
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:{part['media_type']};base64,{part['data']}",
                                                "detail": "high",
                                            },
                                        }
                                    )

                    if openai_content:
                        formatted_messages.append({"role": role, "content": openai_content})

                elif isinstance(content, str) and content.strip():
                    formatted_messages.append({"role": role, "content": content})

            vision_models = [
                "gpt-5",
                "gpt-5-chat-latest",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4-vision-preview",
                "gpt-4.1",
                "gpt-4.1-mini",
            ]
            supports_vision = any(vision_model in model.lower() for vision_model in vision_models)

            if not supports_vision:
                for message in formatted_messages:
                    if isinstance(message.get("content"), list):
                        text_parts = []
                        for part in message["content"]:
                            if part.get("type") == "text":
                                text_parts.append(part["text"])
                            elif part.get("type") == "image_url":
                                text_parts.append("[Image was provided but this model doesn't support vision]")
                        message["content"] = " ".join(text_parts)

            response = await self.client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            response_text = response.choices[0].message.content

            try:
                if hasattr(response, "usage") and response.usage:
                    update_last_token_usage(
                        input_tokens=getattr(response.usage, "prompt_tokens", None),
                        output_tokens=getattr(response.usage, "completion_tokens", None),
                        total_tokens=getattr(response.usage, "total_tokens", None),
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
                return f"? OpenAI API error: {response_text}"

            response_text = re.sub(
                r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}",
                "[IMAGE DATA REMOVED]",
                response_text,
            )
            response_text = re.sub(r"\b[A-Za-z0-9+/=]{100,}\b", "[BASE64 DATA REMOVED]", response_text)

            return response_text

        except Exception as e:
            return f"? OpenAI API error: {str(e)}"

    def get_available_models(self) -> List[str]:
        return [
            "gpt-5",
            "gpt-5-chat-latest",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o3-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
        ]

    def get_default_model(self) -> str:
        return "gpt-4.1"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def supports_vision(self, model: str) -> bool:
        vision_models = [
            "gpt-5",
            "gpt-5-chat-latest",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-vision-preview",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o3-preview",
            "gpt-4",
        ]
        return any(vision_model in model.lower() for vision_model in vision_models)
