from typing import Dict, List, Optional
import asyncio
import re
from google import genai
from google.genai import types  # type: ignore

from discord_buddy.providers.base import AIProvider
from discord_buddy.providers.token_usage import update_last_token_usage


class GeminiProvider(AIProvider):
    """Gemini AI provider using Google's API."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        if api_key:
            self.client = genai.Client(api_key=api_key)

    async def generate_response(
        self,
        messages: List[Dict],
        system_prompt: str,
        temperature: float = 1.0,
        model: str = None,
        max_output_tokens: int = 8192,
        reasoning: Optional[dict] = None,
    ) -> str:
        if not self.api_key:
            return "❌ Gemini API key not configured. Please contact the bot administrator."

        try:
            model = model or self.get_default_model()

            gemini_messages = []
            for i, msg in enumerate(messages):
                try:
                    role = "user" if msg["role"] == "user" else "model"
                    content = msg["content"]

                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict):
                                if part.get("type") == "text":
                                    parts.append({"text": part["text"]})
                                elif part.get("type") == "image":
                                    parts.append(
                                        {
                                            "inline_data": {
                                                "mime_type": part["media_type"],
                                                "data": part["data"],
                                            }
                                        }
                                    )

                        if parts:
                            gemini_messages.append({"role": role, "parts": parts})

                    elif isinstance(content, str) and content.strip():
                        gemini_messages.append({"role": role, "parts": [{"text": content}]})

                except Exception as msg_error:
                    print(f"Gemini: Error processing message {i}: {msg_error}")
                    continue

            if not gemini_messages:
                gemini_messages = [{"role": "user", "parts": [{"text": "Hello"}]}]
            elif gemini_messages[-1]["role"] != "user":
                gemini_messages.append(
                    {"role": "user", "parts": [{"text": "Continue the conversation naturally."}]}
                )

            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_prompt,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                    types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF"),
                ],
            )

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=gemini_messages,
                config=generation_config,
            )

            try:
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    update_last_token_usage(
                        input_tokens=getattr(usage, "prompt_token_count", None),
                        output_tokens=getattr(usage, "candidates_token_count", None),
                        total_tokens=getattr(usage, "total_token_count", None),
                    )
            except Exception:
                pass

            if hasattr(response, "text") and response.text:
                if any(
                    error_indicator in response.text.lower()
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
                    return f"❌ Gemini API error: {response.text}"

                clean_text = re.sub(
                    r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}",
                    "[IMAGE DATA REMOVED]",
                    response.text,
                )
                clean_text = re.sub(r"\b[A-Za-z0-9+/=]{100,}\b", "[BASE64 DATA REMOVED]", clean_text)
                return clean_text

            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, "text"):
                                text_parts.append(part.text)
                        if text_parts:
                            response_text = "".join(text_parts)
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
                                return f"❌ Gemini API error: {response_text}"

                            response_text = re.sub(
                                r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{50,}",
                                "[IMAGE DATA REMOVED]",
                                response_text,
                            )
                            response_text = re.sub(
                                r"\b[A-Za-z0-9+/=]{100,}\b", "[BASE64 DATA REMOVED]", response_text
                            )
                            return response_text

                if hasattr(candidate, "finish_reason"):
                    if candidate.finish_reason == "SAFETY":
                        return "❌ Gemini response blocked by safety filters. Try rephrasing your request."
                    if candidate.finish_reason == "MAX_TOKENS":
                        return "❌ Gemini response was cut off due to token limit."
                    if candidate.finish_reason == "RECITATION":
                        return "❌ Gemini response blocked due to recitation concerns."
                    return f"❌ Gemini stopped generation: {candidate.finish_reason}"

            return "❌ Gemini returned empty response (no text generated)"

        except Exception as e:
            return f"❌ Gemini API error: {str(e)}"

    def get_available_models(self) -> List[str]:
        return [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]

    def get_default_model(self) -> str:
        return "gemini-2.5-flash"

    def is_available(self) -> bool:
        return bool(self.api_key)
