from typing import Any, Dict, Tuple

import aiohttp


async def post_openrouter_json(base_url: str, api_key: str, endpoint: str, payload: Dict[str, Any], timeout: int = 60) -> Tuple[Dict[str, Any], Dict[str, str]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}{endpoint}", json=payload, headers=headers, timeout=timeout) as resp:
            try:
                data = await resp.json()
            except Exception:
                text = await resp.text()
                data = {"error": text}
            header_map = {k.lower(): v for k, v in resp.headers.items()}
    if not isinstance(data, dict):
        data = {"data": data}
    return data, header_map


def extract_openrouter_usage(data: Dict[str, Any]) -> Dict[str, Any]:
    usage = data.get("usage")
    if not isinstance(usage, dict) or not usage:
        inner = data.get("data")
        if isinstance(inner, dict):
            usage = inner.get("usage")
    return usage if isinstance(usage, dict) else {}


def extract_openrouter_generation_id(data: Dict[str, Any], headers: Dict[str, str]) -> str:
    return (
        headers.get("x-openrouter-generation-id")
        or data.get("generation_id")
        or data.get("id")
        or ""
    )


def extract_responses_output_text(data: Dict[str, Any]) -> str:
    output = data.get("output")
    if isinstance(output, list):
        parts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if isinstance(c, dict) and c.get("type") in ["output_text", "text"]:
                        parts.append(c.get("text", ""))
            elif item.get("type") in ["output_text", "text"]:
                parts.append(item.get("text", ""))
        text = "".join(parts).strip()
        if text:
            return text
    text = data.get("output_text") or data.get("text") or ""
    return text.strip() if isinstance(text, str) else ""


def extract_chat_output_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        return content if isinstance(content, str) else str(content)
    return ""
