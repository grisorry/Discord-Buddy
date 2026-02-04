from typing import Any, Dict, Tuple, Callable, Optional, List

import aiohttp
import json


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


def extract_responses_reasoning_summary(data: Dict[str, Any]) -> str:
    output = data.get("output")
    summary_parts = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "reasoning":
                continue
            summary = item.get("summary")
            if isinstance(summary, list):
                for part in summary:
                    text = str(part).strip()
                    if text:
                        summary_parts.append(text)
            elif isinstance(summary, str):
                text = summary.strip()
                if text:
                    summary_parts.append(text)
    return "\n".join(summary_parts).strip()


async def stream_openrouter_responses(
    base_url: str,
    api_key: str,
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int = 60,
    on_summary_delta: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Any], Dict[str, str], str, str, int]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    header_map: Dict[str, str] = {}
    output_text_parts: List[str] = []
    summary_text_parts: List[str] = []
    part_types: Dict[Tuple[int, int], str] = {}
    final_response: Dict[str, Any] = {}
    buffer = ""
    reasoning_delta_count = 0

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{base_url}{endpoint}", json=payload, headers=headers, timeout=timeout) as resp:
            header_map = {k.lower(): v for k, v in resp.headers.items()}

            if resp.status != 200:
                try:
                    error_data = await resp.json()
                except Exception:
                    error_text = await resp.text()
                    error_data = {"error": {"message": error_text}}
                raise RuntimeError(f"OpenRouter streaming error HTTP {resp.status}: {error_data}")

            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        buffer = ""
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type")
                    if event_type == "response.output_item.added":
                        output_index = event.get("output_index")
                        item = event.get("item", {})
                        if isinstance(output_index, int) and isinstance(item, dict):
                            content = item.get("content") or []
                            if isinstance(content, list):
                                for idx, part in enumerate(content):
                                    if isinstance(part, dict):
                                        part_type = part.get("type")
                                        if isinstance(part_type, str):
                                            part_types[(output_index, idx)] = part_type
                        continue
                    elif event_type == "response.content_part.added":
                        output_index = event.get("output_index")
                        content_index = event.get("content_index")
                        part = event.get("part", {})
                        if isinstance(output_index, int) and isinstance(content_index, int) and isinstance(part, dict):
                            part_type = part.get("type")
                            if isinstance(part_type, str):
                                part_types[(output_index, content_index)] = part_type
                        continue
                    elif event_type == "response.content_part.delta":
                        output_index = event.get("output_index")
                        content_index = event.get("content_index")
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            part_type = part_types.get((output_index, content_index))
                            if part_type in ("output_text", "text"):
                                output_text_parts.append(delta)
                        continue
                    elif event_type == "response.output_item.done":
                        item = event.get("item", {})
                        if isinstance(item, dict):
                            content = item.get("content") or []
                            if isinstance(content, list):
                                parts = []
                                for part in content:
                                    if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                                        text = part.get("text")
                                        if isinstance(text, str):
                                            parts.append(text)
                                if parts:
                                    output_text_parts = ["".join(parts)]
                        continue
                    if event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            output_text_parts.append(delta)
                    elif event_type == "response.output_text.done":
                        text = event.get("text")
                        if isinstance(text, str):
                            output_text_parts = [text]
                    elif event_type == "response.reasoning_summary_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            summary_text_parts.append(delta)
                            if on_summary_delta:
                                on_summary_delta(delta)
                    elif event_type in ("response.reasoning.delta", "response.reasoning_text.delta"):
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            reasoning_delta_count += 1
                    elif event_type == "response.reasoning_summary_part.done":
                        part = event.get("part")
                        if isinstance(part, dict) and part.get("type") == "summary_text":
                            text = part.get("text")
                            if isinstance(text, str) and text:
                                summary_text_parts.append(text)
                                if on_summary_delta:
                                    on_summary_delta(text)
                    elif event_type in ("response.completed", "response.done"):
                        response_obj = event.get("response")
                        if isinstance(response_obj, dict):
                            final_response = response_obj
                    elif event_type in ("response.failed", "response.error", "error"):
                        error_info = event.get("error", {})
                        raise RuntimeError(f"OpenRouter Responses error: {error_info}")
                    elif event.get("object") == "response":
                        final_response = event

    output_text = "".join(output_text_parts).strip()
    summary_text = "".join(summary_text_parts).strip()

    if not summary_text and final_response:
        summary_text = extract_responses_reasoning_summary(final_response)
    if not output_text and final_response:
        output_text = extract_responses_output_text(final_response)

    return final_response, header_map, output_text, summary_text, reasoning_delta_count

