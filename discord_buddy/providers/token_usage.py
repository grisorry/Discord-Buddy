from typing import Dict, Optional

last_token_usage: Dict[str, Optional[int]] = {"input": None, "output": None, "total": None}


def update_last_token_usage(
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
) -> None:
    if input_tokens is not None:
        last_token_usage["input"] = input_tokens
    if output_tokens is not None:
        last_token_usage["output"] = output_tokens
    if total_tokens is not None:
        last_token_usage["total"] = total_tokens
    if last_token_usage["total"] is None and last_token_usage["input"] is not None and last_token_usage["output"] is not None:
        last_token_usage["total"] = last_token_usage["input"] + last_token_usage["output"]


def reset_last_token_usage() -> None:
    last_token_usage["input"] = None
    last_token_usage["output"] = None
    last_token_usage["total"] = None
