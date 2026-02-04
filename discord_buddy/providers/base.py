from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class AIProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict],
        system_prompt: str,
        temperature: float = 1.0,
        model: str = None,
        max_tokens: int = 8192,
        reasoning: Optional[dict] = None,
    ) -> str:
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass
