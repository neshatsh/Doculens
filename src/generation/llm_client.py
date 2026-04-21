"""Unified LLM client supporting OpenAI and Anthropic backends."""
from __future__ import annotations
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.config import get_settings
from src.utils.logger import logger

settings = get_settings()


class LLMClient:
    """
    Thin wrapper around OpenAI / Anthropic APIs.
    Swap provider via config — same interface either way.
    Includes retry logic with exponential backoff for rate limits.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self._client = None

    def _get_openai_client(self):
        from openai import OpenAI
        return OpenAI(api_key=settings.openai_api_key)

    def _get_anthropic_client(self):
        import anthropic
        return anthropic.Anthropic(api_key=settings.anthropic_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a system + user message and return the assistant's text response.
        Retries up to 3 times on transient errors.
        """
        tokens = max_tokens or self.max_tokens

        if self.provider == "openai":
            return self._openai_complete(system_prompt, user_message, tokens)
        elif self.provider == "anthropic":
            return self._anthropic_complete(system_prompt, user_message, tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _openai_complete(self, system: str, user: str, max_tokens: int) -> str:
        client = self._get_openai_client()
        logger.debug(f"OpenAI request: model={self.model}, max_tokens={max_tokens}")
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _anthropic_complete(self, system: str, user: str, max_tokens: int) -> str:
        import anthropic
        client = self._get_anthropic_client()
        logger.debug(f"Anthropic request: model={self.model}, max_tokens={max_tokens}")
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=self.temperature,
        )
        return response.content[0].text if response.content else ""

    @property
    def provider_info(self) -> dict:
        return {"provider": self.provider, "model": self.model}
