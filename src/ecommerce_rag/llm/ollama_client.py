from __future__ import annotations

import logging
from typing import Sequence

import ollama


LOGGER = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434") -> None:
        self.model = model
        self.client = ollama.Client(host=host)

    def chat(self, messages: Sequence[dict]) -> str:
        try:
            response = self.client.chat(model=self.model, messages=list(messages))
            content = response.get("message", {}).get("content", "")
            return content.strip()
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            LOGGER.exception("Ollama request failed: %s", exc)
            return (
                "The language model service is unavailable right now. "
                "Please confirm Ollama is running and the configured model is available."
            )
