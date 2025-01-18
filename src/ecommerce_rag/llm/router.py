from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .ollama_client import OllamaClient


@dataclass(frozen=True)
class RoutedIntent:
    task_type: str
    retrieval_query: Optional[str]


class QueryRouter:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def route(self, user_question: str) -> RoutedIntent:
        question = (user_question or "").strip()
        if not question:
            return RoutedIntent(task_type="image", retrieval_query=None)

        system_prompt = (
            "You are a smart query router for a multimodal e-commerce assistant.\n\n"
            "Step 1: Classify the user's question into one of the following types:\n"
            "• 'text' - text-based product search (e.g. features, comparisons)\n"
            "• 'image' - identifying or describing the uploaded product image\n"
            "• 'image_request' - asking to show a specific product image\n\n"
            "Step 2: If the type is 'text' or 'image_request', also provide a compact retrieval query "
            "(focusing on brand, type, and key attributes).\n"
            "Respond ONLY as:\n"
            "type: <text|image|image_request>\n"
            "query: <short query text OR empty if 'image'>"
        )

        content = self.client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        ).lower()

        task_type = "text"
        retrieval_query: Optional[str] = None
        for line in content.splitlines():
            if line.startswith("type:"):
                parsed = line.replace("type:", "", 1).strip()
                if parsed in {"text", "image", "image_request"}:
                    task_type = parsed
            elif line.startswith("query:"):
                parsed_query = line.replace("query:", "", 1).strip()
                retrieval_query = parsed_query or None

        return RoutedIntent(task_type=task_type, retrieval_query=retrieval_query)
