from __future__ import annotations

from typing import Optional

from PIL import Image

from ..config import AppConfig
from ..llm import OllamaClient, QueryRouter
from ..retrieval import ProductContext, RetrievalService


class AnswerPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm_client = OllamaClient(model=config.llm_model, host=config.ollama_host)
        self.router = QueryRouter(client=self.llm_client)
        self.retrieval = RetrievalService(config=config)

    def answer_user_query(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Image.Image] = None,
        top_k: Optional[int] = None,
    ) -> str:
        raw_q = (query_text or "").strip()
        selected_top_k = top_k if top_k is not None else self.config.retrieval_top_k

        if not raw_q and query_image is not None:
            return self._handle_image_query(query_image, selected_top_k)
        if not raw_q and query_image is None:
            return "Please enter a product question, upload an image, or provide both."

        routed = self.router.route(raw_q)
        retrieval_query = (routed.retrieval_query or raw_q).strip()

        if routed.task_type == "text":
            return self._handle_text_query(retrieval_query, selected_top_k, query_image=query_image)
        if routed.task_type == "image":
            if query_image is None:
                return "Please upload an image so I can identify the product."
            return self._handle_image_query(query_image, selected_top_k)
        if routed.task_type == "image_request":
            return self._handle_image_request(retrieval_query, selected_top_k)
        
        return "Sorry, I could not classify your request."

    def _handle_text_query(self, query_text: str, top_k: int, query_image: Optional[Image.Image]) -> str:
        results = self.retrieval.search(query_text=query_text, query_image=query_image, top_k=top_k)
        if not results:
            return "No matching products were found in the retrieval index."
        contexts = self.retrieval.get_product_contexts(results)
        if not contexts:
            return "Retrieval returned candidates, but product metadata could not be resolved."
        prompt = self._build_text_answer_prompt(query_text=query_text, product=contexts[0])
        return self.llm_client.chat(
            [
                {"role": "system", "content": "You are a helpful e-commerce product assistant."},
                {"role": "user", "content": prompt},
            ]
        )

    def _handle_image_query(self, query_image: Image.Image, top_k: int) -> str:
        results = self.retrieval.search(query_text=None, query_image=query_image, top_k=top_k)
        if not results:
            return "I could not identify a close product match from the image."
        contexts = self.retrieval.get_product_contexts(results)
        if not contexts:
            return "Image retrieval succeeded, but product metadata is missing."
        prompt = self._build_image_answer_prompt(product=contexts[0])
        return self.llm_client.chat(
            [
                {"role": "system", "content": "You are a helpful e-commerce product assistant."},
                {"role": "user", "content": prompt},
            ]
        )

    def _handle_image_request(self, query_text: str, top_k: int) -> str:
        results = self.retrieval.search(query_text=query_text, query_image=None, top_k=top_k)
        if not results:
            return "No matching products were found for that image request."
        contexts = self.retrieval.get_product_contexts(results)
        if not contexts:
            return "A product match was found, but product metadata is unavailable."
        product = contexts[0]
        prompt = (
            "User wants a product image. Respond with a short product summary and include "
            f"this markdown image link exactly once: ![]({product.image_url})\n\n"
            f"Product name: {product.name}\n"
            f"Product details: {product.about}"
        )
        return self.llm_client.chat(
            [
                {"role": "system", "content": "You are a helpful e-commerce product assistant."},
                {"role": "user", "content": prompt},
            ]
        )

    @staticmethod
    def _build_text_answer_prompt(query_text: str, product: ProductContext) -> str:
        return (
            "Use only the provided product data to answer the user question.\n"
            f"User question: {query_text}\n\n"
            "Top retrieved product:\n"
            f"Name: {product.name}\n"
            f"Brand: {product.brand}\n"
            f"Price: {product.price}\n"
            f"Features: {product.spec}\n"
            f"Description: {product.about}\n"
            f"Usage: {product.usage}\n"
        )

    @staticmethod
    def _build_image_answer_prompt(product: ProductContext) -> str:
        return (
            "The user submitted a product image.\n"
            "Describe what the product is and how it is typically used, using only this retrieval context:\n"
            f"Name: {product.name}\n"
            f"Description: {product.about}\n"
            f"Features: {product.spec}\n"
            f"Usage: {product.usage}\n"
        )
