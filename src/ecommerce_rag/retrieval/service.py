from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

from ..config import AppConfig


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProductContext:
    name: str
    brand: str
    price: str
    spec: str
    about: str
    usage: str
    image_url: str


class RetrievalService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model: Optional[CLIPModel] = None
        self.clip_processor: Optional[CLIPProcessor] = None
        self.clip_tokenizer: Optional[CLIPTokenizer] = None
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.product_ids: Optional[Sequence[str]] = None
        self.df_cleaned: Optional[pd.DataFrame] = None
        self.last_error: Optional[str] = None

    def initialize(self) -> None:
        if self.index is not None and self.df_cleaned is not None:
            return
        if not self.config.data_csv_path.is_file():
            self.last_error = f"Missing product catalog: {self.config.data_csv_path}"
            LOGGER.warning(self.last_error)
            return
        if not self.config.embeddings_path.is_file() or not self.config.product_ids_path.is_file():
            self.last_error = (
                "Missing retrieval artifacts. Expected files: "
                f"{self.config.embeddings_path} and {self.config.product_ids_path}"
            )
            LOGGER.warning(self.last_error)
            return

        try:
            self.df_cleaned = pd.read_csv(self.config.data_csv_path)
            self.clip_model = CLIPModel.from_pretrained(self.config.clip_model_name).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_name)
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.config.clip_model_name)
            self.embeddings = np.load(self.config.embeddings_path).astype(np.float32)
            self.embeddings = self._normalize(self.embeddings)
            with self.config.product_ids_path.open("rb") as handle:
                self.product_ids = pickle.load(handle)
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            self.last_error = str(exc)
            LOGGER.exception("Retrieval initialization failed: %s", exc)
            self.clip_model = None
            self.clip_processor = None
            self.clip_tokenizer = None
            self.embeddings = None
            self.index = None
            self.product_ids = None
            self.df_cleaned = None

    def search(
        self,
        query_text: Optional[str],
        query_image: Optional[Image.Image],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        self.initialize()
        if self.index is None or self.product_ids is None or self.clip_model is None or self.clip_processor is None:
            if self.last_error:
                LOGGER.warning("Search skipped: %s", self.last_error)
            return []

        query_embedding = self._build_query_embedding(query_text, query_image)
        if query_embedding is None:
            return []

        distances, indices = self.index.search(np.expand_dims(query_embedding, axis=0), top_k)
        results: List[Tuple[str, float]] = []
        for rank, row_idx in enumerate(indices[0]):
            if row_idx < 0 or row_idx >= len(self.product_ids):
                continue
            product_name = str(self.product_ids[row_idx])
            score = float(distances[0][rank])
            results.append((product_name, score))
        return results

    def get_product_contexts(self, results: Sequence[Tuple[str, float]]) -> List[ProductContext]:
        if self.df_cleaned is None:
            self.initialize()
        if self.df_cleaned is None:
            return []

        contexts: List[ProductContext] = []
        for product_name, _score in results:
            row = self.df_cleaned[self.df_cleaned["Product Name"] == product_name]
            if row.empty:
                continue
            item = row.iloc[0]
            raw_context = str(item.get("product_context", "") or "")
            condensed = raw_context[:800] + ("..." if len(raw_context) > 800 else "")
            contexts.append(
                ProductContext(
                    name=str(item.get("Product Name", "")),
                    brand="",
                    price="N/A",
                    spec=condensed,
                    about=raw_context,
                    usage="",
                    image_url=str(item.get("image_url_cleaned", "")),
                )
            )
        return contexts

    def _build_query_embedding(
        self,
        query_text: Optional[str],
        query_image: Optional[Image.Image],
    ) -> Optional[np.ndarray]:
        if query_text and query_image is not None:
            return self._joint_embedding(query_text, query_image, max_tokens=70)
        if query_text:
            inputs = self.clip_processor(
                text=[query_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            with torch.no_grad():
                text_emb = self.clip_model.get_text_features(**inputs)
            return self._normalize(text_emb[0].cpu().numpy().astype(np.float32))
        if query_image is not None:
            inputs = self.clip_processor(images=query_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_emb = self.clip_model.get_image_features(**inputs)
            return self._normalize(image_emb[0].cpu().numpy().astype(np.float32))
        return None

    def _joint_embedding(self, text: str, image: Image.Image, max_tokens: int = 70) -> Optional[np.ndarray]:
        if self.clip_tokenizer is None:
            return None
        try:
            token_ids = self.clip_tokenizer.encode(text)
            chunks = [token_ids[i : i + max_tokens] for i in range(0, len(token_ids), max_tokens)]
            chunk_vectors: List[torch.Tensor] = []
            for chunk in chunks:
                chunk_text = self.clip_tokenizer.decode(chunk)
                inputs = self.clip_processor(
                    text=[chunk_text],
                    images=image,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                text_emb = outputs.text_embeds[0]
                image_emb = outputs.image_embeds[0]
                chunk_vectors.append(((text_emb + image_emb) / 2).cpu())
            if not chunk_vectors:
                return None
            merged = torch.stack(chunk_vectors).mean(dim=0).numpy().astype(np.float32)
            return self._normalize(merged)
        except Exception as exc:  
            LOGGER.exception("Failed to create joint embedding: %s", exc)
            return None

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors if norm == 0 else vectors / norm
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
