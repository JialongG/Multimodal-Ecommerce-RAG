from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class AppConfig:
    llm_model: str
    ollama_host: str
    flask_host: str
    flask_port: int
    flask_debug: bool
    retrieval_top_k: int
    clip_model_name: str
    data_csv_path: Path
    embeddings_path: Path
    product_ids_path: Path

    @classmethod
    def from_mapping(cls, root_dir: Path, values: Dict[str, Any]) -> "AppConfig":
        def _resolve(path_value: str, default_rel: str) -> Path:
            raw = str(values.get(path_value, default_rel))
            p = Path(raw)
            return p if p.is_absolute() else (root_dir / p).resolve()

        return cls(
            llm_model=str(values.get("llm_model", "qwen2.5:7b")),
            ollama_host=str(values.get("ollama_host", "http://127.0.0.1:11434")),
            flask_host=str(values.get("flask_host", "127.0.0.1")),
            flask_port=int(values.get("flask_port", 5000)),
            flask_debug=bool(values.get("flask_debug", False)),
            retrieval_top_k=int(values.get("retrieval_top_k", 5)),
            clip_model_name=str(values.get("clip_model_name", "openai/clip-vit-base-patch32")),
            data_csv_path=_resolve("data_csv_path", "data/preprocessed_data.csv"),
            embeddings_path=_resolve("embeddings_path", "data/clip_embeddings.npy"),
            product_ids_path=_resolve("product_ids_path", "data/product_ids.pkl"),
        )


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(config_path: str | Path | None = None) -> AppConfig:
    root = project_root()
    target = Path(config_path).resolve() if config_path else (root / "config.json").resolve()
    with target.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return AppConfig.from_mapping(root, raw)
