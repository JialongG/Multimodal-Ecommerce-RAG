from __future__ import annotations

from src.ecommerce_rag.app import create_app
from src.ecommerce_rag.config import load_config
from src.ecommerce_rag.pipeline import AnswerPipeline


def test_config_loads() -> None:
    config = load_config()
    assert config.llm_model
    assert config.retrieval_top_k > 0


def test_pipeline_constructs() -> None:
    pipeline = AnswerPipeline(load_config())
    assert pipeline.router is not None
    assert pipeline.retrieval is not None


def test_flask_app_creation() -> None:
    app = create_app()
    assert app is not None
