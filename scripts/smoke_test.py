from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ecommerce_rag.app import create_app
from src.ecommerce_rag.config import load_config
from src.ecommerce_rag.pipeline import AnswerPipeline


def main() -> int:
    config = load_config()
    pipeline = AnswerPipeline(config)
    app = create_app()

    print("Config loaded:")
    print(f"  llm_model={config.llm_model}")
    print(f"  data_csv_path={config.data_csv_path}")
    print(f"  embeddings_path={config.embeddings_path}")
    print("Initialization checks:")
    print(f"  pipeline_class={pipeline.__class__.__name__}")
    print(f"  flask_app_name={app.name}")
    print("Smoke test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
