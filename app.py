from __future__ import annotations

import logging

from src.ecommerce_rag.app import create_app
from src.ecommerce_rag.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

config = load_config()
app = create_app()

if __name__ == "__main__":
    app.run(host=config.flask_host, port=config.flask_port, debug=config.flask_debug)
