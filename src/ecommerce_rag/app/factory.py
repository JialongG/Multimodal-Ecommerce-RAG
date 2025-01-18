from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import markdown
from flask import Flask, render_template, request
from markupsafe import Markup
from PIL import Image

from ..config import AppConfig, load_config
from ..pipeline import AnswerPipeline


LOGGER = logging.getLogger(__name__)


def create_app(config_path: str | None = None) -> Flask:
    config = load_config(config_path)
    pipeline = AnswerPipeline(config)

    root_dir = Path(__file__).resolve().parents[3]
    app = Flask(__name__, template_folder=str(root_dir / "templates"), static_folder=str(root_dir / "static"))

    @app.route("/", methods=["GET", "POST"])
    def index():
        response_html: Optional[Markup] = None
        uploaded_image: Optional[Image.Image] = None
        image_preview: Optional[str] = None
        user_query = ""

        if request.method == "POST":
            user_query = request.form.get("user_query", "").strip()
            image_file = request.files.get("image")
            if image_file and image_file.filename:
                try:
                    uploaded_image = Image.open(image_file.stream).convert("RGB")
                    image_preview = pil_to_base64(uploaded_image)
                except Exception as exc:
                    LOGGER.warning("Failed to load uploaded image: %s", exc)
                    response_html = Markup("Uploaded file is not a valid image.")

            if response_html is None and (user_query or uploaded_image is not None):
                raw_response = pipeline.answer_user_query(query_text=user_query, query_image=uploaded_image)
                response_html = Markup(markdown.markdown(raw_response))

        return render_template(
            "index.html",
            response=response_html,
            image_preview=image_preview,
            user_query=user_query,
        )

    app.config["APP_CONFIG"] = config
    app.config["PIPELINE"] = pipeline
    return app


def pil_to_base64(pil_img: Image.Image) -> str:
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def run_app(config: AppConfig) -> None:
    app = create_app()
    app.run(host=config.flask_host, port=config.flask_port, debug=config.flask_debug)
