# Multimodal Ecommerce RAG (Flask + CLIP + FAISS + Ollama - qwen2.5:7b)

## Project Overview

This repository contains a Flask-based multimodal retrieval-augmented product QA system.
It supports:

- text query (`user_query`)
- image query (uploaded product image)
- mixed query (text + image)

Core stack:

- Flask for backend and web UI
- CLIP (`openai/clip-vit-base-patch32`) for multimodal embeddings
- FAISS for nearest-neighbor retrieval
- Ollama + Qwen (`qwen2.5:7b` by default) for intent routing and final answer generation

## Core Pipeline

Primary runtime chain:

`python app.py` -> Flask route -> `AnswerPipeline.answer_user_query` -> `QueryRouter.route` (intent + retrieval query) -> `RetrievalService.search` -> `RetrievalService.get_product_contexts` -> `OllamaClient.chat` (final grounded answer)

Three stages:

1. **Intent understanding (first LLM call)**
  `QueryRouter` classifies query as `text`, `image`, or `image_request`, and optionally outputs a retrieval query.
2. **Retrieval and reranking (RAG)**
  `RetrievalService` builds CLIP embeddings (text, image, or mixed), chunks long text into <=70 token windows in mixed mode, normalizes vectors, and searches FAISS top-k candidates.
3. **Answer generation (second LLM call)**
  `AnswerPipeline` assembles structured product context and prompts Ollama for grounded response generation.

## Repository Structure

```text
.
├── app.py
├── config.json
├── configs/
│   └── config.example.json
├── data/
│   └── preprocessed_data.csv
├── scripts/
│   └── smoke_test.py
├── src/
│   └── ecommerce_rag/
│       ├── app/
│       │   └── factory.py
│       ├── config/
│       │   └── settings.py
│       ├── llm/
│       │   ├── ollama_client.py
│       │   └── router.py
│       ├── pipeline/
│       │   └── answer_pipeline.py
│       └── retrieval/
│           └── service.py
├── templates/
│   └── index.html
├── tests/
│   └── test_smoke.py
└── requirements.txt
```

## Installation

1. Use Python 3.10+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Ensure Ollama is installed and the configured model exists:

```bash
ollama pull qwen2.5:7b
ollama serve
```

1. Prepare retrieval artifacts:

- product CSV (`data/preprocessed_data.csv`)
- CLIP embedding matrix (`data/clip_embeddings.npy`)
- product id mapping (`data/product_ids.pkl`)

## Configuration

Runtime settings are loaded from `config.json`. You can copy from `configs/config.example.json`.

Important keys:

- `llm_model`: Ollama model name (default `qwen2.5:7b`)
- `ollama_host`: Ollama server URL
- `retrieval_top_k`: number of retrieved products
- `clip_model_name`: CLIP checkpoint name
- `data_csv_path`: catalog CSV path
- `embeddings_path`: CLIP embedding `.npy` path
- `product_ids_path`: product id `.pkl` path
- `flask_host`, `flask_port`, `flask_debug`: Flask startup settings

## Usage

Run web app:

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

Run smoke test:

```bash
python scripts/smoke_test.py
```

Run unit smoke tests:

```bash
pytest
```

## Limitations

- End-to-end answer quality depends on local Ollama runtime and Qwen model availability.
- Retrieval requires local FAISS artifacts (`.npy`, `.pkl`) aligned with CSV row order.
- CLIP model loading can be slow on CPU-only machines.
- This project has not yet been developed to a production-hardened ecommerce platform.

## Acknowledgements

- This project uses third-party models and libraries, including CLIP (OpenAI), FAISS (Meta), and Ollama.
- Qwen model weights are served locally through Ollama and remain subject to their original upstream terms.

