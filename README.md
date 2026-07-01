# TradeScope-AI — Trade Assistant RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot for trade and finance education. It uses a **website** (not PDFs) as its knowledge base, with a Flask web UI similar to MedScope-style chatbots.

## Features

- Website crawling with `robots.txt` respect, depth/page limits, and polite delays
- Text cleaning, chunking, and metadata (URL, title, section)
- Hugging Face `sentence-transformers` embeddings
- **FAISS** (local) or **Pinecone** (cloud) vector store — toggle via config
- Pluggable LLM providers: **OpenAI**, **Anthropic**, or **local** (Ollama/LM Studio)
- Modern Bootstrap chat UI with source citations
- CLI index builder and protected rebuild API endpoint

## Project structure

```
trade-chat-bot/
├── app/
│   ├── config.py
│   ├── routes.py
│   ├── rag/
│   │   ├── ingestion.py
│   │   ├── preprocess.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retrieval.py
│   │   ├── llm.py
│   │   └── pipeline.py
│   └── utils/
├── templates/
├── static/
├── data/raw/ & data/processed/
├── vector_store/
├── scripts/build_index.py
├── tests/
├── run.py
└── requirements.txt
```

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run will download the embedding model (~90 MB for `all-MiniLM-L6-v2`).

### 3. Configure environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` and set at minimum:

```env
API_KEY=your-openai-or-anthropic-key
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini
TARGET_WEBSITE_BASE_URL=https://www.investopedia.com
VECTOR_STORE_TYPE=faiss
```

### 4. Build the knowledge index

This crawls the target website, chunks content, embeds it, and saves a FAISS index:

```bash
python scripts/build_index.py
```

Options:

```bash
python scripts/build_index.py --max-pages 20 --max-depth 1
python scripts/build_index.py --from-chunks   # re-index existing chunks only
```

### 5. Run the Flask app

```bash
python run.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

Alternatively:

```bash
set FLASK_APP=run.py
flask run
```

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_WEBSITE_BASE_URL` | investopedia.com | Website to crawl |
| `MAX_PAGES` | 50 | Max pages to index |
| `MAX_DEPTH` | 2 | Link crawl depth |
| `VECTOR_STORE_TYPE` | `faiss` | `faiss` or `pinecone` |
| `EMBEDDING_MODEL_NAME` | all-MiniLM-L6-v2 | Hugging Face model |
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, `local` |
| `LLM_MODEL_NAME` | gpt-4o-mini | Model name for provider |
| `API_KEY` | — | API key for cloud LLMs |
| `LLM_BASE_URL` | — | Base URL for local LLM |
| `TOP_K` | 5 | Chunks retrieved per query |
| `REBUILD_API_KEY` | — | Key for `/api/rebuild-index` |

## Swapping the knowledge base website

1. Set `TARGET_WEBSITE_BASE_URL` in `.env` to your new site.
2. Adjust `MAX_PAGES` / `MAX_DEPTH` as needed.
3. Run `python scripts/build_index.py`.

## Switching vector store (FAISS ↔ Pinecone)

**FAISS (default, local):**

```env
VECTOR_STORE_TYPE=faiss
```

**Pinecone:**

```env
VECTOR_STORE_TYPE=pinecone
PINECONE_API_KEY=your-key
PINECONE_INDEX_NAME=trade-assistant
```

Create a Pinecone index with dimension **384** (for `all-MiniLM-L6-v2`) and cosine similarity before building.

## Switching LLM provider

**OpenAI:**

```env
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini
API_KEY=sk-...
```

**Anthropic:**

```env
LLM_PROVIDER=anthropic
LLM_MODEL_NAME=claude-3-5-haiku-latest
API_KEY=sk-ant-...
```

**Local (Ollama example):**

```env
LLM_PROVIDER=local
LLM_MODEL_NAME=llama3
LLM_BASE_URL=http://localhost:11434/v1
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI |
| `GET` | `/about` | About page |
| `POST` | `/api/chat` | `{"message": "..."}` → answer + sources |
| `POST` | `/api/rebuild-index` | Re-crawl & rebuild (requires `X-API-Key`) |
| `GET` | `/api/health` | Health check |

### Chat example

```bash
curl -X POST http://localhost:5000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"What is a stop-loss order?\"}"
```

### Rebuild index example

```bash
curl -X POST http://localhost:5000/api/rebuild-index ^
  -H "X-API-Key: your-rebuild-secret"
```

## Running tests

```bash
pytest tests/ -v
```

## Disclaimer

This project is for **educational purposes only**. It does **not** provide financial advice. Always consult qualified professionals before making investment decisions.

## License

MIT (or your preferred license)
