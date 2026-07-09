# RAG Knowledge Chatbot - Backend

Retrieval-Augmented Generation backend built with FastAPI.

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy and fill in)
cp .env.example .env

# Run development server
python run.py
```

Server starts at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

## Tech Stack

- **FastAPI** - Web framework
- **ChromaDB** - Vector database
- **Sentence Transformers** - Embeddings
- **OpenAI / Local LLM** - LLM inference
