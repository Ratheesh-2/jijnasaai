# Perplexity Clone

A single-user, Perplexity-style AI chat application with multi-model support, RAG (document Q&A), and voice input/output.

## Features

- **Multi-model chat** with streaming responses — OpenAI (GPT-4o), Anthropic (Claude), Google (Gemini)
- **RAG (Retrieval-Augmented Generation)** — upload PDF, TXT, or MD files and ask questions grounded in them
- **Voice input** — record audio, transcribe via Whisper, and chat hands-free
- **Voice output** — TTS playback of assistant responses
- **Cost tracking** — per-conversation and global token usage and cost estimates
- **Conversation management** — create, switch, delete conversations with auto-generated titles
- **Custom system prompts** per conversation

## Quick Start

### 1. Prerequisites

- Python 3.12+
- At least one API key: OpenAI (required for embeddings/voice), Anthropic, or Google

### 2. Install dependencies

```bash
cd "single-user Perplexity clone"
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here    # optional
GOOGLE_API_KEY=your-google-key-here        # optional
```

OpenAI is required (used for embeddings, STT, and TTS). Anthropic and Google are optional — models for unconfigured providers won't work but the app will still run.

### 4. Run the app

**Option A — Single command:**

```bash
python run.py
```

This starts both the backend (port 8000) and frontend (port 8501).

**Option B — Separate terminals:**

Terminal 1 (backend):
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2 (frontend):
```bash
python -m streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1
```

### 5. Open the app

Navigate to **http://localhost:8501** in your browser.

## Usage

### Chat
1. Select a model from the sidebar dropdown
2. Type a message and press Enter
3. Watch the streaming response appear

### RAG (Document Q&A)
1. Upload PDF, TXT, or MD files using the sidebar uploader
2. Toggle **"Use documents (RAG)"** on
3. Ask questions — answers will be grounded in your uploaded documents with source citations

### Voice
1. Click the microphone input to record audio
2. Your speech is transcribed and sent as a chat message
3. Toggle **TTS** to hear responses read aloud

### Cost Tracking
The sidebar shows per-conversation and total costs based on token usage and the pricing configured in `config/settings.yaml`.

## Configuration

### `config/settings.yaml`

- **Models**: Add or remove models, set max tokens
- **RAG parameters**: Chunk size, overlap, retrieval count, similarity threshold
- **Voice**: STT/TTS model and voice selection
- **Pricing**: Per-1M-token costs for cost estimation

### Data Storage

All data is stored locally:
- `data/perplexity.db` — SQLite database (conversations, messages, cost logs)
- `data/vectorstore/` — ChromaDB embeddings
- `data/uploads/` — temporary file uploads (cleaned after ingestion)

## Architecture

```
FastAPI Backend (port 8000)          Streamlit Frontend (port 8501)
├── /chat/completions (SSE)    <-->  Chat UI with streaming
├── /conversations (CRUD)      <-->  Conversation sidebar
├── /documents/upload          <-->  File uploader
├── /voice/transcribe          <-->  Mic input
├── /voice/synthesize          <-->  TTS playback
└── /costs/summary             <-->  Cost display
```

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI + SSE (sse-starlette)
- **LLM Providers**: OpenAI, Anthropic, Google GenAI SDKs
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Store**: ChromaDB (local persistent)
- **Database**: SQLite (aiosqlite)
- **Voice**: OpenAI Whisper (STT) + OpenAI TTS
