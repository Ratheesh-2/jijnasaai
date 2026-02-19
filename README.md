# JijnasaAI

JijnasaAI is a web-based AI workspace for multi-model chat, document Q&A (RAG), and voice conversations, with real-time cost visibility and local-first storage on the server.  
When deployed (for example, on Railway), anyone with the URL can create and use their own conversations on the shared instance.

---

## Features

- **Multi-model chat** with streaming responses  
  Supports OpenAI (GPT-4o), Anthropic (Claude), and Google (Gemini), selectable per conversation.

- **Document Q&A (RAG)**  
  Upload PDF, TXT, or MD files and ask grounded questions with source-aware answers.

- **Voice input**  
  Record audio in the UI, transcribe it with Whisper, and send as a chat message.

- **Voice output (TTS)**  
  Listen to AI responses using OpenAI text-to-speech.

- **Cost tracking**  
  See per-conversation and global token usage and estimated costs based on configurable pricing.

- **Conversation management**  
  Create, rename, switch, and delete conversations with auto-generated titles.

- **Custom system prompts**  
  Set per-conversation system instructions to steer behavior.

---

## Quick Start

### 1. Prerequisites

- Python 3.12+  
- At least one API key:  
  - OpenAI (required for embeddings, STT, and TTS)  
  - Anthropic (optional)  
  - Google (optional)

### 2. Install dependencies

```bash
git clone https://github.com/Ratheesh-2/jijnasaai.git
cd jijnasaai

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here   # optional
GOOGLE_API_KEY=your-google-key-here      # optional
```

OpenAI is required for embeddings, speech-to-text, and text-to-speech; Anthropic and Google are optional.

### 4. Run the app

**Option A â€” Single command**

```bash
python run.py
```

This starts both the FastAPI backend (port 8000) and Streamlit frontend (port 8501) in one process.

**Option B â€” Separate terminals**

Backend:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend:

```bash
python -m streamlit run frontend/app.py --server.port 8501 --server.address 127.0.0.1
```

### 5. Open the app

Visit:

- http://localhost:8501

When deployed on Railway, use the public URL shown in your Railway service.

---

## Usage

### Chat

1. Select a model from the sidebar dropdown.  
2. Type a message and press Enter.  
3. Watch the response stream token by token.

You can switch models mid-conversation; each message is sent to the currently selected provider.

### Document Q&A (RAG)

1. Upload PDF, TXT, or MD files from the sidebar.  
2. Toggle **"Use documents (RAG)"** to enable retrieval.  
3. Ask questions; answers will be grounded in your uploaded content with references to the most relevant chunks.

Embeddings are stored in a local ChromaDB instance, and reused across sessions for faster retrieval.

### Voice

- Click the microphone button to start recording.  
- Your speech is transcribed with Whisper and added as a message.  
- Toggle **TTS** to hear the assistant's response spoken aloud.

Voice features require OpenAI enabled in your `.env`.

### Cost tracking

The sidebar shows:

- Estimated cost for the current conversation  
- Total cost across all conversations in the current database

Costs are computed from token usage and per-1M-token pricing configured in `config/settings.yaml`.

---

## Configuration

### `config/settings.yaml`

You can customize:

- **Models**: model IDs, max tokens, provider enable/disable.  
- **RAG**: chunk size, overlap, top-k retrieval count, similarity threshold.  
- **Voice**: STT/TTS models and voice choices.  
- **Pricing**: per-1M-token costs used for cost estimation.

### Data storage

Server-side data is stored locally in the container or host filesystem:

- `data/perplexity.db` â€” SQLite database for conversations, messages, and cost logs.  
- `data/vectorstore/` â€” ChromaDB vector store for document embeddings.  
- `data/uploads/` â€” temporary file uploads (cleaned after ingestion where appropriate).

If you run a shared instance (e.g., Railway), all users of that URL share this backend state.

---

## Architecture

```text
FastAPI Backend (port 8000)         Streamlit Frontend (port 8501)
â”œâ”€â”€ /chat/completions (SSE)    <--> Chat UI with streaming responses
â”œâ”€â”€ /conversations (CRUD)      <--> Conversation sidebar
â”œâ”€â”€ /documents/upload          <--> File uploader
â”œâ”€â”€ /voice/transcribe          <--> Mic input handling
â”œâ”€â”€ /voice/synthesize          <--> TTS playback
â””â”€â”€ /costs/summary             <--> Cost display in sidebar
```

- **Backend**: FastAPI with Server-Sent Events (`sse-starlette`) for streaming.  
- **Frontend**: Streamlit app for chat, RAG, and voice controls.

---

## Running tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Tech stack

- **Frontend**: Streamlit  
- **Backend**: FastAPI + SSE (`sse-starlette`)  
- **LLM providers**: OpenAI, Anthropic, Google GenAI SDKs  
- **Embeddings**: OpenAI `text-embedding-3-small`  
- **Vector store**: ChromaDB (local persistent)  
- **Database**: SQLite (`aiosqlite`)  
- **Voice**: OpenAI Whisper (STT) + OpenAI TTS

---

## Roadmap (ideas)

- User accounts / authentication for multi-tenant setups  
- Persistent per-user workspaces when running a shared deployment  
- Advanced workflows and agent-like multi-step tools on top of the current APIs
