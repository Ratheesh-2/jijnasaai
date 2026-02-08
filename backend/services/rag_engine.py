import logging
import uuid
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader

from backend.config import Settings
from backend.services.vectorstore import VectorStoreManager
from backend.database import get_db

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, settings: Settings, vs_manager: VectorStoreManager):
        self._settings = settings
        self._vs = vs_manager
        self._openai = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        self._embedding_model = settings.embedding_config.get("model", "text-embedding-3-small")
        self._chunk_size = settings.rag_config.get("chunk_size", 1000)
        self._chunk_overlap = settings.rag_config.get("chunk_overlap", 200)
        self._retrieval_k = settings.rag_config.get("retrieval_k", 5)

    # --- Embedding ---
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via OpenAI embedding API."""
        # OpenAI embeddings API has a limit of ~8000 tokens per input;
        # chunk into batches of 100 texts at a time
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._openai.embeddings.create(
                model=self._embedding_model,
                input=batch,
            )
            all_embeddings.extend(item.embedding for item in response.data)
        return all_embeddings

    def _embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    # --- Document Loading ---
    def _load_file(self, file_path: Path) -> str:
        """Load text from PDF, TXT, or MD files."""
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(str(file_path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
        elif ext in (".txt", ".md"):
            return file_path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # --- Chunking ---
    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks using recursive character splitting."""
        if not text.strip():
            return []

        separators = ["\n\n", "\n", ". ", " "]

        def _split(text: str, sep_idx: int = 0) -> list[str]:
            if len(text) <= self._chunk_size:
                return [text.strip()] if text.strip() else []

            if sep_idx >= len(separators):
                # Force split at chunk_size
                result = []
                for i in range(0, len(text), self._chunk_size - self._chunk_overlap):
                    chunk = text[i:i + self._chunk_size]
                    if chunk.strip():
                        result.append(chunk.strip())
                return result

            sep = separators[sep_idx]
            parts = text.split(sep)
            result = []
            current = ""

            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= self._chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        result.append(current.strip())
                    if len(part) > self._chunk_size:
                        result.extend(_split(part, sep_idx + 1))
                        current = ""
                    else:
                        current = part

            if current.strip():
                result.append(current.strip())
            return result

        raw_chunks = _split(text)

        # Add overlap by prepending tail of previous chunk
        if self._chunk_overlap > 0 and len(raw_chunks) > 1:
            overlapped = [raw_chunks[0]]
            for i in range(1, len(raw_chunks)):
                prev_tail = raw_chunks[i - 1][-self._chunk_overlap:]
                overlapped.append(prev_tail + " " + raw_chunks[i])
            return overlapped

        return raw_chunks

    # --- Ingest ---
    async def ingest_document(
        self, file_path: Path, original_filename: str,
        file_size: int = 0, conversation_id: str | None = None,
    ) -> dict:
        """Full pipeline: load -> chunk -> embed -> store."""
        text = self._load_file(file_path)
        chunks = self._chunk_text(text)

        if not chunks:
            raise ValueError("No text content found in file")

        doc_id = str(uuid.uuid4())

        # Embed all chunks in batch
        embeddings = self._embed(chunks)

        # Prepare ChromaDB data
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": doc_id,
                "filename": original_filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "conversation_id": conversation_id or "",
            }
            for i in range(len(chunks))
        ]

        self._vs.add_chunks(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        # Record in SQLite
        async with get_db() as db:
            await db.execute(
                """INSERT INTO documents
                   (id, filename, file_type, file_size, chunk_count, conversation_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (doc_id, original_filename, file_path.suffix.lower(),
                 file_size, len(chunks), conversation_id),
            )
            await db.commit()

        logger.info(f"Ingested {original_filename}: {len(chunks)} chunks, doc_id={doc_id}")

        return {
            "id": doc_id,
            "filename": original_filename,
            "chunk_count": len(chunks),
            "file_size": file_size,
        }

    # --- Retrieval ---
    async def retrieve_context(
        self, query: str, conversation_id: str | None = None,
    ) -> tuple[str, list[dict]]:
        """Retrieve relevant chunks for a query. Returns (context_text, sources)."""
        query_embedding = self._embed_query(query)

        # Search all documents (optionally could filter by conversation_id)
        where_filter = None
        if conversation_id:
            where_filter = {"conversation_id": conversation_id}

        try:
            results = self._vs.query(
                query_embedding=query_embedding,
                n_results=self._retrieval_k,
                where=where_filter,
            )
        except Exception:
            # If no documents match the filter, try without filter
            results = self._vs.query(
                query_embedding=query_embedding,
                n_results=self._retrieval_k,
            )

        if not results["documents"] or not results["documents"][0]:
            return "", []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if results.get("distances") else []

        context_parts = []
        sources = []
        threshold = self._settings.rag_config.get("similarity_threshold", 0.3)

        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            distance = distances[i] if i < len(distances) else 1.0
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            similarity = 1.0 - distance
            if similarity < threshold:
                continue

            context_parts.append(
                f"[Source: {meta.get('filename', 'Unknown')}, "
                f"Chunk {meta.get('chunk_index', '?')}]\n{doc}"
            )
            sources.append({
                "filename": meta.get("filename", "Unknown"),
                "chunk_index": meta.get("chunk_index"),
                "content_preview": doc[:200],
                "similarity": round(similarity, 3),
            })

        context = "\n\n---\n\n".join(context_parts)
        return context, sources
