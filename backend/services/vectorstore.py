import chromadb
from backend.config import Settings


class VectorStoreManager:
    """Manages ChromaDB PersistentClient — singleton pattern."""

    _instance: "VectorStoreManager | None" = None

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
        )
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def get_instance(cls, settings: Settings) -> "VectorStoreManager":
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    @property
    def collection(self):
        return self._collection

    def add_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def get_document_count(self) -> int:
        return self._collection.count()

    def delete_by_document_id(self, document_id: str):
        """Delete all chunks belonging to a document."""
        self._collection.delete(where={"document_id": document_id})
