"""
FARSIX ChromaDB Vector Memory Store

Stores completed mission results as embeddings for semantic similarity lookup.
Before each new mission, the memory is queried for similar past results.
If similarity ≥ 0.85, the past context is injected into the Nemotron prompt.

Uses ChromaDB's default sentence-transformers embedding (CPU-friendly).
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional

# ChromaDB data directory — persisted between runs
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_CHROMA_PATH = os.path.join(_PROJECT_ROOT, "chroma_data")
os.makedirs(_CHROMA_PATH, exist_ok=True)

COLLECTION_NAME = "farsix_missions"


class ChromaStore:
    """
    Vector memory store using ChromaDB.

    Usage:
        store = ChromaStore()
        store.add("factory floor analysis...", {"mission_id": "m-001"})
        results = store.search("motor bearing temperature", n_results=3)
    """

    def __init__(self):
        self._client = None
        self._collection = None
        self._available = False
        self._error: Optional[str] = None
        self._init()

    def _init(self) -> None:
        """Initialise ChromaDB client and collection. Fails gracefully."""
        try:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=_CHROMA_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Use default embedding function (uses local sentence-transformers)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True

        except Exception as exc:
            self._available = False
            self._error = str(exc)

    # ------------------------------------------------------------------ #
    #  Write                                                                #
    # ------------------------------------------------------------------ #

    def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Embed and store a content string.

        Args:
            content:  Text to embed (mission input + output concatenated).
            metadata: Optional key-value metadata stored alongside embedding.

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self._available or self._collection is None:
            return False

        try:
            doc_id = f"doc-{uuid.uuid4().hex[:12]}"
            meta = metadata or {}
            meta["stored_at"] = str(time.time())
            meta["content_preview"] = content[:200]

            self._collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[meta],
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  Query                                                                #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        n_results: int = 3,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for similar past missions.

        Args:
            query:          Query text to embed and search.
            n_results:      Max number of results to return.
            min_similarity: Filter results below this cosine similarity (0-1).

        Returns:
            List of dicts: [{"content": str, "similarity": float, "metadata": dict}]
        """
        if not self._available or self._collection is None:
            return []

        try:
            count = self._collection.count()
            if count == 0:
                return []

            actual_n = min(n_results, count)
            results = self._collection.query(
                query_texts=[query],
                n_results=actual_n,
                include=["documents", "metadatas", "distances"],
            )

            output = []
            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            for doc, dist, meta in zip(docs, distances, metas):
                # ChromaDB cosine distance: similarity = 1 - distance
                similarity = max(0.0, 1.0 - dist)
                if similarity >= min_similarity:
                    output.append({
                        "content": doc,
                        "similarity": round(similarity, 4),
                        "metadata": meta or {},
                    })

            return output

        except Exception:
            return []

    # ------------------------------------------------------------------ #
    #  Stats                                                                #
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """Return number of stored embeddings."""
        if not self._available or self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    def is_available(self) -> bool:
        return self._available

    def error_message(self) -> Optional[str]:
        return self._error

    def reset(self) -> bool:
        """Clear all stored embeddings (destructive)."""
        if not self._available or self._client is None:
            return False
        try:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            return True
        except Exception:
            return False

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Return metadata for all stored documents (for dashboard display)."""
        if not self._available or self._collection is None:
            return []
        try:
            results = self._collection.get(include=["metadatas"])
            return results.get("metadatas", []) or []
        except Exception:
            return []


# ------------------------------------------------------------------ #
#  Singleton                                                            #
# ------------------------------------------------------------------ #

_chroma_store: Optional[ChromaStore] = None


def get_chroma_store() -> ChromaStore:
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore()
    return _chroma_store
