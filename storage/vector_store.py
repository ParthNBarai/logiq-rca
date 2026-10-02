"""ChromaDB vector store with strict per-tenant collection isolation.

Each tenant gets its own named collection (``logs_tenant_x``) so semantic
search can never leak documents across tenant boundaries.

Embedding model: ``all-MiniLM-L6-v2`` via sentence-transformers — runs
locally, no API key required, ~80 MB download on first use.
"""

import logging
from typing import Any, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import settings
from storage.models import LogRecord

logger = logging.getLogger(__name__)

# ChromaDB metadata values must be str | int | float | bool — never None.
_NULL_INT_SENTINEL = -1
_NULL_STR_SENTINEL = ""


class VectorStore:
    """Manages a ChromaDB persistent client and per-tenant log collections.

    Thread-safety: ChromaDB's ``PersistentClient`` is thread-safe. The
    ``_collections`` cache is populated lazily; concurrent first-access for
    the same tenant is harmless because ``get_or_create_collection`` is
    idempotent.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collections: dict[str, chromadb.Collection] = {}
        logger.info(
            "VectorStore initialised. Persist dir: %s", settings.CHROMA_PERSIST_DIR
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def upsert_logs(
        self, tenant_id: str, log_records: list[LogRecord]
    ) -> None:
        """Insert or update *log_records* into the tenant-scoped collection.

        ``ingestion_id`` is used as the ChromaDB document id so re-upserts
        for the same record are idempotent.

        Document text format:
            ``<ISO-timestamp> <LEVEL> <Component>: <message>``
        """
        if not log_records:
            return

        collection = self._get_collection(tenant_id)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for record in log_records:
            ids.append(record.ingestion_id)
            documents.append(
                f"{record.timestamp.isoformat()} {record.level} "
                f"{record.component}: {record.message}"
            )
            metadatas.append(self._build_metadata(record))

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.debug(
            "Upserted %d documents into collection for tenant %s",
            len(ids),
            tenant_id,
        )

    def search(
        self,
        tenant_id: str,
        query: str,
        n_results: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Run a semantic search scoped to *tenant_id*'s collection only.

        Args:
            tenant_id:       Tenant whose collection to search.
            query:           Natural-language query string.
            n_results:       Maximum number of results to return.
            filter_metadata: Optional ChromaDB ``where`` clause for
                             pre-filtering by metadata fields, e.g.
                             ``{"is_error": True}`` or
                             ``{"level": {"$eq": "ERROR"}}``.

        Returns:
            List of dicts with keys ``document``, ``metadata``, ``distance``.
            Ordered by ascending cosine distance (most similar first).
        """
        collection = self._get_collection(tenant_id)

        where = filter_metadata if filter_metadata else None

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        # ingestion_id is the ChromaDB document id — surface it explicitly so
        # the retriever can deduplicate against structural hits by the same key.
        return [
            {
                "ingestion_id": id_,
                "document": doc,
                "metadata": meta,
                "distance": dist,
            }
            for id_, doc, meta, dist in zip(ids, docs, metas, distances)
        ]

    def collection_count(self, tenant_id: str) -> int:
        """Return the number of documents in *tenant_id*'s collection."""
        return self._get_collection(tenant_id).count()

    def list_tenants(self) -> list[str]:
        """Return all tenant IDs that have a collection in this store."""
        collections = self._client.list_collections()
        tenants: list[str] = []
        for col in collections:
            name: str = col.name if hasattr(col, "name") else str(col)
            if name.startswith("logs_"):
                tenants.append(name[len("logs_"):].upper().replace("_", "-"))
        return tenants

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_collection(self, tenant_id: str) -> chromadb.Collection:
        """Return (or lazily create) the ChromaDB collection for *tenant_id*."""
        if tenant_id not in self._collections:
            collection_name = (
                "logs_" + tenant_id.lower().replace("-", "_").replace(" ", "_")
            )
            self._collections[tenant_id] = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.debug("Opened/created ChromaDB collection %r", collection_name)
        return self._collections[tenant_id]

    @staticmethod
    def _build_metadata(record: LogRecord) -> dict[str, Any]:
        """Convert a ``LogRecord`` to a flat ChromaDB-compatible metadata dict.

        ChromaDB only accepts ``str | int | float | bool``; None values are
        replaced with typed sentinels so filter queries remain well-typed.
        """
        return {
            "tenant_id": record.tenant_id,
            "level": record.level,
            "component": record.component,
            "logger_class": record.logger_class,
            # Full message text stored in metadata so retriever can reconstruct
            # evidence dicts from semantic hits without a secondary DB lookup.
            "message": record.message,
            "timestamp_unix": int(record.timestamp.timestamp()),
            "is_error": record.is_error,
            "http_status": record.http_status
            if record.http_status is not None
            else _NULL_INT_SENTINEL,
            "latency_ms": record.latency_ms
            if record.latency_ms is not None
            else _NULL_INT_SENTINEL,
            "db_cluster": record.db_cluster or _NULL_STR_SENTINEL,
            # circuit_breaker_state needed by the retriever's RCA scoring.
            "circuit_breaker_state": record.circuit_breaker_state or _NULL_STR_SENTINEL,
            "exception_type": record.exception_type or _NULL_STR_SENTINEL,
            "file_source": record.file_source,
        }
