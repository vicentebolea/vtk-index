"""Public query API for vtk-index.

Connects to a running Qdrant instance with a snapshot loaded and exposes
hybrid retrieval (dense + BM25) with RRF fusion.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

from ..chunking.base import Chunk
from ..embedding.dense import DenseEmbedder
from ..embedding.sparse import SparseEmbedder
from .filters import PayloadFilter, build_filter


class Retriever:
    """Public query API over a Qdrant instance with a vtk-index snapshot loaded.

    Performance target: <50 ms per query end-to-end on a workstation with an
    in-memory Qdrant instance.

    Args:
        qdrant_url: URL of the running Qdrant service.
        vtk_version: VTK version tag to validate snapshot alignment.
        dense_model: sentence-transformers model name.
        sparse_model: FastEmbed model name.
    """

    DOCS_COLLECTION = "vtk_docs"
    CODE_COLLECTION = "vtk_code"

    def __init__(
        self,
        qdrant_url: str = ":memory:",
        qdrant_path: str | None = None,
        vtk_version: str = "",
        dense_model: str = DenseEmbedder.DEFAULT_MODEL,
        sparse_model: str = SparseEmbedder.DEFAULT_MODEL,
    ) -> None:
        self.client = (
            QdrantClient(path=qdrant_path) if qdrant_path is not None else QdrantClient(qdrant_url)
        )
        self.vtk_version = vtk_version
        self.dense = DenseEmbedder(dense_model)
        self.sparse = SparseEmbedder(sparse_model)

    @classmethod
    def from_artifact(
        cls,
        vtk_version: str,
        repository: str = "vicentebolea/vtk-index",
        dense_model: str = DenseEmbedder.DEFAULT_MODEL,
        sparse_model: str = SparseEmbedder.DEFAULT_MODEL,
    ) -> Retriever:
        """Return a Retriever backed by the pre-built embedded storage for *vtk_version*.

        Downloads from ``ghcr.io/{repository}:{vtk_version}-embedded`` on first call;
        subsequent calls return immediately from the local cache at
        ``~/.cache/vtk-index/storage-{vtk_version}/``.

        No Qdrant server or embedding step required.

        Example::

            retriever = Retriever.from_artifact("9.6.1")
            chunks = retriever.search_docs("sphere source")
        """
        from ..artifact.fetcher import fetch_embedded_storage

        storage_path = fetch_embedded_storage(vtk_version, repository=repository)
        return cls(
            qdrant_path=str(storage_path),
            vtk_version=vtk_version,
            dense_model=dense_model,
            sparse_model=sparse_model,
        )

    def search_docs(
        self,
        query: str,
        k: int = 10,
        filters: PayloadFilter | dict[str, Any] | Filter | None = None,
    ) -> list[Chunk]:
        return self.hybrid_search(query, self.DOCS_COLLECTION, k=k, filters=filters)

    def search_code(
        self,
        query: str,
        k: int = 10,
        filters: PayloadFilter | dict[str, Any] | Filter | None = None,
    ) -> list[Chunk]:
        return self.hybrid_search(query, self.CODE_COLLECTION, k=k, filters=filters)

    def hybrid_search(
        self,
        query: str,
        collection: str,
        k: int = 10,
        alpha: float = 0.5,
        filters: PayloadFilter | dict[str, Any] | Filter | None = None,
        prefetch_limit: int = 20,
    ) -> list[Chunk]:
        """Reciprocal Rank Fusion of dense and sparse results."""
        dense_vec = self.dense.encode(query)
        sparse_emb = self.sparse.embed(query)
        sparse_vec = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )

        f = _to_filter(filters)

        results = self.client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dense_vec, using="content", limit=prefetch_limit, filter=f),
                Prefetch(query=sparse_vec, using="bm25", limit=prefetch_limit, filter=f),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
        )
        return [_to_chunk(p) for p in results.points]


def _to_filter(f) -> Filter | None:
    if isinstance(f, PayloadFilter):
        return f.build()
    return build_filter(f)


def _to_chunk(point) -> Chunk:
    from ..chunking.base import ChunkType

    payload = point.payload or {}
    try:
        ct = ChunkType(payload.get("chunk_type", "class_overview"))
    except ValueError:
        ct = ChunkType.CLASS_OVERVIEW
    return Chunk(
        chunk_id=str(point.id),
        chunk_type=ct,
        content=payload.get("content", ""),
        class_names=payload.get("class_names", []),
        module_names=payload.get("module_names", []),
        role=payload.get("role"),
        input_datatype=payload.get("input_datatype"),
        output_datatype=payload.get("output_datatype"),
        visibility_score=payload.get("visibility_score"),
        source=payload.get("source", ""),
        source_path=payload.get("source_path", ""),
        vtk_version=payload.get("vtk_version", ""),
    )
