"""FastEmbed BM25 sparse embedding wrapper."""

from __future__ import annotations


class SparseEmbedder:
    """Thin wrapper around FastEmbed BM25 model."""

    DEFAULT_MODEL = "Qdrant/bm25"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name)
        self.model_name = model_name

    def embed(self, text: str):
        """Return a single SparseEmbedding for *text*."""
        return next(iter(self._model.embed([text])))

    def embed_batch(self, texts: list[str]):
        """Return a list of SparseEmbeddings for *texts*."""
        return list(self._model.embed(texts))
