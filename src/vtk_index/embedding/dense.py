"""sentence-transformers dense embedding wrapper."""

from __future__ import annotations


class DenseEmbedder:
    """Thin wrapper around SentenceTransformer.

    Loads the model once; call ``encode(text)`` to get a dense vector.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.encode(texts)]
