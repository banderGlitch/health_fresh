"""Embedding model wrapper for RAG."""

# Embedder → tool → text → vector

from typing import Any

import numpy as np

from config import EMBEDDING_MODEL


class Embedder:
    """Sentence-transformers based embedder."""

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or EMBEDDING_MODEL
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (n, dim) array."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
#             "fever male 25"
#           → [0.12, -0.44, 0.88, ...]
# This vector is used to search for similar cases.
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns (dim,) array."""
        return self.embed([text])[0]
