"""Retrieve top-k similar cases from the vector index."""

from pathlib import Path
from typing import Any

import numpy as np

from config import INDEX_DIR, TOP_K
from .embeddings import Embedder


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a (1, dim) and b (n, dim). Returns (n,)."""
    a_flat = a.reshape(-1)
    a_norm = a_flat / (np.linalg.norm(a_flat) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.dot(b_norm, a_norm)


class Retriever:
    """Load index and retrieve similar cases."""

    def __init__(self, index_dir: Path | None = None):
        self._index_dir = index_dir or INDEX_DIR
        self._embeddings: np.ndarray | None = None
        self._cases: list[dict[str, Any]] | None = None
        self._embedder: Embedder | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        emb_path = self._index_dir / "embeddings.npy"
        cases_path = self._index_dir / "cases.json"
        if not emb_path.exists() or not cases_path.exists():
            raise FileNotFoundError(
                f"Index not found at {self._index_dir}. Run: python scripts/build_index.py"
            )
        import json
        self._embeddings = np.load(emb_path)
        with open(cases_path, encoding="utf-8") as f:
            self._cases = json.load(f)
        self._embedder = Embedder()
        self._loaded = True

    def retrieve(
        self,
        query_text: str,
        top_k: int | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Retrieve top-k most similar cases.
        Returns list of (case_dict, similarity_score).
        """
        self._load()
        k = top_k or TOP_K
        k = min(k, len(self._cases))

        q_emb = self._embedder.embed_single(query_text)
        sims = _cosine_similarity(q_emb.reshape(1, -1), self._embeddings)
        sims = np.atleast_1d(np.asarray(sims).ravel())
        # [0.91, 0.33, 0.77, 0.12, ...]
        # This is the similarity score for each case.
        # The higher the score, the more similar the case is to the query.
        top_indices = np.argsort(sims)[::-1][:k]

        return [(self._cases[int(i)], float(sims[int(i)])) for i in top_indices]
