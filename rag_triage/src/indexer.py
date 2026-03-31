"""Build and save vector index from SYNAPSE cases."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import INDEX_DIR
from .data_loader import load_synapse_cases
from .embeddings import Embedder


def build_index(max_samples: int | None = None, index_dir: Path | None = None) -> Path:
    """
    Load SYNAPSE, embed texts, save index and metadata.
    Returns path to index directory.
    """
    index_dir = index_dir or INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    cases = load_synapse_cases(max_samples=max_samples)
    if not cases:
        raise ValueError("No cases loaded from SYNAPSE. Check SYNAPSE_DATA_PATH.")

    texts = [c["text"] for c in cases]
    embedder = Embedder()
    embeddings = embedder.embed(texts)

    np.save(index_dir / "embeddings.npy", embeddings.astype(np.float32))
    with open(index_dir / "cases.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=0, ensure_ascii=False)

    with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "count": len(cases),
            "dim": int(embeddings.shape[1]),
            "model": embedder._model_name,
        }, f, indent=2)

    return index_dir
