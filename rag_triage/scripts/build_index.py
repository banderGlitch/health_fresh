"""Build RAG vector index from SYNAPSE dataset."""

import sys
from pathlib import Path

# Ensure rag_triage root is on path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.indexer import build_index
from config import MAX_SAMPLES


def main():
    print("[RAG] Building index from SYNAPSE dataset...")
    path = build_index(max_samples=MAX_SAMPLES)
    print(f"[RAG] Index saved to {path}")
    print("[RAG] Run predictor: from rag_triage.src.triage import RAGTriagePredictor")


if __name__ == "__main__":
    main()
