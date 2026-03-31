"""Build RAG index from SYNAPSE dataset. Run from rag_triage folder."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT.parent / ".env")
except ImportError:
    pass

from src.indexer import build_index

if __name__ == "__main__":
    print("Building RAG index...")
    path = build_index()
    print(f"Done. Index saved to {path}")
    meta = path / "meta.json"
    if meta.exists():
        import json
        with open(meta) as f:
            info = json.load(f)
        print(f"  Cases indexed: {info.get('count', '?')}")
