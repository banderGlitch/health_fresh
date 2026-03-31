"""RAG Triage configuration."""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DATA_ROOT = _ROOT.parent / "data"

SYNAPSE_DATA_PATH = os.getenv(
    "SYNAPSE_DATA_PATH",
    str(_DATA_ROOT / "SYNAPSE_An Expert Annotated Dataset of Patient symptoms and Demographics.csv"),
)

INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", str(_ROOT / "index")))
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RAG_TOP_K", "20"))
MAX_SAMPLES = int(os.getenv("RAG_MAX_SAMPLES", "130637"))  # Full dataset ~130k
