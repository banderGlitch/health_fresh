"""One-off: test MongoDB URI from .env (does not print secrets)."""
from __future__ import annotations

import sys
from pathlib import Path

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
    except ImportError:
        pass

    import os

    uri = (os.getenv("MONGODB_URI") or os.getenv("mongodb_uri") or "").strip()
    if not uri:
        print("RESULT: FAIL — set MONGODB_URI or mongodb_uri in .env")
        return 1
    if "<db_password>" in uri or "<password>" in uri.lower():
        print("RESULT: FAIL — URI still contains a placeholder")
        return 1

    try:
        from pymongo import MongoClient
    except ModuleNotFoundError:
        print("RESULT: FAIL — install pymongo: pip install pymongo")
        return 1

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=15000)
        client.admin.command("ping")
        print("RESULT: OK — MongoDB Atlas responded to ping")
        client.close()
        return 0
    except Exception as e:
        print("RESULT: FAIL —", type(e).__name__ + ":", str(e)[:400])
        return 1


if __name__ == "__main__":
    sys.exit(main())
