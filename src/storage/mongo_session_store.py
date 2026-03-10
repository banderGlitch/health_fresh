"""
Standalone MongoDB session store for triage conversations.

This module is intentionally not wired into the API yet.
Use it later by importing MongoSessionStore in api/main.py.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection


class MongoSessionStore:
    """
    Minimal session persistence layer.

    Session document shape (suggested):
    {
      "session_id": "<uuid>",
      "patient_id": "<optional-patient-id>",
      "status": "collecting|completed|expired",
      "status_reason": "followup_needed|max_rounds_reached|...",
      "round": 1,
      "max_rounds": 4,
      "conversation": "...",
      "demographics": {...},
      "history": {...},
      "clarifying_questions": [...],
      "asked_questions": [...],
      "created_at": datetime,
      "updated_at": datetime,
      "expires_at": datetime
    }
    """

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = "cepialabs_healthcare",
        collection_name: str = "sessions",
    ):
        self.mongo_uri = (
            mongo_uri
            or os.getenv("MONGODB_URI")
            or os.getenv("mongodb_uri")
            or ""
        ).strip()
        if not self.mongo_uri:
            raise ValueError("Missing MongoDB URI. Set MONGODB_URI in .env")
        if "<db_password>" in self.mongo_uri:
            raise ValueError("MongoDB URI still contains <db_password> placeholder")

        self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        self.collection: Collection = self.db[collection_name]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.collection.create_index("session_id", unique=True)
        # Auto-delete expired sessions.
        self.collection.create_index("expires_at", expireAfterSeconds=0)

    def create_session(
        self,
        session_id: str,
        data: dict[str, Any],
        ttl_hours: int = 24,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        doc = {
            "session_id": session_id,
            "patient_id": data.get("patient_id"),
            "status": data.get("status", "collecting"),
            "status_reason": data.get("status_reason", "followup_needed"),
            "round": int(data.get("round", 1)),
            "max_rounds": int(data.get("max_rounds", 4)),
            "conversation": data.get("conversation", ""),
            "demographics": data.get("demographics", {}),
            "history": data.get("history", {}),
            "clarifying_questions": data.get("clarifying_questions", []),
            "asked_questions": data.get("asked_questions", []),
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(hours=ttl_hours),
        }
        self.collection.insert_one(doc)
        return self._clean(doc)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        doc = self.collection.find_one({"session_id": session_id})
        return self._clean(doc) if doc else None

    def update_session(self, session_id: str, updates: dict[str, Any]) -> bool:
        patch = dict(updates or {})
        patch["updated_at"] = datetime.now(timezone.utc)
        result = self.collection.update_one({"session_id": session_id}, {"$set": patch})
        return result.matched_count > 0

    def delete_session(self, session_id: str) -> bool:
        result = self.collection.delete_one({"session_id": session_id})
        return result.deleted_count > 0

    @staticmethod
    def _clean(doc: dict[str, Any] | None) -> dict[str, Any] | None:
        if not doc:
            return None
        out = dict(doc)
        out.pop("_id", None)
        return out

