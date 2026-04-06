"""
Session persistence — save/load research session transcripts.

Stores session transcripts as JSON files for later review.
Mirrors Claude Code's transcript recording in QueryEngine.ts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SESSION_DIR = Path("./sessions")


class SessionStorage:
    """
    Persist research session transcripts to disk.

    Each session is saved as a JSON file containing the session
    summary (query, final answer, citations, plan findings, etc.).
    """

    def __init__(self, base_dir: str | Path = DEFAULT_SESSION_DIR):
        self.base_dir = Path(base_dir).expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_session(
        self,
        session_id: str,
        summary: dict,
    ) -> Path:
        """Save a completed session to disk."""
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            **summary,
        }

        filename = f"{session_id}.json"
        path = self.base_dir / filename
        path.write_text(
            json.dumps(session_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(f"Session saved: {path}")
        return path

    def load_session(self, session_id: str) -> dict | None:
        """Load a session from disk."""
        path = self.base_dir / f"{session_id}.json"
        if not path.exists():
            return None

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        """List recent sessions (metadata only, not full transcripts)."""
        sessions = []
        for path in self.base_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": data.get("session_id", path.stem),
                    "query": data.get("query", ""),
                    "timestamp": data.get("timestamp", ""),
                    "turn_count": data.get("turn_count", 0),
                    "num_citations": data.get("num_citations", 0),
                })
            except Exception:
                continue

        # Sort by timestamp descending (most recent first)
        sessions.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
        return sessions[:limit]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session from disk. Returns True if deleted."""
        path = self.base_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
