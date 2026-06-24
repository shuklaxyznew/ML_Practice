import sqlite3
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from observability.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class HistoricalMemory:
    """
    Persistent memory across sessions using SQLite.
    Stores completed incident investigations for future reference.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.historical_memory_db
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id          TEXT PRIMARY KEY,
                    title       TEXT NOT NULL,
                    service     TEXT,
                    severity    TEXT,
                    root_cause  TEXT,
                    recommendations TEXT,
                    confidence  REAL,
                    report      TEXT,
                    created_at  REAL
                )
            """)
            conn.commit()
        logger.info(f"Historical memory DB ready: {self.db_path}")

    def save_incident(
        self,
        incident_id: str,
        title: str,
        service: str,
        severity: str,
        root_cause: str,
        recommendations: List[str],
        confidence: float,
        report: str,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO incidents
                (id, title, service, severity, root_cause,
                 recommendations, confidence, report, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident_id, title, service, severity, root_cause,
                    json.dumps(recommendations), confidence, report, time.time()
                )
            )
            conn.commit()
        logger.info(f"Saved incident {incident_id} to historical memory")

    def get_incident(self, incident_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM incidents WHERE id = ?", (incident_id,)
            ).fetchone()

        if not row:
            return None

        result = dict(row)
        result["recommendations"] = json.loads(result["recommendations"] or "[]")
        return result

    def get_recent(self, limit: int = 10) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            r["recommendations"] = json.loads(r["recommendations"] or "[]")
            results.append(r)
        return results

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        
    def close(self) -> None:
        """Explicitly close any open connections. Call in tests and shutdown."""
        import gc
        gc.collect()
        logger.debug("Historical memory connections released")