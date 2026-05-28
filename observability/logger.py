"""Interaction logging with SQLite + JSONL backends."""

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH = Path(__file__).parent.parent / "logs" / "interactions.db"
_JSONL_PATH = Path(__file__).parent.parent / "logs" / "interactions.jsonl"


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT    NOT NULL,
            model       TEXT    NOT NULL,
            category    TEXT,
            prompt      TEXT    NOT NULL,
            response    TEXT    NOT NULL,
            latency_ms  REAL,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            safe_input  INTEGER DEFAULT 1,
            safe_output INTEGER DEFAULT 1
        )
    """)
    conn.commit()


class AssistantLogger:
    """Thread-safe logger for assistant interactions."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        jsonl_path: Optional[Path] = None,
    ) -> None:
        self._db_path = db_path or _DB_PATH
        self._jsonl_path = jsonl_path or _JSONL_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            _ensure_db(conn)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def log_interaction(
        self,
        *,
        model: str,
        prompt: str,
        response: str,
        latency_ms: float,
        category: str = "general",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        safe_input: bool = True,
        safe_output: bool = True,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        row = {
            "ts": ts,
            "model": model,
            "category": category,
            "prompt": prompt,
            "response": response,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "safe_input": int(safe_input),
            "safe_output": int(safe_output),
        }
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO interactions
                   (ts, model, category, prompt, response, latency_ms,
                    input_tokens, output_tokens, safe_input, safe_output)
                   VALUES (:ts, :model, :category, :prompt, :response, :latency_ms,
                           :input_tokens, :output_tokens, :safe_input, :safe_output)""",
                row,
            )
            conn.commit()
        finally:
            conn.close()
        with open(self._jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def get_metrics(self, model: Optional[str] = None) -> Dict[str, Any]:
        clause = "WHERE model = ?" if model else ""
        params = (model,) if model else ()
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""SELECT
                       COUNT(*)             AS total,
                       AVG(latency_ms)      AS avg_latency_ms,
                       MIN(latency_ms)      AS min_latency_ms,
                       MAX(latency_ms)      AS max_latency_ms,
                       SUM(input_tokens)    AS total_input_tokens,
                       SUM(output_tokens)   AS total_output_tokens,
                       SUM(1 - safe_input)  AS unsafe_inputs,
                       SUM(1 - safe_output) AS unsafe_outputs
                   FROM interactions {clause}""",
                params,
            ).fetchone()
        finally:
            conn.close()
        keys = [
            "total", "avg_latency_ms", "min_latency_ms", "max_latency_ms",
            "total_input_tokens", "total_output_tokens", "unsafe_inputs", "unsafe_outputs",
        ]
        return dict(zip(keys, rows))

    def get_recent(self, n: int = 20, model: Optional[str] = None) -> List[Dict[str, Any]]:
        clause = "WHERE model = ?" if model else ""
        params = (model,) if model else ()
        conn = self._connect()
        try:
            cur = conn.execute(
                f"SELECT * FROM interactions {clause} ORDER BY id DESC LIMIT ?",
                (*params, n),
            )
            rows = cur.fetchall()
            cols = [d[0] for d in (cur.description or [])]
        finally:
            conn.close()
        if not cols:
            cols = ["id", "ts", "model", "category", "prompt", "response",
                    "latency_ms", "input_tokens", "output_tokens", "safe_input", "safe_output"]
        return [dict(zip(cols, r)) for r in rows]


# Singleton for easy import
default_logger = AssistantLogger()


class timer:
    """Context manager: measures elapsed time in ms."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
