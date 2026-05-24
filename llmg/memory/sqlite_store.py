"""Read-only SQLite corpus store (versioned rows keyed by doc_id)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SqlArticle:
    doc_id: str
    subject_sitelink: str
    article: str
    first_edited: str
    last_edited: str
    slice: str
    split: str


class SqliteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def load_all(self) -> dict[str, SqlArticle]:
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT doc_id, subject_sitelink, article, first_edited, last_edited, slice, split
            FROM articles
            """
        ).fetchall()
        out: dict[str, SqlArticle] = {}
        for row in rows:
            out[row["doc_id"]] = SqlArticle(
                doc_id=row["doc_id"],
                subject_sitelink=row["subject_sitelink"],
                article=row["article"],
                first_edited=row["first_edited"] or "",
                last_edited=row["last_edited"] or "",
                slice=row["slice"] or "",
                split=row["split"] or "",
            )
        return out

    def corpus_dict(self) -> dict[str, str]:
        return {k: v.article for k, v in self.load_all().items()}

    def search_fts_like(self, query: str, limit: int = 5) -> list[str]:
        conn = self.connect()
        pattern = f"%{query[:200]}%"
        rows = conn.execute(
            """
            SELECT doc_id FROM articles
            WHERE article LIKE ? OR subject_sitelink LIKE ?
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()
        return [r["doc_id"] for r in rows]
