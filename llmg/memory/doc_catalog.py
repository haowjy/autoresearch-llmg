"""Index metadata for versioned corpus documents (one row per train slice)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llmg.data.corpus_export import ARTICLES_SUBDIR, ArticleRecord
from llmg.memory.fs_store import parse_article_file


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    subject_sitelink: str
    first_edited: str
    last_edited: str
    slice: str
    split: str
    path: Path | None = None


class DocCatalog:
    """doc_id-keyed corpus; multiple docs per subject allowed."""

    def __init__(self, meta_by_id: dict[str, DocMeta], texts: dict[str, str]) -> None:
        self.meta_by_id = meta_by_id
        self.texts = texts

    def __len__(self) -> int:
        return len(self.meta_by_id)

    def doc_meta_tuple(self) -> dict[str, tuple[str, str, str]]:
        return {
            did: (m.subject_sitelink, m.first_edited, m.last_edited)
            for did, m in self.meta_by_id.items()
        }

    @classmethod
    def from_records(cls, records: list[ArticleRecord]) -> DocCatalog:
        meta_by_id = {
            r.doc_id: DocMeta(
                doc_id=r.doc_id,
                subject_sitelink=r.subject_sitelink,
                first_edited=r.first_edited,
                last_edited=r.last_edited,
                slice=r.slice,
                split=r.split,
            )
            for r in records
        }
        texts = {r.doc_id: r.article for r in records}
        return cls(meta_by_id, texts)

    @classmethod
    def from_corpus_root(cls, corpus_root: Path) -> DocCatalog:
        articles_dir = corpus_root / ARTICLES_SUBDIR
        meta_by_id: dict[str, DocMeta] = {}
        texts: dict[str, str] = {}
        if not articles_dir.is_dir():
            return cls(meta_by_id, texts)
        for path in sorted(articles_dir.glob("*.md")):
            art = parse_article_file(path)
            doc_id = art.doc_id or path.stem
            meta_by_id[doc_id] = DocMeta(
                doc_id=doc_id,
                subject_sitelink=art.subject_sitelink,
                first_edited=art.first_edited,
                last_edited=art.last_edited,
                slice=art.slice,
                split=art.split,
                path=path,
            )
            texts[doc_id] = art.article
        return cls(meta_by_id, texts)

    @classmethod
    def from_sqlite(cls, db_path: Path) -> DocCatalog:
        import sqlite3

        meta_by_id: dict[str, DocMeta] = {}
        texts: dict[str, str] = {}
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT doc_id, subject_sitelink, article, first_edited, last_edited, slice, split
                FROM articles
                """
            ).fetchall()
        finally:
            conn.close()
        for row in rows:
            doc_id = row["doc_id"]
            meta_by_id[doc_id] = DocMeta(
                doc_id=doc_id,
                subject_sitelink=row["subject_sitelink"],
                first_edited=row["first_edited"] or "",
                last_edited=row["last_edited"] or "",
                slice=row["slice"] or "",
                split=row["split"] or "",
            )
            texts[doc_id] = row["article"]
        return cls(meta_by_id, texts)

    def subject_from_doc_id(self, doc_id: str) -> str | None:
        m = self.meta_by_id.get(doc_id)
        return m.subject_sitelink if m else None

    def doc_id_from_path(self, path: Path, corpus_root: Path) -> str | None:
        try:
            rel = path.resolve().relative_to((corpus_root / ARTICLES_SUBDIR).resolve())
        except ValueError:
            return None
        if rel.suffix != ".md":
            return None
        art = parse_article_file(corpus_root / ARTICLES_SUBDIR / rel.name)
        return art.doc_id or rel.stem
