"""Export TemporalWiki rows to filesystem + SQLite (one document per dataset row)."""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict

from llmg.data.temporalwiki import load_tw_easy

log = logging.getLogger(__name__)

ARTICLES_SUBDIR = "articles"
CORPUS_VERSION = 2
MANIFEST_FILENAME = "corpus_manifest.yaml"
DOC_TO_SUBJECT_FILENAME = "doc_to_subject.yaml"
# Legacy name from v1 exports (doc_id -> subject); prefer DOC_TO_SUBJECT_FILENAME.
LEGACY_SUBJECT_INDEX_FILENAME = "subject_index.yaml"


@dataclass(frozen=True)
class ArticleRecord:
    doc_id: str
    subject_sitelink: str
    article: str
    first_edited: str
    last_edited: str
    slice: str
    split: str


def slugify_subject(subject: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", subject.strip(), flags=re.UNICODE)
    slug = slug.strip("_") or "unknown"
    return slug[:120]


def slugify_slice(slice_label: str) -> str:
    slug = re.sub(r"[^\w]+", "_", (slice_label or "unknown").strip(), flags=re.UNICODE)
    return slug.strip("_")[:80] or "unknown"


def doc_id_for_row(subject: str, slice_label: str) -> str:
    return f"{slugify_subject(subject)}__{slugify_slice(slice_label)}"


def _row_to_record(row: dict, split_name: str) -> ArticleRecord:
    subj = row["subject_sitelink"]
    sl = str(row.get("slice") or "")
    return ArticleRecord(
        doc_id=doc_id_for_row(subj, sl),
        subject_sitelink=subj,
        article=row["article"],
        first_edited=str(row.get("snapshot_old") or ""),
        last_edited=str(row.get("snapshot_new") or ""),
        slice=sl,
        split=split_name,
    )


def iter_records_from_splits(ds: DatasetDict, splits: list[str]) -> list[ArticleRecord]:
    """One ArticleRecord per dataset row (no subject deduplication)."""
    records: list[ArticleRecord] = []
    for split_name in splits:
        if split_name not in ds:
            raise KeyError(f"split {split_name!r} not in dataset (have {list(ds.keys())})")
        split: Dataset = ds[split_name]
        for row in split:
            records.append(_row_to_record(row, split_name))
    return records


@dataclass(frozen=True)
class DedupeStats:
    input_rows: int
    unique_doc_ids: int
    dropped_duplicate_rows: int
    text_mismatch_warnings: int


def dedupe_records(records: list[ArticleRecord]) -> tuple[list[ArticleRecord], DedupeStats]:
    """Keep one row per doc_id; log when duplicate rows disagree on article text."""
    by_id: dict[str, ArticleRecord] = {}
    dropped = 0
    mismatches = 0
    for rec in records:
        prev = by_id.get(rec.doc_id)
        if prev is None:
            by_id[rec.doc_id] = rec
            continue
        dropped += 1
        if prev.article != rec.article:
            mismatches += 1
            log.warning(
                "duplicate doc_id %r with differing article text (split %s vs %s)",
                rec.doc_id,
                prev.split,
                rec.split,
            )
    stats = DedupeStats(
        input_rows=len(records),
        unique_doc_ids=len(by_id),
        dropped_duplicate_rows=dropped,
        text_mismatch_warnings=mismatches,
    )
    if dropped:
        log.info(
            "corpus dedupe: %d rows -> %d doc_ids (%d duplicates dropped, %d text mismatches)",
            stats.input_rows,
            stats.unique_doc_ids,
            stats.dropped_duplicate_rows,
            stats.text_mismatch_warnings,
        )
    return list(by_id.values()), stats


def records_to_corpus(records: list[ArticleRecord]) -> dict[str, str]:
    deduped, _ = dedupe_records(records)
    return {r.doc_id: r.article for r in deduped}


def read_corpus_manifest(corpus_root: Path) -> dict | None:
    path = corpus_root / MANIFEST_FILENAME
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def corpus_is_current(corpus_root: Path) -> bool:
    """True when on-disk export matches CORPUS_VERSION and has temporal frontmatter."""
    manifest = read_corpus_manifest(corpus_root)
    if manifest is None or manifest.get("corpus_version") != CORPUS_VERSION:
        return False
    articles_dir = corpus_root / ARTICLES_SUBDIR
    if not articles_dir.is_dir():
        return False
    sample = next(articles_dir.glob("*.md"), None)
    if sample is None:
        return False
    from llmg.memory.fs_store import parse_article_file

    art = parse_article_file(sample)
    if not art.last_edited:
        return False
    return True


def require_versioned_corpus(corpus_root: Path) -> None:
    if corpus_is_current(corpus_root):
        return
    manifest = read_corpus_manifest(corpus_root)
    if manifest is None:
        raise ValueError(
            f"Corpus at {corpus_root} has no {MANIFEST_FILENAME}; re-export required "
            f"(expected corpus_version={CORPUS_VERSION})."
        )
    raise ValueError(
        f"Corpus at {corpus_root} is stale (manifest version={manifest.get('corpus_version')!r}, "
        f"expected {CORPUS_VERSION}). Delete the corpus dir or re-run export."
    )


def export_filesystem(records: list[ArticleRecord], corpus_root: Path) -> Path:
    """Write corpus_root/articles/<doc_id>.md with corpus-style edit metadata."""
    articles_dir = corpus_root / ARTICLES_SUBDIR
    articles_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    for rec in records:
        path = articles_dir / f"{rec.doc_id}.md"
        front = {
            "doc_id": rec.doc_id,
            "subject_sitelink": rec.subject_sitelink,
            "first_edited": rec.first_edited,
            "last_edited": rec.last_edited,
            "slice": rec.slice,
            "split": rec.split,
        }
        body = (
            "---\n"
            + yaml.safe_dump(front, sort_keys=False, allow_unicode=True)
            + "---\n"
            + rec.article
        )
        path.write_text(body, encoding="utf-8")
        index[rec.doc_id] = str(path.relative_to(corpus_root))
    (corpus_root / "doc_index.yaml").write_text(
        yaml.safe_dump(index, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    doc_to_subject = {r.doc_id: r.subject_sitelink for r in records}
    (corpus_root / DOC_TO_SUBJECT_FILENAME).write_text(
        yaml.safe_dump(doc_to_subject, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    return corpus_root


def write_corpus_manifest(
    corpus_root: Path,
    *,
    index_splits: list[str],
    stats: DedupeStats,
) -> None:
    (corpus_root / MANIFEST_FILENAME).write_text(
        yaml.safe_dump(
            {
                "corpus_version": CORPUS_VERSION,
                "index_splits": index_splits,
                "input_rows": stats.input_rows,
                "unique_doc_ids": stats.unique_doc_ids,
                "dropped_duplicate_rows": stats.dropped_duplicate_rows,
                "text_mismatch_warnings": stats.text_mismatch_warnings,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def export_sqlite(records: list[ArticleRecord], db_path: Path) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE articles (
                doc_id TEXT PRIMARY KEY,
                subject_sitelink TEXT NOT NULL,
                article TEXT NOT NULL,
                first_edited TEXT,
                last_edited TEXT,
                slice TEXT,
                split TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX idx_articles_subject ON articles(subject_sitelink)"
        )
        conn.execute(
            "CREATE INDEX idx_articles_edited ON articles(first_edited, last_edited)"
        )
        conn.executemany(
            """
            INSERT INTO articles
                (doc_id, subject_sitelink, article, first_edited, last_edited, slice, split)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.doc_id,
                    r.subject_sitelink,
                    r.article,
                    r.first_edited,
                    r.last_edited,
                    r.slice,
                    r.split,
                )
                for r in records
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def export_all(
    *,
    corpus_root: Path,
    index_splits: list[str],
    ds: DatasetDict | None = None,
) -> tuple[Path, Path, list[ArticleRecord]]:
    if ds is None:
        ds = load_tw_easy()
    raw = iter_records_from_splits(ds, index_splits)
    records, stats = dedupe_records(raw)
    fs_root = export_filesystem(records, corpus_root)
    db_path = export_sqlite(records, corpus_root / "corpus.db")
    write_corpus_manifest(corpus_root, index_splits=index_splits, stats=stats)
    return fs_root, db_path, records


def index_mode_to_splits(index_mode: str) -> list[str]:
    if index_mode == "train":
        return ["train"]
    if index_mode == "train_stable":
        return ["train", "stable"]
    raise ValueError(f"unknown index_mode: {index_mode!r}")
