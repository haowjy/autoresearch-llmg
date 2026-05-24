"""Filesystem corpus layout from corpus_export."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from llmg.data.corpus_export import ARTICLES_SUBDIR, slugify_subject


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class FsArticle:
    doc_id: str
    subject_sitelink: str
    path: Path
    article: str
    first_edited: str
    last_edited: str
    slice: str
    split: str


def parse_article_file(path: Path) -> FsArticle:
    text = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body = text
    m = _FRONTMATTER_RE.match(text)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = text[m.end() :]
    doc_id = str(meta.get("doc_id") or path.stem)
    # Backward compat: old exports used valid_from / valid_to
    first_ed = str(meta.get("first_edited") or meta.get("valid_from") or "")
    last_ed = str(meta.get("last_edited") or meta.get("valid_to") or "")
    return FsArticle(
        doc_id=doc_id,
        subject_sitelink=str(meta.get("subject_sitelink") or path.stem),
        path=path,
        article=body,
        first_edited=first_ed,
        last_edited=last_ed,
        slice=str(meta.get("slice") or ""),
        split=str(meta.get("split") or ""),
    )


class FsStore:
    def __init__(self, corpus_root: Path) -> None:
        self.corpus_root = corpus_root
        self.articles_dir = corpus_root / ARTICLES_SUBDIR
        self._by_doc_id: dict[str, FsArticle] | None = None

    def load_all(self) -> dict[str, FsArticle]:
        if self._by_doc_id is not None:
            return self._by_doc_id
        articles: dict[str, FsArticle] = {}
        if not self.articles_dir.is_dir():
            self._by_doc_id = articles
            return articles
        for path in sorted(self.articles_dir.glob("*.md")):
            art = parse_article_file(path)
            articles[art.doc_id] = art
        self._by_doc_id = articles
        return articles

    def corpus_dict(self) -> dict[str, str]:
        return {k: v.article for k, v in self.load_all().items()}

    def path_for_doc_id(self, doc_id: str) -> Path | None:
        articles = self.load_all()
        if doc_id in articles:
            return articles[doc_id].path
        candidate = self.articles_dir / f"{doc_id}.md"
        if candidate.is_file():
            return candidate
        return None

    def path_for_subject(self, subject: str) -> Path | None:
        """First path matching subject (ambiguous when multiple slices exist)."""
        for art in self.load_all().values():
            if art.subject_sitelink == subject:
                return art.path
        slug_path = self.articles_dir / f"{slugify_subject(subject)}.md"
        if slug_path.is_file():
            return slug_path
        return None

    def subject_from_path(self, path: Path) -> str | None:
        art = self._article_from_path(path)
        return art.subject_sitelink if art else None

    def doc_id_from_path(self, path: Path) -> str | None:
        art = self._article_from_path(path)
        return art.doc_id if art else None

    def _article_from_path(self, path: Path) -> FsArticle | None:
        path = path.resolve()
        articles_resolved = self.articles_dir.resolve()
        try:
            rel = path.relative_to(articles_resolved)
        except ValueError:
            if path.suffix == ".md" and path.parent == articles_resolved:
                rel = path.name
            else:
                return None
        if Path(rel).suffix != ".md":
            return None
        article_path = articles_resolved / Path(rel).name
        if not article_path.is_file():
            return None
        return parse_article_file(article_path)
