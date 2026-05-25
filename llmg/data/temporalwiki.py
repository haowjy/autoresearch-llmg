"""Load TemporalWiki drift CL variants from Hugging Face cache."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

DATASET_ID_EASY = "saxenan3/temporalwiki-drift-cl-easy"
DATASET_ID_CL = "saxenan3/temporalwiki-drift-cl"
# Back-compat alias for corpus_export and legacy callers.
DATASET_ID = DATASET_ID_EASY


def load_tw_easy() -> DatasetDict:
    return load_dataset(DATASET_ID_EASY)


def load_tw_cl() -> DatasetDict:
    """Base CL variant: triples + articles (no NL `question` column)."""
    return load_dataset(DATASET_ID_CL)


def cl_query_from_row(row: dict[str, Any]) -> str:
    """Deterministic retrieval query from a base-variant triple row."""
    subj = row["subject_sitelink"]
    rel = row["relation"]
    snap = row["snapshot_new"]
    return f"What is the {rel} of {subj} as of {snap}?"


def dedupe_articles(split: Dataset) -> dict[str, str]:
    """Map subject_sitelink -> article text (last row wins per subject)."""
    corpus: dict[str, str] = {}
    for row in split:
        key = row["subject_sitelink"]
        corpus[key] = row["article"]
    return corpus


def merge_corpus_splits(ds: DatasetDict, splits: list[str]) -> dict[str, str]:
    """Union article corpora from named splits (later splits override same subject)."""
    corpus: dict[str, str] = {}
    for name in splits:
        if name not in ds:
            raise KeyError(f"split {name!r} not in dataset (have {list(ds.keys())})")
        corpus.update(dedupe_articles(ds[name]))
    return corpus
