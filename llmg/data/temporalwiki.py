"""Load TemporalWiki drift CL (easy) from Hugging Face cache."""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

DATASET_ID = "saxenan3/temporalwiki-drift-cl-easy"


def load_tw_easy() -> DatasetDict:
    return load_dataset(DATASET_ID)


def dedupe_articles(split: Dataset) -> dict[str, str]:
    """Map subject_sitelink -> article text (last row wins per subject)."""
    corpus: dict[str, str] = {}
    for row in split:
        key = row["subject_sitelink"]
        corpus[key] = row["article"]
    return corpus
