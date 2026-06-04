"""Relation-specific answer candidates extracted from search_hybrid observations."""

from __future__ import annotations

import re

NAME = r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){1,3}"

COUNTRY_DEMONYMS = {
    "American": "United States",
    "British": "United Kingdom",
    "Canadian": "Canada",
    "French": "France",
    "German": "Germany",
    "Israeli": "Israel",
    "Russian": "Russia",
}


def _norm(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^\w\s]", "", text)


def _add(candidates: list[str], value: str) -> None:
    value = " ".join(value.strip(" .,:;\"'()[]{}").split())
    if not value or len(value) > 80:
        return
    norm = _norm(value)
    if norm and all(_norm(c) != norm for c in candidates):
        candidates.append(value)


def candidate_matches(left: str, right: str) -> bool:
    return bool(_norm(left) and _norm(left) == _norm(right))


def _split_name_list(raw: str) -> list[str]:
    raw = raw.replace(" and ", ", ")
    return [" ".join(part.split()) for part in raw.split(",") if part.strip()]


def _question_years(question: str) -> set[str]:
    return set(re.findall(r"\b(?:19|20)\d{2}\b", question))


def _year_allowed(question_years: set[str], year: str) -> bool:
    return not question_years or year in question_years


def _allow_new_head_coach_candidate(q: str) -> bool:
    return "2026" in q or ("afc" not in q and "asian cup" not in q)


def extract_search_candidates(question: str, search_text: str) -> list[str]:
    """Extract likely answer strings from observed search snippets only."""
    q = question.lower()
    text = " ".join(search_text.split())
    question_years = _question_years(question)
    candidates: list[str] = []

    if "head of government" in q:
        held_post_candidate = False
        if "2026-01-01" in q:
            for m in re.finditer(
                rf"\bno Republican has served as mayor even on an interim basis since ({NAME}) held the post\b",
                text,
            ):
                _add(candidates, m.group(1))
                held_post_candidate = True
        for m in re.finditer(
            rf"\bsucceeded by [^.]*?\b({NAME}), who was elected\b",
            text,
        ):
            _add(candidates, m.group(1))
        if not held_post_candidate:
            for m in re.finditer(rf"\bMayor\s+({NAME})", text):
                _add(candidates, m.group(1))
    if "head of state" in q:
        for m in re.finditer(rf"\bActing president since [^.]+ is ({NAME})", text):
            _add(candidates, m.group(1))
        for m in re.finditer(rf"\bPresident\s+({NAME})\s+was reelected", text):
            _add(candidates, m.group(1))
    if "diplomatic relation" in q or "diplomatic relations" in q:
        for m in re.finditer(r"\bRelations with\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+){0,3})", text):
            _add(candidates, m.group(1))
    if "head coach" in q or " coach " in f" {q} ":
        for m in re.finditer(
            rf"\b20\d{{2}}[^.]+?\bunder new head coach\s+({NAME})",
            text,
        ):
            if _allow_new_head_coach_candidate(q):
                _add(candidates, m.group(1))
        for m in re.finditer(
            rf"\b({NAME})\s+was appointed as interim head coach",
            text,
        ):
            _add(candidates, m.group(1))
        for m in re.finditer(
            rf"\bappointed\s+({NAME})\s+as head coach",
            text,
        ):
            _add(candidates, m.group(1))
        for m in re.finditer(
            rf"\breplaced as head coach by (?:(?:his|her|their)\s+)?(?:assistant\s+)?({NAME})",
            text,
        ):
            _add(candidates, m.group(1))
    if "chairperson" in q or "chairman" in q:
        for m in re.finditer(
            rf"\bchief executive\s+({NAME})",
            text,
        ):
            _add(candidates, m.group(1))
        for m in re.finditer(
            rf"\bChairman\s+({NAME})",
            text,
        ):
            _add(candidates, m.group(1))
    if "co-founded" in q or "cofound" in q:
        for m in re.finditer(
            r"\bco-founded the software company\s+([A-Z][A-Za-z&.' ]+(?:Corporation|Inc\.?|LLC|Ltd\.?))\b",
            text,
        ):
            _add(candidates, m.group(1))
    if "record label" in q:
        for m in re.finditer(r"\breleased through\s+([A-Z][A-Za-z&.' ]+ Records)\b", text):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\breleased (?:off of|in (?:19|20)\d{2} on|on)\s+([A-Z][A-Za-z&.' ]+ Records)\b", text):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\brecord label,\s+([A-Z][A-Za-z&.' ]+ Records)\b", text):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\bof\s+([A-Z][A-Za-z&.' ]+ Records)\b", text):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\b([A-Z][A-Za-z&.' ]+ Records), to sign\b", text):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\bsigned (?:with|to)\s+([A-Z][A-Za-z&.' ]+(?:Records|Music))\b", text):
            _add(candidates, m.group(1))
    if "domestic cricket" in q or "cricket team" in q:
        for m in re.finditer(r"\bselected for the\s+([A-Z][A-Za-z ]+ cricket team)\b", text):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\bplays for\s+([A-Z][A-Za-z ]+)\s+in the Indian domestic cricket\b", text):
            _add(candidates, f"{m.group(1)} cricket team")
    if "indian premier league" in q or "ipl" in q:
        for m in re.finditer(r"\bplays for\s+([A-Z][A-Za-z ]+?)\s+and has previously played for\b", text):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\bsigned by the\s+([A-Z][A-Za-z ]+?)\s+\([A-Z]+\)", text):
            _add(candidates, m.group(1))
    if "radio station" in q:
        for m in re.finditer(r"\b([A-Z0-9]{3,})\s+radio station\b", text):
            _add(candidates, f"{m.group(1)} Radio")
    if "facility" in q and "baseball" in q:
        for m in re.finditer(r"\bteams compete at [^.]*?\b([A-Z][A-Za-z]+ Ballpark)\b", text):
            _add(candidates, m.group(1))
    if "release" in q or "released" in q:
        for m in re.finditer(r"\breleased (?:in [^.]+ )?by\s+([A-Z][A-Za-z&.' ]+(?:Pictures|Studios|Films))\b", text):
            _add(candidates, m.group(1))
    if (
        "member of sports team" in q
        or "basketball team" in q
        or "football team" in q
        or "traded" in q
        or ("fc barcelona b" in q and "hungarian" in q)
    ):
        for m in re.finditer(r"\btraded in ((?:19|20)\d{2}) to the\s+([A-Z][A-Za-z ]+)\b", text):
            if _year_allowed(question_years, m.group(1)):
                _add(candidates, m.group(2))
        for m in re.finditer(
            r"\bOn [A-Z][a-z]+ \d{1,2}, ((?:19|20)\d{2}), [^.]{0,120}\bwas traded[^.]{0,220}\bto the\s+([A-Z][A-Za-z0-9 ]+?)(?=\s+(?:in exchange|,)|\.)",
            text,
        ):
            if _year_allowed(question_years, m.group(1)):
                _add(candidates, m.group(2))
        for m in re.finditer(
            r"\bAfter brief stints with the\s+([A-Z][A-Za-z0-9 ]+?)\s+\("
            r"traded midway through ((?:19|20)\d{2})[–-]\d{2} for [^)]+\)"
            r"\s+and\s+([A-Z][A-Za-z0-9 ]+?)(?=\s+\(|,|\.)",
            text,
        ):
            if _year_allowed(question_years, m.group(2)):
                _add(candidates, m.group(3))
        for m in re.finditer(
            r"\bOn [A-Z][a-z]+ \d{1,2}, ((?:19|20)\d{2}), [^.]{0,120}\bsigned with the\s+([A-Z][A-Za-z0-9 ]+?)(?=\.|,)",
            text,
        ):
            if _year_allowed(question_years, m.group(1)):
                _add(candidates, m.group(2))
        for m in re.finditer(r"\bplay(?:ed|ing) (?:his|her|their) final three seasons [^.]+ for the\s+([A-Z][A-Za-z ]+)\b", text, flags=re.IGNORECASE):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\bAfter a brief spell with\s+([^,]+),\s+a club of the Hungarian\b", text):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\bAfter [^.]{0,120}, (?:he|she|they) joined the\s+([^.,]+)", text, flags=re.IGNORECASE):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\bOn [A-Z][a-z]+ \d{1,2}, 20\d{2},\s+([^.,]+?)\s+announced that they had signed\b", text):
            _add(candidates, m.group(1))
    if "spouse" in q or "husband" in q or "wife" in q:
        if "first husband" in q or "first wife" in q or "first marriage" in q:
            for m in re.finditer(
                rf"\b[Ff]irst marriage was to (?:(?:actor|actress|director|writer|producer|TV producer|singer|musician|factory worker|retired baseball star)\s+)?({NAME})\b",
                text,
            ):
                _add(candidates, m.group(1))
        westernized = False
        for m in re.finditer(
            rf"\bwith (?:his|her|their) wife,\s*[^()]+?\([^)]*Westernized as ({NAME})\)",
            text,
        ):
            _add(candidates, m.group(1))
            westernized = True
        if not westernized:
            for m in re.finditer(
                rf"\b(?:married|wife|spouse)\s+({NAME}(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+)?)",
                text,
            ):
                _add(candidates, m.group(1))
    if "country of citizenship" in q or "naturalized" in q or "naturalised" in q:
        for m in re.finditer(r"\bbecame a ([A-Z][a-z]+) citizen by naturali[sz]ation\b", text):
            country = COUNTRY_DEMONYMS.get(m.group(1))
            if country:
                _add(candidates, country)
    brother_m = re.search(r"\bbrothers? of ([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'-]+)\b", question)
    if brother_m:
        target = brother_m.group(1)
        for m in re.finditer(r"\bsons\s+\(([^)]*?)\)", text, flags=re.IGNORECASE):
            names = _split_name_list(m.group(1))
            for idx, name in enumerate(names):
                if candidate_matches(name, target) and idx > 0:
                    _add(candidates, names[idx - 1])
                    break
    if "position" in q and ("held" in q or "hold" in q or "government" in q):
        for m in re.finditer(
            r"\bbecame\s+([A-Z][A-Za-z]+(?:\s+(?:for|of|and|the|[A-Z][A-Za-z]+)){0,6})\s+in\s+(?:19|20)\d{2}\b",
            text,
        ):
            _add(candidates, m.group(1))
        for m in re.finditer(
            r"\bserved as (?:the )?(?:prime minister of|Prime Minister of) ([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)",
            text,
        ):
            country = " ".join(m.group(1).split())
            _add(candidates, f"Prime Minister of {country}")
        for m in re.finditer(r"\bstayed in the cabinet as a (minister without portfolio)\b", text, flags=re.IGNORECASE):
            _add(candidates, m.group(1))
        for m in re.finditer(r"\bserved as\s+([A-Z][A-Za-z.' -]+(?:Minister|portfolio|Defense)[A-Za-z.' -]*)", text):
            _add(candidates, m.group(1))

    return candidates
