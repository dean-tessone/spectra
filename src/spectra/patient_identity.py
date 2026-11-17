"""Versioned patient matching; source identifiers remain provenance, not split keys."""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping

IDENTITY_POLICY_ID = "patient_identity_v2"
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−﹣－"})
_ALIASES: dict[str, str] = {}   # study-specific identifier aliases are not distributed


def canonical_patient_id(value: str) -> str:
    """Clean display spelling without replacing the recorded source ID.

    NRG trailing parentheses are operator annotations. Bare initials are also
    removed when they follow a numeric NRG ID. Other studies' annotations and
    meaningful study numbers are preserved.
    """
    if not isinstance(value, str):
        raise TypeError("patient ID must be a nonempty string")
    text = unicodedata.normalize("NFKC", value).translate(_DASHES)
    text = " ".join(text.replace("\u200b", "").replace("\ufeff", "").split())
    text = re.sub(r"\s*-\s*", "-", text)
    if not text:
        raise ValueError("patient ID must be a nonempty string")
    if re.match(r"^nrg(?:-|\d)", text, re.IGNORECASE):
        text = re.sub(r"(?:\s*\([^)]*\))+\s*$", "", text).strip()
        bare = re.fullmatch(r"(NRG[\w-]*\d)[\s_-]+([A-Za-z.,/ ]+)", text, re.IGNORECASE)
        if bare:
            initials = re.sub(r"[^A-Za-z]", "", bare[2])
            if 1 <= len(initials) <= 5 and initials.casefold() not in {"blood", "test", "pre", "post", "pb"}:
                text = bare[1]
    return _ALIASES.get(text.casefold(), text)


def patient_identity_key(value: str) -> str:
    """Ignore final patient-number padding within a fixed study namespace.

    CASE-003, CASE-03 and CASE-3 match; NRG-BR002 and NRG-BR02 do not.
    No edit-distance, OCR, or arbitrary spelling correction is performed.
    MDA draw suffixes must already be removed to obtain a donor ID.
    """
    text = canonical_patient_id(value).casefold()
    match = re.fullmatch(r"(.*?)([0-9]+)", text)
    if match:
        return match[1] + str(int(match[2]))
    return text


def assert_patient_disjoint(train_ids: Iterable[str], validation_ids: Iterable[str]) -> None:
    overlap = {patient_identity_key(v) for v in train_ids} & {
        patient_identity_key(v) for v in validation_ids
    }
    if overlap:
        raise ValueError(f"canonical patients cross train/validation: {sorted(overlap)[:20]}")


def identity_audit(rows: Iterable[Mapping[str, str]]) -> dict:
    aliases: dict[str, set[str]] = defaultdict(set)
    splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        donor = row.get("donor_id") or row.get("patient_id")
        key = patient_identity_key(donor)
        aliases[key].add(donor)
        if row.get("split"):
            split = "validation" if row["split"] == "val" else row["split"]
            splits[key].add(split)
    return {
        "identity_policy": IDENTITY_POLICY_ID,
        "canonical_donors": len(aliases),
        "aliases": {k: sorted(v) for k, v in sorted(aliases.items()) if len(v) > 1},
        "split_conflicts": {k: sorted(v) for k, v in sorted(splits.items()) if len(v) > 1},
    }
