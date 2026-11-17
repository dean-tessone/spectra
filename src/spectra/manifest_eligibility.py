"""Shared acquisition eligibility and global patient exclusions for all manifests."""

from __future__ import annotations

import pandas as pd

from .patient_identity import patient_identity_key

POLICY_ID = "streck_marienfeld_1well_v1"
EPIC_POLICY_ID = "streck_marienfeld_1well_epic_unknown_v1"
REQUIRED_VALUES = {"tube_type": "Streck", "slide_type": "Marienfeld 1 well"}


def normalized(value: object) -> str:
    """Ignore cosmetic case/whitespace, but never accept a substring match."""
    return "" if pd.isna(value) else " ".join(str(value).split()).casefold()


def is_globally_excluded_patient(value: object) -> bool:
    """Exclude every JHWP patient regardless of diagnosis, label or split."""
    text = normalized(value)
    return bool(text) and patient_identity_key(text).startswith("jhwp-")


def eligibility_decisions(
    metadata: pd.DataFrame, *, allow_epic_unknown_tube: bool = False
) -> pd.DataFrame:
    required = {"slide_id", *REQUIRED_VALUES}
    if allow_epic_unknown_tube:
        required.add("sites")
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing eligibility metadata columns: {sorted(missing)}")
    if metadata.slide_id.map(normalized).eq("").any() or not metadata.slide_id.is_unique:
        raise ValueError("Missing or duplicate metadata slide_id")
    result = metadata[["slide_id", *REQUIRED_VALUES]].copy()
    reasons = pd.Series("", index=result.index, dtype=object)
    for field, expected in REQUIRED_VALUES.items():
        values = result[field].map(normalized)
        for index in result.index[values.ne(normalized(expected))]:
            reason = f"missing_{field}" if not values.at[index] else f"{field}_not_allowed"
            reasons.at[index] = ";".join(filter(None, [reasons.at[index], reason]))
    # Explicit October-cohort exception; missing values and other tube types
    # remain excluded, and the Marienfeld rule is never relaxed.
    if allow_epic_unknown_tube:
        epic = (
            metadata.sites.map(normalized).eq("epic science")
            & result.tube_type.map(normalized).eq("unknown")
            & result.slide_type.map(normalized).eq("marienfeld 1 well")
        )
        reasons.loc[epic] = ""
        result["sites"] = metadata.sites
        result["epic_unknown_tube_exception"] = epic
    result["eligibility_policy"] = EPIC_POLICY_ID if allow_epic_unknown_tube else POLICY_ID
    result["eligibility_exclusion_reason"] = reasons
    result["acquisition_eligible"] = reasons.eq("")
    return result


def filter_manifest(
    manifest: pd.DataFrame, metadata: pd.DataFrame, *, allow_epic_unknown_tube: bool = False
):
    """Apply the same AND rule to both labels, keeping authoritative raw values.

    Missing field values exclude slides; absent/ambiguous metadata records stop
    preparation so an incomplete metadata export cannot silently shrink a cohort.
    """
    if not {"slide_id", "label", "patient_id"}.issubset(manifest.columns):
        raise ValueError("Manifest requires slide_id, label and patient_id")
    if manifest.patient_id.map(normalized).eq("").any():
        raise ValueError("Manifest requires a nonempty patient_id for every slide")
    if not manifest.label.isin([0, 1]).all():
        raise ValueError("Manifest labels must be cancer=1 or control=0")
    decisions = eligibility_decisions(metadata, allow_epic_unknown_tube=allow_epic_unknown_tube)
    missing = set(manifest.slide_id) - set(decisions.slide_id)
    if missing:
        raise ValueError(f"Slides missing from eligibility metadata: {sorted(missing)}")
    authoritative = [column for column in decisions if column != "slide_id"]
    joined = manifest.drop(columns=authoritative, errors="ignore").merge(
        decisions, on="slide_id", how="left", validate="many_to_one", sort=False
    )
    excluded_patient = joined.patient_id.map(is_globally_excluded_patient)
    joined.loc[excluded_patient, "eligibility_exclusion_reason"] = joined.loc[
        excluded_patient, "eligibility_exclusion_reason"
    ].map(lambda reason: ";".join(filter(None, [reason, "excluded_jhwp_patient"])))
    # acquisition_eligible records only the tube/slide rule; patient exclusion
    # overrides it even for slides with fully approved acquisition metadata.
    eligible = joined.acquisition_eligible & ~excluded_patient
    if allow_epic_unknown_tube:
        noncontrol_exception = joined.epic_unknown_tube_exception & joined.label.ne(0)
        joined.loc[noncontrol_exception, "eligibility_exclusion_reason"] = joined.loc[
            noncontrol_exception, "eligibility_exclusion_reason"
        ].map(lambda reason: ";".join(filter(None, [reason, "epic_exception_requires_control"])))
        eligible &= ~noncontrol_exception
    return (
        joined.loc[eligible].reset_index(drop=True),
        joined.loc[~eligible].reset_index(drop=True),
    )
