"""Shared development-only manifest guards and outer-fold evaluation helpers."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from .evaluation import (
    entity_cluster_bootstrap_auroc,
    entity_weighted_auroc,
    entity_weights,
    nbd_donor_icc,
    nbd_donor_score_dispersion,
)
from .patient_identity import patient_identity_key


def reject_reserved_test_path(path: str | Path) -> Path:
    """Reject test-like paths before opening or resolving them."""
    candidate = Path(path).expanduser()
    lowered_parts = [part.casefold() for part in candidate.parts]
    if candidate.name.casefold() == "test_manifest.csv" or any(
        part in {"test", "reserved_test", "fixed_test"} for part in lowered_parts
    ):
        raise ValueError(f"reserved test access is forbidden during development: {candidate}")
    return candidate.resolve(strict=True)


def read_clean_development_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a clean cancer/NBD development manifest under the permanent test guard."""
    manifest = reject_reserved_test_path(path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest is empty: {manifest}")
    required = {"group_id", "label", "patient_id", "cohort", "slide_id"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    labels_by_entity: dict[str, str] = {}
    for row in rows:
        label = row["label"]
        entity = row["patient_id"].strip()
        expected_cohort = "patient_cancer" if label == "1" else "synthetic_nbd"
        if row["cohort"] != expected_cohort:
            raise ValueError(f"non-clean cohort in development manifest: {row['cohort']!r}")
        if entity.casefold() in set():
            raise ValueError(f"quarantined NBD donor in manifest: {entity}")
        searchable = " ".join(row.values()).casefold()
        for explicit_control in (
            "non-spiked",
            "nonspiked",
            "non-spike",
            "non spiked",
            "non spike",
        ):
            searchable = searchable.replace(explicit_control, "")
        if "spike" in searchable:
            raise ValueError(f"spike-associated row in clean manifest: {row['slide_id']}")
        previous = labels_by_entity.setdefault(entity, label)
        if previous != label:
            raise ValueError(f"entity crosses labels: {entity}")
    return rows


def validate_entity_disjoint_split(
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
) -> dict[str, int]:
    """Require complete patient/NBD-donor exclusion between development folds."""
    train_groups = {row["group_id"] for row in train_rows}
    validation_groups = {row["group_id"] for row in validation_rows}
    train_slides = {row["slide_id"] for row in train_rows}
    validation_slides = {row["slide_id"] for row in validation_rows}
    train_entities = {patient_identity_key(row.get("donor_id") or row["patient_id"]) for row in train_rows}
    validation_entities = {patient_identity_key(row.get("donor_id") or row["patient_id"]) for row in validation_rows}
    if train_groups & validation_groups:
        raise AssertionError("group leakage between training and validation")
    if train_slides & validation_slides:
        raise AssertionError("slide leakage between training and validation")
    if train_entities & validation_entities:
        raise AssertionError("patient/NBD-donor leakage between training and validation")
    return {
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "train_slides": len(train_slides),
        "validation_slides": len(validation_slides),
        "train_entities": len(train_entities),
        "validation_entities": len(validation_entities),
        "entity_overlap": 0,
    }


def manifest_group_metadata(
    rows: Sequence[dict[str, str]],
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        output.setdefault(row["group_id"], row)
    return output


def scored_rows(
    prediction_path: Path, manifest_rows: list[dict[str, str]], fold: int
) -> list[dict[str, object]]:
    metadata = manifest_group_metadata(manifest_rows)
    predictions = pq.read_table(prediction_path).to_pylist()
    if {str(row["group_id"]) for row in predictions} != set(metadata):
        raise ValueError(f"prediction/manifest group mismatch: {prediction_path}")
    output: list[dict[str, object]] = []
    for row in predictions:
        group_id = str(row["group_id"])
        item: dict[str, object] = {
            "inner_fold": fold,
            "group_id": group_id,
            "patient_id": metadata[group_id]["patient_id"],
            "cancer_types": metadata[group_id].get("cancer_types", ""),
            "label": int(row["label"]),
            "logit": float(row["logit"]),
        }
        for key, value in row.items():
            if key.endswith("_logit"):
                item[key] = float(value)
        output.append(item)
    return output


def score_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    labels = np.asarray([int(row["label"]) for row in rows])
    logits = np.asarray([float(row["logit"]) for row in rows])
    entities = [str(row["patient_id"]) for row in rows]
    return {
        "group_auroc": float(roc_auc_score(labels, logits)),
        "entity_weighted_auroc": entity_weighted_auroc(labels, logits, entities),
        "nbd_donor_dispersion": nbd_donor_score_dispersion(labels, logits, entities),
        "nbd_donor_icc": nbd_donor_icc(labels, logits, entities),
    }


def ensemble_outer_rows(
    prediction_paths: Sequence[Path], validation_rows: list[dict[str, str]]
) -> list[dict[str, object]]:
    predictions = [
        {str(row["group_id"]): row for row in pq.read_table(path).to_pylist()}
        for path in prediction_paths
    ]
    metadata = manifest_group_metadata(validation_rows)
    if any(set(seed_predictions) != set(metadata) for seed_predictions in predictions):
        raise ValueError("outer prediction groups do not match validation manifest")
    output: list[dict[str, object]] = []
    for group_id, group_metadata in metadata.items():
        seed_rows = [seed_predictions[group_id] for seed_predictions in predictions]
        row: dict[str, object] = {
            "group_id": group_id,
            "patient_id": group_metadata["patient_id"],
            "cancer_types": group_metadata.get("cancer_types", ""),
            "label": int(group_metadata["label"]),
            "logit": float(np.mean([float(item["logit"]) for item in seed_rows])),
            "seed_logit_standard_deviation": float(
                np.std([float(item["logit"]) for item in seed_rows])
            ),
        }
        for name in (
            "qc_original_cell_count",
            "qc_retained_cell_count",
            "qc_retained_cell_fraction",
        ):
            present = [name in item for item in seed_rows]
            if not any(present):
                continue
            if not all(present):
                raise ValueError(f"seed predictions inconsistently provide {name}")
            values = [item[name] for item in seed_rows]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f"seed predictions disagree on {name} for {group_id}")
            row[name] = values[0]
        component_names = set.intersection(
            *[
                {key for key in item if key.endswith("_logit") and key != "logit"}
                for item in seed_rows
            ]
        )
        for name in sorted(component_names):
            row[name] = float(np.mean([float(item[name]) for item in seed_rows]))
        output.append(row)
    return output


def detailed_outer_summary(
    rows: list[dict[str, object]], include_bootstrap: bool = False
) -> dict[str, object]:
    labels = np.asarray([int(row["label"]) for row in rows])
    calibrated = np.asarray([float(row["calibrated_logit"]) for row in rows])
    entities = [str(row["patient_id"]) for row in rows]
    predictions = np.asarray([int(row["prediction_at_95_specificity"]) for row in rows], dtype=bool)
    weights = entity_weights(entities)
    positive = labels == 1
    negative = ~positive
    summary: dict[str, object] = {
        "groups": len(rows),
        "entities": len(set(entities)),
        "group_auroc": float(roc_auc_score(labels, calibrated)),
        "entity_weighted_auroc": entity_weighted_auroc(labels, calibrated, entities),
        "sensitivity_at_cross_fitted_95_specificity": float(predictions[positive].mean()),
        "entity_weighted_sensitivity_at_cross_fitted_95_specificity": float(
            np.average(predictions[positive], weights=weights[positive])
        ),
        "achieved_donor_weighted_specificity": float(
            np.average(~predictions[negative], weights=weights[negative])
        ),
        "nbd_donor_dispersion": nbd_donor_score_dispersion(labels, calibrated, entities),
        "nbd_donor_icc": nbd_donor_icc(labels, calibrated, entities),
        "mean_seed_logit_standard_deviation": float(
            np.mean([float(row["seed_logit_standard_deviation"]) for row in rows])
        ),
    }
    if include_bootstrap:
        lower, upper = entity_cluster_bootstrap_auroc(
            labels, calibrated, entities, repetitions=2_000, seed=17
        )
        summary["entity_cluster_bootstrap_95_interval"] = [lower, upper]
    donors: dict[str, list[int]] = {}
    for index, (label, entity) in enumerate(zip(labels, entities, strict=True)):
        if label == 0:
            donors.setdefault(entity, []).append(index)
    summary["per_nbd_donor"] = {
        donor: {
            "groups": len(indices),
            "mean_calibrated_logit": float(calibrated[indices].mean()),
            "false_positive_rate": float(predictions[indices].mean()),
        }
        for donor, indices in sorted(donors.items())
    }
    subtype_reports: dict[str, object] = {}
    cancer_rows_by_subtype: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if int(row["label"]) == 1:
            cancer_rows_by_subtype.setdefault(str(row["cancer_types"]), []).append(index)
    for subtype, subtype_indices in sorted(cancer_rows_by_subtype.items()):
        patients = {entities[index] for index in subtype_indices}
        if len(patients) >= 20:
            subtype_reports[subtype] = {
                "patients": len(patients),
                "groups": len(subtype_indices),
                "sensitivity_at_cross_fitted_95_specificity": float(
                    predictions[subtype_indices].mean()
                ),
            }
    summary["cancer_subtypes_at_least_20_patients"] = subtype_reports
    return summary
