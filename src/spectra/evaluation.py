"""Leakage-safe metrics and selection rules for development model comparisons."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .patient_identity import patient_identity_key


def entity_weights(entity_ids: Sequence[str]) -> np.ndarray:
    counts = Counter(entity_ids)
    return np.asarray([1.0 / counts[value] for value in entity_ids], dtype=np.float64)


def entity_weighted_auroc(
    labels: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    entity_ids: Sequence[str],
) -> float:
    targets = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(logits, dtype=np.float64)
    return float(roc_auc_score(targets, scores, sample_weight=entity_weights(entity_ids)))


def patient_auroc(
    labels: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    entity_ids: Sequence[str],
) -> float:
    """Score each canonical donor once using the mean of their bag logits.

    Canonicalize donor IDs before aggregation. Mixed-label donors (e.g. spike and
    control conditions) require a separate endpoint and are rejected here.
    """
    targets = np.asarray(labels)
    scores = np.asarray(logits, dtype=np.float64)
    if targets.ndim != 1 or scores.ndim != 1 or not (len(targets) == len(scores) == len(entity_ids)):
        raise ValueError("labels, logits and donor IDs must be aligned one-dimensional arrays")
    if not len(scores) or not np.isfinite(scores).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("patient AUROC requires finite logits and binary labels")
    by_donor: dict[str, tuple[int, list[float]]] = {}
    for label, score, donor in zip(targets, scores, entity_ids, strict=True):
        if not isinstance(donor, str) or not donor.strip():
            raise ValueError("patient AUROC requires nonempty canonical donor IDs")
        donor = patient_identity_key(donor)
        previous, values = by_donor.setdefault(donor, (int(label), []))
        if previous != label:
            raise ValueError(f"donor crosses labels: {donor}")
        values.append(float(score))
    return float(roc_auc_score(
        [label for label, _ in by_donor.values()],
        [np.mean(values) for _, values in by_donor.values()],
    ))


def nbd_donor_score_dispersion(
    labels: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    entity_ids: Sequence[str],
) -> float:
    """SD of NBD donor means standardized by group-level NBD score SD."""
    targets = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(logits, dtype=np.float64)
    negative = targets == 0
    by_donor: dict[str, list[float]] = {}
    for score, donor in zip(scores[negative], np.asarray(entity_ids)[negative], strict=True):
        by_donor.setdefault(str(donor), []).append(float(score))
    if len(by_donor) < 2 or scores[negative].std(ddof=1) <= 0:
        return float("nan")
    donor_means = np.asarray([np.mean(values) for values in by_donor.values()])
    return float(donor_means.std(ddof=1) / scores[negative].std(ddof=1))


def nbd_donor_icc(
    labels: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    entity_ids: Sequence[str],
) -> float:
    """One-way random-effects ICC(1,1), supporting unequal donor group counts."""
    targets = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(logits, dtype=np.float64)
    by_donor: dict[str, list[float]] = {}
    for label, score, donor in zip(targets, scores, entity_ids, strict=True):
        if label == 0:
            by_donor.setdefault(donor, []).append(float(score))
    groups = [np.asarray(values) for values in by_donor.values()]
    donor_count = len(groups)
    total_count = sum(len(group) for group in groups)
    if donor_count < 2 or total_count <= donor_count:
        return float("nan")
    grand_mean = np.concatenate(groups).mean()
    between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    within = sum(((group - group.mean()) ** 2).sum() for group in groups)
    between_mean_square = between / (donor_count - 1)
    within_mean_square = within / (total_count - donor_count)
    n_zero = (total_count - sum(len(group) ** 2 for group in groups) / total_count) / (
        donor_count - 1
    )
    denominator = between_mean_square + (n_zero - 1.0) * within_mean_square
    return float(
        (between_mean_square - within_mean_square) / denominator
        if denominator > 0
        else float("nan")
    )


@dataclass(frozen=True)
class MonotoneAffineCalibrator:
    scale: float
    intercept: float

    def transform(self, logits: Sequence[float] | np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(logits, dtype=np.float64) + self.intercept


def fit_monotone_affine_calibrator(
    logits: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    entity_ids: Sequence[str],
) -> MonotoneAffineCalibrator:
    """Fit positive-scale Platt calibration with entity-balanced BCE."""
    raw = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    target = torch.as_tensor(np.asarray(labels), dtype=torch.float64)
    weights = torch.as_tensor(entity_weights(entity_ids), dtype=torch.float64)
    weights = weights / weights.sum()
    raw_scale = torch.nn.Parameter(torch.tensor(0.5413248546, dtype=torch.float64))
    intercept = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        (raw_scale, intercept), max_iter=100, tolerance_grad=1e-10, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scale = torch.nn.functional.softplus(raw_scale)
        calibrated = scale * raw + intercept
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                calibrated, target, reduction="none"
            )
            * weights
        ).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return MonotoneAffineCalibrator(
        scale=float(torch.nn.functional.softplus(raw_scale).detach()),
        intercept=float(intercept.detach()),
    )


def threshold_at_donor_weighted_specificity(
    calibrated_logits: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    entity_ids: Sequence[str],
    target_specificity: float = 0.95,
) -> float:
    if not 0 < target_specificity <= 1:
        raise ValueError("target specificity must be in (0, 1]")
    scores = np.asarray(calibrated_logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int8)
    negative = targets == 0
    if not negative.any():
        raise ValueError("threshold selection requires NBD predictions")
    negative_scores = scores[negative]
    negative_entities = np.asarray(entity_ids)[negative].tolist()
    weights = entity_weights(negative_entities)
    order = np.argsort(negative_scores)
    sorted_scores = negative_scores[order]
    cumulative = np.cumsum(weights[order]) / weights.sum()
    index = min(int(np.searchsorted(cumulative, target_specificity, side="left")), len(order) - 1)
    return float(np.nextafter(sorted_scores[index], math.inf))


def entity_cluster_bootstrap_auroc_difference(
    labels: Sequence[int] | np.ndarray,
    candidate_logits: Sequence[float] | np.ndarray,
    baseline_logits: Sequence[float] | np.ndarray,
    entity_ids: Sequence[str],
    repetitions: int = 2_000,
    seed: int = 17,
) -> tuple[float, float]:
    """Paired percentile interval, resampling patients/donors as clusters."""
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    labels_array = np.asarray(labels, dtype=np.int8)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    baseline = np.asarray(baseline_logits, dtype=np.float64)
    entities = np.asarray(entity_ids)
    unique = np.unique(entities)
    positions = {entity: np.flatnonzero(entities == entity) for entity in unique}
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, len(unique), replace=True)
        sampled_positions = np.concatenate([positions[entity] for entity in chosen])
        sampled_labels = labels_array[sampled_positions]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_entity_ids = [f"{entity}__draw_{draw}" for draw, entity in enumerate(chosen)]
        expanded_entity_ids = [
            sampled_entity_ids[draw]
            for draw, entity in enumerate(chosen)
            for _ in positions[entity]
        ]
        differences.append(
            entity_weighted_auroc(sampled_labels, candidate[sampled_positions], expanded_entity_ids)
            - entity_weighted_auroc(
                sampled_labels, baseline[sampled_positions], expanded_entity_ids
            )
        )
    if not differences:
        return float("nan"), float("nan")
    return tuple(np.quantile(differences, (0.025, 0.975)).tolist())  # type: ignore[return-value]


def entity_cluster_bootstrap_auroc(
    labels: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    entity_ids: Sequence[str],
    repetitions: int = 2_000,
    seed: int = 17,
) -> tuple[float, float]:
    labels_array = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(logits, dtype=np.float64)
    entities = np.asarray(entity_ids)
    unique = np.unique(entities)
    positions = {entity: np.flatnonzero(entities == entity) for entity in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        chosen = rng.choice(unique, len(unique), replace=True)
        sampled_positions = np.concatenate([positions[entity] for entity in chosen])
        sampled_labels = labels_array[sampled_positions]
        if len(np.unique(sampled_labels)) < 2:
            continue
        sampled_entity_ids = [
            f"{entity}__draw_{draw}"
            for draw, entity in enumerate(chosen)
            for _ in positions[entity]
        ]
        values.append(
            entity_weighted_auroc(sampled_labels, scores[sampled_positions], sampled_entity_ids)
        )
    if not values:
        return float("nan"), float("nan")
    return tuple(np.quantile(values, (0.025, 0.975)).tolist())  # type: ignore[return-value]


def ladder_step_advances(
    previous_aurocs: Sequence[float],
    candidate_aurocs: Sequence[float],
    previous_donor_dispersion: float,
    candidate_donor_dispersion: float,
    previous_pooled_auroc: float | None = None,
    candidate_pooled_auroc: float | None = None,
) -> tuple[bool, dict[str, float | int | bool]]:
    if len(previous_aurocs) != 3 or len(candidate_aurocs) != 3:
        raise ValueError("the pre-registered ladder requires exactly three inner folds")
    changes = np.asarray(candidate_aurocs) - np.asarray(previous_aurocs)
    previous_pooled = (
        float(np.mean(previous_aurocs)) if previous_pooled_auroc is None else previous_pooled_auroc
    )
    candidate_pooled = (
        float(np.mean(candidate_aurocs))
        if candidate_pooled_auroc is None
        else candidate_pooled_auroc
    )
    pooled_change = candidate_pooled - previous_pooled
    improved_folds = int((changes > 0).sum())
    worst_decline = float(changes.min())
    dispersion_ratio = (
        candidate_donor_dispersion / previous_donor_dispersion
        if previous_donor_dispersion > 0
        else float("inf")
    )
    advances = (
        pooled_change >= 0.01
        and improved_folds >= 2
        and worst_decline >= -0.02
        and dispersion_ratio <= 1.10
    )
    return advances, {
        "advances": advances,
        "pooled_entity_weighted_auroc_change": pooled_change,
        "improved_folds": improved_folds,
        "worst_fold_change": worst_decline,
        "nbd_donor_dispersion_ratio": dispersion_ratio,
    }
