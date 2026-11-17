"""Leakage-safe linear batch residualization from crossed technical replicates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PairedBatchResidualizer:
    """A fixed standardized nuisance projection estimated from technical repeats."""

    center: np.ndarray
    scale: np.ndarray
    basis: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray

    @property
    def dimensions(self) -> int:
        return int(self.center.shape[0])

    @property
    def maximum_components(self) -> int:
        return int(self.basis.shape[1])

    def transform(
        self,
        values: np.ndarray,
        components: int,
        strength: float = 1.0,
    ) -> np.ndarray:
        """Standardize values and remove a bounded fraction of nuisance coordinates."""
        if values.shape[-1] != self.dimensions:
            raise ValueError(f"expected {self.dimensions} features, got {values.shape[-1]}")
        if not 0 <= components <= self.maximum_components:
            raise ValueError("components exceed the fitted nuisance basis")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("residualization strength must be in [0, 1]")
        standardized = (np.asarray(values, dtype=np.float64) - self.center) / self.scale
        if components:
            basis = self.basis[:, :components]
            standardized = standardized - strength * (standardized @ basis) @ basis.T
        return standardized


def _validate_fit_inputs(
    slide_means: np.ndarray,
    specimens: list[str],
    batches: list[str],
    center: np.ndarray,
    scale: np.ndarray,
) -> None:
    if slide_means.ndim != 2 or slide_means.shape[0] < 2:
        raise ValueError("slide means must be a nonempty two-dimensional matrix")
    if len(specimens) != len(slide_means) or len(batches) != len(slide_means):
        raise ValueError("slide means, specimens, and batches must have equal length")
    if center.shape != (slide_means.shape[1],) or scale.shape != center.shape:
        raise ValueError("center and scale must match the slide feature dimension")
    if not np.isfinite(slide_means).all() or not np.isfinite(center).all():
        raise ValueError("residualizer inputs must be finite")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("residualizer scale must be finite and positive")
    if any(not specimen for specimen in specimens) or any(not batch for batch in batches):
        raise ValueError("specimen and batch identifiers must be nonempty")


def fit_paired_batch_residualizer(
    slide_means: np.ndarray,
    specimens: list[str],
    batches: list[str],
    center: np.ndarray,
    scale: np.ndarray,
    maximum_components: int = 32,
    norm_winsor_quantile: float = 0.95,
) -> tuple[PairedBatchResidualizer, dict[str, Any]]:
    """Fit nuisance directions from within-specimen differences of batch centroids.

    Each crossed specimen contributes equal total squared row weight before the
    magnitude of its standardized contrasts is considered. Contrast norms are
    winsorized to prevent one failed specimen from defining the entire basis.
    """
    slide_means = np.asarray(slide_means, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    _validate_fit_inputs(slide_means, specimens, batches, center, scale)
    if maximum_components < 1:
        raise ValueError("maximum_components must be positive")
    if not 0.5 <= norm_winsor_quantile <= 1.0:
        raise ValueError("norm_winsor_quantile must lie in [0.5, 1]")

    standardized = (slide_means - center) / scale
    by_specimen_batch: dict[str, dict[str, list[int]]] = {}
    for index, (specimen, batch) in enumerate(zip(specimens, batches, strict=True)):
        by_specimen_batch.setdefault(specimen, {}).setdefault(batch, []).append(index)

    raw_contrasts: list[np.ndarray] = []
    specimen_weights: list[float] = []
    crossed_specimens: list[str] = []
    crossed_slides = 0
    for specimen, by_batch in sorted(by_specimen_batch.items()):
        if len(by_batch) < 2:
            continue
        crossed_specimens.append(specimen)
        crossed_slides += sum(len(indices) for indices in by_batch.values())
        centroids = [standardized[indices].mean(axis=0) for indices in by_batch.values()]
        contrasts = [
            centroids[left] - centroids[right]
            for left in range(len(centroids))
            for right in range(left + 1, len(centroids))
        ]
        weight = 1.0 / np.sqrt(len(contrasts))
        raw_contrasts.extend(contrasts)
        specimen_weights.extend([weight] * len(contrasts))
    if len(crossed_specimens) < 2 or not raw_contrasts:
        raise ValueError("at least two cross-batch specimens are required")

    contrast_matrix = np.stack(raw_contrasts)
    norms = np.linalg.norm(contrast_matrix, axis=1)
    positive_norms = norms[norms > np.finfo(np.float64).eps]
    if not len(positive_norms):
        raise ValueError("all within-specimen batch contrasts are zero")
    norm_cap = float(np.quantile(positive_norms, norm_winsor_quantile))
    shrink = np.minimum(1.0, norm_cap / np.maximum(norms, np.finfo(np.float64).eps))
    weighted = contrast_matrix * (np.asarray(specimen_weights, dtype=np.float64) * shrink)[:, None]
    _, singular_values, right_vectors = np.linalg.svd(weighted, full_matrices=False)
    component_count = min(maximum_components, len(singular_values), slide_means.shape[1])
    singular_values = singular_values[:component_count]
    basis = right_vectors[:component_count].T
    variance = np.square(singular_values)
    total_variance = float(np.square(weighted).sum())
    explained = variance / total_variance
    residualizer = PairedBatchResidualizer(
        center=center,
        scale=scale,
        basis=basis,
        explained_variance_ratio=explained,
        singular_values=singular_values,
    )
    summary: dict[str, Any] = {
        "training_slides": len(slide_means),
        "training_specimens": len(by_specimen_batch),
        "crossed_specimens": len(crossed_specimens),
        "crossed_slides": crossed_slides,
        "contrasts": len(raw_contrasts),
        "maximum_components": component_count,
        "norm_winsor_quantile": norm_winsor_quantile,
        "contrast_norm_cap": norm_cap,
        "cumulative_explained_variance": np.cumsum(explained).tolist(),
    }
    return residualizer, summary


def save_paired_batch_residualizer(
    path: str | Path,
    residualizer: PairedBatchResidualizer,
    metadata: dict[str, Any],
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    with output.open("wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray([1], dtype=np.int64),
            center=residualizer.center.astype(np.float32),
            scale=residualizer.scale.astype(np.float32),
            basis=residualizer.basis.astype(np.float32),
            explained_variance_ratio=residualizer.explained_variance_ratio.astype(np.float64),
            singular_values=residualizer.singular_values.astype(np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    return output


def load_paired_batch_residualizer(
    path: str | Path,
) -> tuple[PairedBatchResidualizer, dict[str, Any]]:
    artifact = Path(path).expanduser().resolve(strict=True)
    import json

    with np.load(artifact, allow_pickle=False) as payload:
        if payload["format_version"].tolist() != [1]:
            raise ValueError(f"unsupported batch residualizer artifact: {artifact}")
        residualizer = PairedBatchResidualizer(
            center=payload["center"].astype(np.float64),
            scale=payload["scale"].astype(np.float64),
            basis=payload["basis"].astype(np.float64),
            explained_variance_ratio=payload["explained_variance_ratio"].astype(np.float64),
            singular_values=payload["singular_values"].astype(np.float64),
        )
        metadata = json.loads(payload["metadata_json"].item())
    if residualizer.basis.shape[0] != residualizer.dimensions:
        raise ValueError("batch residualizer basis has the wrong feature dimension")
    if not np.allclose(
        residualizer.basis.T @ residualizer.basis,
        np.eye(residualizer.maximum_components),
        atol=1e-5,
    ):
        raise ValueError("batch residualizer basis is not orthonormal")
    return residualizer, metadata
