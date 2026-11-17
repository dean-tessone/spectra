"""Frozen single-cell QC scoring and shared nonlinear reliability weights."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

QC_V1_THRESHOLD = 0.46509784932900855
QC_FEATURE_COUNT = 512


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve(strict=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qc_log_penalty_numpy(
    unusable_scores: np.ndarray,
    *,
    threshold: float = QC_V1_THRESHOLD,
    exponent: float = 2.0,
    max_suppression: float = 1.0,
) -> np.ndarray:
    """Return the bounded log-reliability penalty used by every model family."""
    scores = np.asarray(unusable_scores)
    if not 0.0 <= threshold < 1.0:
        raise ValueError("QC threshold must be in [0, 1)")
    if exponent <= 0.0:
        raise ValueError("QC penalty exponent must be positive")
    if max_suppression < 1.0 or not math.isfinite(max_suppression):
        raise ValueError("QC maximum suppression must be finite and at least one")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("QC unusability scores must be finite probabilities")
    excess = np.clip((scores - threshold) / (1.0 - threshold), 0.0, 1.0)
    return -math.log(max_suppression) * np.power(excess, exponent)


def qc_reliability_numpy(
    unusable_scores: np.ndarray,
    *,
    threshold: float = QC_V1_THRESHOLD,
    exponent: float = 2.0,
    max_suppression: float = 1.0,
) -> np.ndarray:
    """Return strictly positive cell reliabilities in [1/R, 1]."""
    return np.exp(
        qc_log_penalty_numpy(
            unusable_scores,
            threshold=threshold,
            exponent=exponent,
            max_suppression=max_suppression,
        )
    )


@dataclass(frozen=True)
class FrozenQCModel:
    """Portable raw-Z512 logistic QC classifier."""

    coefficient: np.ndarray
    intercept: float
    threshold: float
    metadata: dict[str, Any]
    artifact_path: Path
    artifact_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> FrozenQCModel:
        artifact_path = Path(path).expanduser().resolve(strict=True)
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("unsupported frozen QC model format")
        if payload.get("positive_class") != "unusable":
            raise ValueError("frozen QC model positive class must be unusable")
        if payload.get("input") != "raw_z512":
            raise ValueError("frozen QC model input must be raw_z512")
        coefficient = np.asarray(payload.get("coefficient"), dtype=np.float64)
        if coefficient.shape != (QC_FEATURE_COUNT,) or not np.isfinite(coefficient).all():
            raise ValueError("frozen QC coefficient must contain 512 finite values")
        intercept = float(payload.get("intercept"))
        threshold = float(payload.get("unusable_threshold"))
        if not math.isfinite(intercept) or not 0.0 <= threshold < 1.0:
            raise ValueError("frozen QC intercept or threshold is invalid")
        return cls(
            coefficient=coefficient,
            intercept=intercept,
            threshold=threshold,
            metadata=payload,
            artifact_path=artifact_path,
            artifact_sha256=file_sha256(artifact_path),
        )

    def predict_proba(self, raw_z512: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_z512, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != QC_FEATURE_COUNT:
            raise ValueError("QC model expects a [cells, 512] raw-Z512 matrix")
        if not np.isfinite(values).all():
            raise ValueError("QC model input contains non-finite values")
        logits = values @ self.coefficient + self.intercept
        probabilities = np.empty_like(logits)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_logits = np.exp(logits[~positive])
        probabilities[~positive] = exp_logits / (1.0 + exp_logits)
        return probabilities


def freeze_joblib_qc_model(
    source_joblib: str | Path,
    output_path: str | Path,
    *,
    metrics_path: str | Path | None = None,
) -> Path:
    """Convert the trusted Q14 sklearn bundle into a portable JSON artifact."""
    import joblib

    source = Path(source_joblib).expanduser().resolve(strict=True)
    bundle = joblib.load(source)
    if bundle.get("positive_class") != "unusable" or bundle.get("input") != "z512_only":
        raise ValueError("source QC bundle is not the expected Z512 unusability model")
    pipeline = bundle.get("deployment_model")
    if pipeline is None or set(pipeline.named_steps) != {"scale", "classifier"}:
        raise ValueError("source QC deployment pipeline has unexpected steps")
    scaler = pipeline.named_steps["scale"]
    classifier = pipeline.named_steps["classifier"]
    scaled_coefficient = np.asarray(classifier.coef_, dtype=np.float64)
    if scaled_coefficient.shape != (1, QC_FEATURE_COUNT):
        raise ValueError("source QC classifier must have one 512-dimensional coefficient")
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    if mean.shape != (QC_FEATURE_COUNT,) or scale.shape != mean.shape or np.any(scale <= 0.0):
        raise ValueError("source QC scaler has invalid statistics")
    coefficient = scaled_coefficient[0] / scale
    intercept = float(classifier.intercept_[0] - np.dot(coefficient, mean))
    threshold = float(bundle.get("unusable_threshold"))
    if not math.isclose(threshold, QC_V1_THRESHOLD, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Q14 threshold mismatch: expected {QC_V1_THRESHOLD:.13f}, got {threshold:.13f}"
        )
    metrics: dict[str, Any] = {}
    metrics_sha256: str | None = None
    if metrics_path is not None:
        resolved_metrics = Path(metrics_path).expanduser().resolve(strict=True)
        metrics = json.loads(resolved_metrics.read_text(encoding="utf-8"))
        metrics_sha256 = file_sha256(resolved_metrics)
        recorded = float(metrics.get("unusable_threshold_selected_on_validation"))
        if not math.isclose(recorded, threshold, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("QC metrics and deployment bundle thresholds do not match")
    payload = {
        "format_version": 1,
        "model_version": "qc-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": bundle.get("model_id"),
        "input": "raw_z512",
        "feature_columns": [f"z{index}" for index in range(QC_FEATURE_COUNT)],
        "positive_class": "unusable",
        "unusable_threshold": threshold,
        "coefficient": coefficient.tolist(),
        "intercept": intercept,
        "source_joblib": str(source),
        "source_joblib_sha256": file_sha256(source),
        "source_metrics_sha256": metrics_sha256,
        "annotation_count_at_fit": bundle.get("annotation_count_at_fit"),
        "definite_annotation_count_at_fit": bundle.get("definite_annotation_count_at_fit"),
        "primary_annotation_count_at_fit": bundle.get("primary_annotation_count_at_fit"),
        "primary_definite_annotation_count_at_fit": bundle.get(
            "primary_definite_annotation_count_at_fit"
        ),
        "atlas_fingerprint": bundle.get("atlas_fingerprint"),
        "supplement_atlas_fingerprints": bundle.get("supplement_atlas_fingerprints"),
        "validation_slide_balanced": metrics.get("validation_slide_balanced"),
        "fixed_test_read": False,
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)

    # The portable affine form must reproduce the source deployment pipeline.
    probe = np.vstack((mean, mean + scale, mean - scale))
    expected = pipeline.predict_proba(probe)[:, 1]
    observed = FrozenQCModel.load(output).predict_proba(probe)
    if not np.allclose(expected, observed, rtol=1e-12, atol=1e-12):
        output.unlink(missing_ok=True)
        raise AssertionError("portable QC artifact does not reproduce the source classifier")
    return output


def _slide_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_qc_score_cache(
    model_path: str | Path,
    manifest_paths: list[str | Path],
    output_dir: str | Path,
) -> Path:
    """Score every unique development slide and write a fail-closed cache."""
    from .data import load_group_manifest
    from .development import reject_reserved_test_path

    if not manifest_paths:
        raise ValueError("at least one development manifest is required")
    model = FrozenQCModel.load(model_path)
    records = []
    manifest_fingerprints = []
    for manifest_value in manifest_paths:
        manifest = reject_reserved_test_path(manifest_value)
        manifest_fingerprints.append({"path": str(manifest), "sha256": file_sha256(manifest)})
        records.extend(load_group_manifest(manifest, source_roots=()))
    slides: dict[str, Path] = {}
    for record in records:
        for slide_id, slide_path in zip(record.slide_ids, record.slide_paths, strict=True):
            previous = slides.setdefault(slide_id, slide_path)
            if previous != slide_path:
                raise ValueError(f"slide {slide_id!r} resolves to multiple paths")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    score_root = root / "slides"
    score_root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.json"
    reusable_slides: dict[str, dict[str, Any]] = {}
    if metadata_path.is_file():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing_metadata.get("format_version") == 1
            and existing_metadata.get("complete") is True
            and existing_metadata.get("qc_model_sha256") == model.artifact_sha256
        ):
            reusable_slides = dict(existing_metadata.get("slides", {}))
    columns = [f"z{index}" for index in range(QC_FEATURE_COUNT)]
    slide_rows: dict[str, dict[str, Any]] = {}
    reused_slide_count = 0
    for slide_id, slide_path in sorted(slides.items()):
        source = slide_path.expanduser().resolve(strict=True)
        source_fingerprint = _slide_fingerprint(source)
        digest = hashlib.sha256(f"{slide_id}\0{source}".encode()).hexdigest()
        relative_score = Path("slides") / digest[:2] / f"{digest}.npy"
        score_path = root / relative_score
        reusable = reusable_slides.get(slide_id)
        if isinstance(reusable, dict) and all(
            reusable.get(key) == source_fingerprint[key]
            for key in ("path", "size", "mtime_ns")
        ):
            existing_score_path = (root / str(reusable.get("score_file"))).resolve()
            if root in existing_score_path.parents and existing_score_path.is_file():
                existing_scores = np.load(existing_score_path, mmap_mode="r", allow_pickle=False)
                expected_rows = int(reusable.get("rows", -1))
                if (
                    existing_scores.dtype == np.float32
                    and existing_scores.shape == (expected_rows,)
                    and np.isfinite(existing_scores).all()
                    and np.all((existing_scores >= 0.0) & (existing_scores <= 1.0))
                ):
                    slide_rows[slide_id] = {
                        **source_fingerprint,
                        "rows": expected_rows,
                        "score_file": existing_score_path.relative_to(root).as_posix(),
                    }
                    reused_slide_count += 1
                    continue
        table = pq.read_table(source, columns=[*columns, "slide_id"], memory_map=True)
        source_slide_ids = pc.unique(table.column("slide_id")).to_pylist()
        if source_slide_ids != [slide_id]:
            raise ValueError(
                f"{source} contains slide IDs {source_slide_ids!r}; expected {slide_id!r}"
            )
        values = np.column_stack(
            [table.column(column).to_numpy(zero_copy_only=False) for column in columns]
        ).astype(np.float64, copy=False)
        scores = model.predict_proba(values).astype(np.float32)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = score_path.with_name(f".{score_path.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.save(handle, scores, allow_pickle=False)
            os.replace(temporary, score_path)
        finally:
            temporary.unlink(missing_ok=True)
        slide_rows[slide_id] = {
            **source_fingerprint,
            "rows": len(scores),
            "score_file": relative_score.as_posix(),
        }
    metadata = {
        "format_version": 1,
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "positive_class": "unusable",
        "qc_model_path": str(model.artifact_path),
        "qc_model_sha256": model.artifact_sha256,
        "qc_model_id": model.metadata.get("model_id"),
        "unusable_threshold": model.threshold,
        "manifests": manifest_fingerprints,
        "reused_slides": reused_slide_count,
        "scored_slides": len(slides) - reused_slide_count,
        "slides": slide_rows,
    }
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    try:
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary_metadata.unlink(missing_ok=True)
    return metadata_path


class QCScoreCache:
    """Validated read-only access to per-slide QC probabilities."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        metadata_path = self.root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1 or metadata.get("complete") is not True:
            raise ValueError("QC score cache is incomplete or has an unsupported format")
        if metadata.get("positive_class") != "unusable":
            raise ValueError("QC score cache positive class must be unusable")
        model_path = Path(str(metadata.get("qc_model_path"))).expanduser().resolve(strict=True)
        if file_sha256(model_path) != metadata.get("qc_model_sha256"):
            raise ValueError("QC score cache model fingerprint is stale")
        self.metadata = metadata
        self.slides = dict(metadata.get("slides", {}))

    def load(self, path: str | Path, expected_slide_id: str, expected_rows: int) -> np.ndarray:
        source = Path(path).expanduser().resolve(strict=True)
        row = self.slides.get(expected_slide_id)
        if not isinstance(row, dict):
            raise FileNotFoundError(f"QC score cache has no entry for slide {expected_slide_id!r}")
        current = _slide_fingerprint(source)
        for key in ("path", "size", "mtime_ns"):
            if row.get(key) != current[key]:
                raise ValueError(f"QC score cache source fingerprint is stale for {source}")
        score_path = (self.root / str(row.get("score_file"))).resolve(strict=True)
        if self.root not in score_path.parents:
            raise ValueError("QC score cache entry escapes its root")
        scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
        if scores.dtype != np.float32 or scores.shape != (expected_rows,):
            raise ValueError(f"QC scores do not align with source rows for {source}")
        if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError(f"QC score cache contains invalid probabilities for {source}")
        return scores
