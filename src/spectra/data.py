"""Manifest-driven parquet loading for grouped slide contexts."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import uuid
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from .config import DataConfig
from .patient_identity import patient_identity_key
from .quality import QCScoreCache

LOGGER = logging.getLogger(__name__)

TOP_SUFFIX = "_top10000.parquet"
INSTANCE_METADATA_COLUMNS = (
    "frame_id",
    "cell_id",
    "y",
    "x",
    "area",
    "eccentricity",
    "DAPI_mean",
    "TRITC_mean",
    "CY5_mean",
    "FITC_mean",
    "wbcclass",
    "outlier_score",
)


def transform_auxiliary_values(column: str, values: np.ndarray) -> np.ndarray:
    """Apply the feature transform used by the model to one slide metadata column."""
    transformed = values.astype(np.float32, copy=False)
    if column == "outlier_score":
        return (transformed - 100.0) / 50.0
    if column == "area":
        return np.log1p(transformed) / 10.0
    if column in {"DAPI_mean", "TRITC_mean", "CY5_mean", "FITC_mean"}:
        return np.log1p(transformed) / 12.0
    return transformed


def embedding_cache_path(
    root: str | Path,
    path: str | Path,
    expected_slide_id: str,
    columns: Sequence[str],
) -> Path:
    """Return the stable cache location for one source slide and feature view."""
    source_path = Path(path).expanduser().resolve(strict=True)
    source = source_path.stat()
    fingerprint = {
        "format_version": 1,
        "path": str(source_path),
        "size": source.st_size,
        "mtime_ns": source.st_mtime_ns,
        "slide_id": expected_slide_id,
        "columns": tuple(columns),
    }
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()
    return Path(root).expanduser().resolve() / digest[:2] / f"{digest}.npy"


@dataclass(frozen=True)
class GroupRecord:
    """One model example, preserving the slide order from the supplied manifest."""

    group_id: str
    slide_ids: tuple[str, ...]
    slide_paths: tuple[Path, ...]
    label: float
    entity_id: str
    sampling_stratum: str | None = None
    sample_weight: float = 1.0

    @property
    def num_slides(self) -> int:
        return len(self.slide_ids)


class NegativeEntityPairSampler(Sampler[int]):
    """Yield balanced blocks with two groups from one negative donor.

    Each four-item block contains a same-donor negative pair plus two groups
    drawn independently from positive entities.  With a batch size divisible
    by four, every negative pair remains in one batch, enabling an explicit
    consistency objective without changing two-slide inference semantics.
    """

    def __init__(
        self,
        records: Sequence[GroupRecord],
        num_samples: int,
        seed: int,
        negative_entity_balance_exponent: float = 1.0,
        positive_stratum_balance_exponent: float = 0.0,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        by_label_entity: dict[float, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(records):
            by_label_entity[record.label][record.entity_id].append(index)
        if set(by_label_entity) != {0.0, 1.0}:
            raise ValueError("negative entity-pair sampling requires binary labels 0 and 1")
        if not 0.0 <= positive_stratum_balance_exponent <= 1.0:
            raise ValueError("positive_stratum_balance_exponent must be in [0, 1]")
        if not 0.0 <= negative_entity_balance_exponent <= 1.0:
            raise ValueError("negative_entity_balance_exponent must be in [0, 1]")
        self.negative_groups = tuple(tuple(indices) for indices in by_label_entity[0.0].values())
        self.negative_entity_weights = torch.tensor(
            [
                len(indices) ** (1.0 - negative_entity_balance_exponent)
                for indices in self.negative_groups
            ],
            dtype=torch.float64,
        )
        positive_entities = tuple(by_label_entity[1.0].items())
        positive_strata: list[str] = []
        for entity_id, indices in positive_entities:
            strata = {records[index].sampling_stratum for index in indices}
            if len(strata) != 1:
                raise ValueError(f"positive entity {entity_id!r} has inconsistent sampling strata")
            stratum = next(iter(strata))
            if positive_stratum_balance_exponent and stratum is None:
                raise ValueError(
                    "positive stratum balancing requires a stratum for every positive entity"
                )
            positive_strata.append(stratum or "__all_positive_entities__")
        self.positive_groups = tuple(tuple(indices) for _, indices in positive_entities)
        stratum_counts = Counter(positive_strata)
        self.positive_entity_weights = torch.tensor(
            [
                stratum_counts[stratum] ** (-positive_stratum_balance_exponent)
                for stratum in positive_strata
            ],
            dtype=torch.float64,
        )
        self.num_samples = num_samples
        self.generator = torch.Generator().manual_seed(seed)

    def _choice(self, values: Sequence[object]) -> object:
        index = int(torch.randint(len(values), (1,), generator=self.generator))
        return values[index]

    def _positive_entity_choice(self) -> tuple[int, ...]:
        position = int(
            torch.multinomial(
                self.positive_entity_weights,
                1,
                replacement=True,
                generator=self.generator,
            )
        )
        return self.positive_groups[position]

    def _negative_entity_choice(self) -> tuple[int, ...]:
        position = int(
            torch.multinomial(
                self.negative_entity_weights,
                1,
                replacement=True,
                generator=self.generator,
            )
        )
        return self.negative_groups[position]

    def __iter__(self):  # type: ignore[no-untyped-def]
        indices: list[int] = []
        while len(indices) < self.num_samples:
            negative_entity = self._negative_entity_choice()
            negative_pair = [
                int(self._choice(negative_entity)),
                int(self._choice(negative_entity)),
            ]
            positive_groups = []
            for _ in range(2):
                positive_entity = self._positive_entity_choice()
                positive_groups.append(int(self._choice(positive_entity)))
            block = negative_pair + positive_groups
            permutation = torch.randperm(4, generator=self.generator).tolist()
            indices.extend(block[index] for index in permutation)
        return iter(indices[: self.num_samples])

    def __len__(self) -> int:
        return self.num_samples


class EntityStructuredBatchSampler(Sampler[list[int]]):
    """Build fixed-composition batches with distinct cancer and NBD entities."""

    def __init__(
        self,
        records: Sequence[GroupRecord],
        positive_entities_per_batch: int,
        negative_entities_per_batch: int,
        seed: int,
        positive_stratum_balance_exponent: float = 0.25,
    ) -> None:
        if positive_entities_per_batch < 1 or negative_entities_per_batch < 1:
            raise ValueError("entity batch class counts must be positive")
        if not 0.0 <= positive_stratum_balance_exponent <= 1.0:
            raise ValueError("positive stratum balance exponent must be in [0, 1]")
        by_label_entity: dict[float, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, record in enumerate(records):
            by_label_entity[record.label][record.entity_id].append(index)
        if set(by_label_entity) != {0.0, 1.0}:
            raise ValueError("structured entity batches require binary labels 0 and 1")
        positive_entities = tuple(sorted(by_label_entity[1.0].items()))
        negative_entities = tuple(sorted(by_label_entity[0.0].items()))
        if len(positive_entities) < positive_entities_per_batch:
            raise ValueError("not enough distinct positive entities for one batch")
        if len(negative_entities) < negative_entities_per_batch:
            raise ValueError("not enough distinct negative entities for one batch")

        positive_strata: list[str] = []
        for entity_id, indices in positive_entities:
            strata = {records[index].sampling_stratum for index in indices}
            if len(strata) != 1:
                raise ValueError(f"positive entity {entity_id!r} has inconsistent sampling strata")
            stratum = next(iter(strata))
            if positive_stratum_balance_exponent and stratum is None:
                raise ValueError(
                    "positive stratum balancing requires every positive entity's stratum"
                )
            positive_strata.append(stratum or "__all_positive_entities__")
        stratum_counts = Counter(positive_strata)
        self.positive_weights = torch.tensor(
            [
                stratum_counts[stratum] ** (-positive_stratum_balance_exponent)
                for stratum in positive_strata
            ],
            dtype=torch.float64,
        )
        self.positive_groups = tuple(tuple(indices) for _, indices in positive_entities)
        self.negative_groups = tuple(tuple(indices) for _, indices in negative_entities)
        self.positive_entities_per_batch = positive_entities_per_batch
        self.negative_entities_per_batch = negative_entities_per_batch
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return math.ceil(len(self.positive_groups) / self.positive_entities_per_batch)

    def _record_from_entity(self, groups: tuple[int, ...]) -> int:
        position = int(torch.randint(len(groups), (), generator=self.generator))
        return groups[position]

    def __iter__(self):  # type: ignore[no-untyped-def]
        for _ in range(len(self)):
            # Sampling is weighted across batches but without replacement inside
            # a batch: rare cancer strata receive mildly more epoch-level mass
            # while an individual patient can never repeat in one objective.
            positive_positions = torch.multinomial(
                self.positive_weights,
                self.positive_entities_per_batch,
                replacement=False,
                generator=self.generator,
            ).tolist()
            negative_positions = torch.randperm(
                len(self.negative_groups), generator=self.generator
            )[: self.negative_entities_per_batch].tolist()
            batch = [
                self._record_from_entity(self.positive_groups[position])
                for position in positive_positions
            ]
            batch.extend(
                self._record_from_entity(self.negative_groups[position])
                for position in negative_positions
            )
            permutation = torch.randperm(len(batch), generator=self.generator).tolist()
            yield [batch[position] for position in permutation]


class MaterializedBatchSampler(Sampler[list[int]]):
    """Replay one immutable, epoch-indexed sequence of dataset indices."""

    def __init__(self, epochs: Sequence[Sequence[Sequence[int]]], position: int = 0) -> None:
        self.epochs = tuple(
            tuple(tuple(int(index) for index in batch) for batch in epoch) for epoch in epochs
        )
        self.position = int(position)
        if not self.epochs or not 0 <= self.position <= len(self.epochs):
            raise ValueError("invalid materialized batch schedule or position")

    def __len__(self) -> int:
        if self.position >= len(self.epochs):
            return len(self.epochs[-1])
        return len(self.epochs[self.position])

    def __iter__(self):  # type: ignore[no-untyped-def]
        if self.position >= len(self.epochs):
            raise RuntimeError("materialized batch schedule is exhausted")
        epoch = self.epochs[self.position]
        self.position += 1
        yield from epoch

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        position = int(state["position"])
        if not 0 <= position <= len(self.epochs):
            raise ValueError("materialized batch schedule position is incompatible")
        self.position = position


def feature_columns(input_dim: int) -> tuple[str, ...]:
    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    return tuple(f"z{index}" for index in range(input_dim))


def resolve_slide_paths(
    slide_ids: Sequence[str], source_roots: Sequence[str | Path]
) -> dict[str, Path]:
    """Resolve only requested slide IDs by deterministic filename, without scanning roots."""
    if not source_roots:
        raise ValueError("at least one source root is required")
    roots: list[Path] = []
    for root_value in source_roots:
        root = Path(root_value).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        roots.append(root)

    requested = tuple(dict.fromkeys(slide_ids))
    LOGGER.info(
        "Resolving manifest slide filenames directly: slides=%d roots=%d",
        len(requested),
        len(roots),
    )

    def find_candidates(slide_id: str) -> tuple[str, list[Path]]:
        candidates = [
            root / f"{slide_id}{TOP_SUFFIX}"
            for root in roots
            if (root / f"{slide_id}{TOP_SUFFIX}").is_file()
        ]
        return slide_id, candidates

    paths: dict[str, Path] = {}
    workers = min(32, max(1, len(requested)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        resolved_candidates = executor.map(find_candidates, requested)
    for slide_id, candidates in resolved_candidates:
        if not candidates:
            raise FileNotFoundError(
                f"manifest slide_id {slide_id!r} has no {TOP_SUFFIX} source file"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"slide_id {slide_id!r} is ambiguous: found in {', '.join(map(str, candidates))}"
            )
        paths[slide_id] = candidates[0]
    LOGGER.info("Direct slide filename resolution complete: slides=%d", len(paths))
    return paths


def _resolve_manifest_path(
    raw_path: object,
    manifest_path: Path,
    slide_id: str,
    row_number: int,
    path_column: str,
) -> Path:
    if raw_path is None or str(raw_path).strip() == "":
        raise ValueError(f"missing {path_column} at manifest row {row_number}")
    candidate = Path(str(raw_path).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    candidate = candidate.absolute()
    expected_name = f"{slide_id}{TOP_SUFFIX}"
    if candidate.name != expected_name:
        raise ValueError(
            f"manifest row {row_number} path filename must be {expected_name!r}, "
            f"got {candidate.name!r}"
        )
    return candidate


def _read_manifest_rows(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".parquet", ".pq"}:
        return pq.read_table(path, memory_map=True).to_pylist()
    raise ValueError("manifest must be a CSV or parquet file")


def _parse_label(group_id: str, raw_value: object) -> float:
    if raw_value is None or str(raw_value).strip() == "":
        raise ValueError(f"group {group_id!r} has a missing label")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"group {group_id!r} has invalid label {raw_value!r}") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"group {group_id!r} label must be in [0, 1], got {value}")
    return value


def load_group_manifest(
    manifest_path: str | Path,
    source_roots: Sequence[str | Path] = ("raw/source_a", "raw/source_b"),
    slide_id_column: str = "slide_id",
    slide_path_column: str | None = "slide_path",
    group_id_column: str = "group_id",
    label_column: str = "label",
    entity_id_column: str | None = "patient_id",
    sampling_stratum_column: str | None = None,
    sample_weight_column: str | None = None,
    allow_reused_slides: bool = False,
    max_slides_per_group: int | None = None,
    canonicalize_entity_ids: bool = False,
) -> list[GroupRecord]:
    """Join manifest slide IDs to immutable source files and group in manifest order."""
    path = Path(manifest_path).expanduser().resolve(strict=True)
    LOGGER.info("Reading group manifest: path=%s", path)
    rows = _read_manifest_rows(path)
    if not rows:
        raise ValueError(f"manifest contains no rows: {path}")
    required = {slide_id_column, group_id_column, label_column}
    if sampling_stratum_column is not None:
        required.add(sampling_stratum_column)
    if sample_weight_column is not None:
        required.add(sample_weight_column)
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    LOGGER.info(
        "Manifest parsed: rows=%d slide_column=%s group_column=%s label_column=%s",
        len(rows),
        slide_id_column,
        group_id_column,
        label_column,
    )

    uses_manifest_paths = slide_path_column is not None and slide_path_column in rows[0]
    fallback_paths: dict[str, Path] | None = None
    if uses_manifest_paths:
        LOGGER.info(
            "Using direct slide paths from manifest column without source indexing: column=%s",
            slide_path_column,
        )
    else:
        requested_slide_ids: list[str] = []
        for row_number, row in enumerate(rows, start=2):
            raw_slide_id = row.get(slide_id_column)
            if raw_slide_id is None or str(raw_slide_id).strip() == "":
                raise ValueError(f"missing slide_id at manifest row {row_number}")
            requested_slide_ids.append(str(raw_slide_id).strip())
        LOGGER.info(
            "Manifest has no %s column; using direct filename lookup fallback", slide_path_column
        )
        fallback_paths = resolve_slide_paths(requested_slide_ids, source_roots)

    grouped: dict[str, list[tuple[str, Path, float, str, str | None, float]]] = {}
    seen_slides: set[str] = set()
    seen_paths: set[Path] = set()
    for row_number, row in enumerate(rows, start=2):
        raw_slide_id = row.get(slide_id_column)
        raw_group_id = row.get(group_id_column)
        if raw_slide_id is None or str(raw_slide_id).strip() == "":
            raise ValueError(f"missing slide_id at manifest row {row_number}")
        if raw_group_id is None or str(raw_group_id).strip() == "":
            raise ValueError(f"missing group_id at manifest row {row_number}")
        slide_id = str(raw_slide_id).strip()
        group_id = str(raw_group_id).strip()
        raw_entity_id = row.get(entity_id_column) if entity_id_column is not None else None
        entity_id = (
            str(raw_entity_id).strip()
            if raw_entity_id is not None and str(raw_entity_id).strip()
            else group_id
        )
        if canonicalize_entity_ids:
            if raw_entity_id is None or not str(raw_entity_id).strip():
                raise ValueError(f"canonical donor ID missing at manifest row {row_number}")
            entity_id = patient_identity_key(entity_id)
        raw_sampling_stratum = (
            row.get(sampling_stratum_column) if sampling_stratum_column is not None else None
        )
        sampling_stratum = (
            str(raw_sampling_stratum).strip()
            if raw_sampling_stratum is not None and str(raw_sampling_stratum).strip()
            else None
        )
        raw_sample_weight = row.get(sample_weight_column) if sample_weight_column else 1.0
        try:
            sample_weight = float(raw_sample_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid sample weight at manifest row {row_number}") from exc
        if not math.isfinite(sample_weight) or sample_weight <= 0.0:
            raise ValueError(f"sample weight must be finite and positive at row {row_number}")
        if not allow_reused_slides and slide_id in seen_slides:
            raise ValueError(f"slide_id {slide_id!r} occurs more than once in the manifest")
        seen_slides.add(slide_id)
        if uses_manifest_paths:
            assert slide_path_column is not None
            slide_path = _resolve_manifest_path(
                row.get(slide_path_column), path, slide_id, row_number, slide_path_column
            )
        else:
            assert fallback_paths is not None
            slide_path = fallback_paths[slide_id]
        if not allow_reused_slides and slide_path in seen_paths:
            raise ValueError(f"slide path {slide_path} occurs more than once in the manifest")
        seen_paths.add(slide_path)
        label = _parse_label(group_id, row.get(label_column))
        grouped.setdefault(group_id, []).append(
            (slide_id, slide_path, label, entity_id, sampling_stratum, sample_weight)
        )

    records: list[GroupRecord] = []
    for group_id, slides in grouped.items():
        labels = {item[2] for item in slides}
        if len(labels) != 1:
            raise ValueError(f"group {group_id!r} has inconsistent labels: {sorted(labels)}")
        if max_slides_per_group is not None and len(slides) > max_slides_per_group:
            raise ValueError(
                f"group {group_id!r} has {len(slides)} slides; configured maximum is "
                f"{max_slides_per_group}"
            )
        entity_ids = {item[3] for item in slides}
        if len(entity_ids) != 1:
            raise ValueError(
                f"group {group_id!r} has inconsistent entity IDs: {sorted(entity_ids)}"
            )
        sampling_strata = {item[4] for item in slides}
        if len(sampling_strata) != 1:
            raise ValueError(
                f"group {group_id!r} has inconsistent sampling strata: "
                f"{sorted(str(value) for value in sampling_strata)}"
            )
        sample_weights = {item[5] for item in slides}
        if len(sample_weights) != 1:
            raise ValueError(f"group {group_id!r} has inconsistent sample weights")
        records.append(
            GroupRecord(
                group_id=group_id,
                slide_ids=tuple(item[0] for item in slides),
                slide_paths=tuple(item[1] for item in slides),
                label=slides[0][2],
                entity_id=slides[0][3],
                sampling_stratum=slides[0][4],
                sample_weight=slides[0][5],
            )
        )
    positive_groups = sum(record.label >= 0.5 for record in records)
    negative_groups = len(records) - positive_groups
    LOGGER.info(
        "Manifest validation complete: groups=%d slides=%d positives=%d negatives=%d",
        len(records),
        sum(record.num_slides for record in records),
        positive_groups,
        negative_groups,
    )
    return records


class GroupedParquetDataset(Dataset[dict[str, Any]]):
    """Lazily concatenate every ordered top-10k slide in a manifest-defined group."""

    def __init__(
        self,
        records: Sequence[GroupRecord],
        input_dim: int = 512,
        max_rows_per_slide: int | None = 10_000,
        top_rows_per_slide: int | None = None,
        embedding_indices: Sequence[int] | None = None,
        embedding_normalization_path: str | Path | None = None,
        auxiliary_columns: Sequence[str] = (),
        feature_dtype: str = "float16",
        slide_cache_size: int = 256,
        raw_embedding_cache_dir: str | Path | None = None,
        require_raw_embedding_cache: bool = False,
        transformed_embedding_cache_dir: str | Path | None = None,
        qc_score_cache_dir: str | Path | None = None,
        qc_hard_filter_threshold: float | None = None,
        immune_score_hard_filter_threshold: float | None = None,
        immune_score_column: str | None = None,
        marker_mean_hard_filter_threshold: float | None = None,
        marker_mean_hard_filter_columns: Sequence[str] = (
            "TRITC_mean",
            "CY5_mean",
            "FITC_mean",
        ),
        preserve_filtered_slide_exposure: bool = False,
        negative_entity_dynamic_slide_pairing: bool = False,
        negative_cross_entity_pair_probability: float = 0.0,
    ) -> None:
        self.preserve_filtered_slide_exposure = preserve_filtered_slide_exposure
        self.records = list(records)
        if not self.records:
            raise ValueError("dataset requires at least one group")
        if slide_cache_size < 0:
            raise ValueError("slide_cache_size must be non-negative")
        self.slide_cache_size = slide_cache_size
        self._slide_cache: OrderedDict[tuple[Path, str], torch.Tensor] = OrderedDict()
        self.negative_entity_dynamic_slide_pairing = negative_entity_dynamic_slide_pairing
        if not 0.0 <= negative_cross_entity_pair_probability <= 1.0:
            raise ValueError("negative cross-entity pair probability must be in [0, 1]")
        if negative_cross_entity_pair_probability and not negative_entity_dynamic_slide_pairing:
            raise ValueError(
                "negative cross-entity pairing requires dynamic negative slide pairing"
            )
        self.negative_cross_entity_pair_probability = negative_cross_entity_pair_probability
        negative_slide_pools: dict[str, dict[str, Path]] = {}
        for record in self.records:
            if record.label >= 0.5:
                continue
            entity_pool = negative_slide_pools.setdefault(record.entity_id, {})
            for slide_id, slide_path in zip(record.slide_ids, record.slide_paths):
                previous_path = entity_pool.setdefault(slide_id, slide_path)
                if previous_path != slide_path:
                    raise ValueError(
                        f"negative entity {record.entity_id!r} maps slide "
                        f"{slide_id!r} to multiple paths"
                    )
        self.negative_slide_pools = {
            entity_id: tuple(slides.items()) for entity_id, slides in negative_slide_pools.items()
        }
        self.negative_entity_ids = tuple(sorted(self.negative_slide_pools))
        if self.negative_entity_dynamic_slide_pairing:
            LOGGER.info(
                "Dynamic negative-entity slide pairing enabled: entities=%d "
                "slides=%d slides_per_sample=2",
                len(self.negative_slide_pools),
                sum(len(pool) for pool in self.negative_slide_pools.values()),
            )
            if self.negative_cross_entity_pair_probability:
                LOGGER.info(
                    "Cross-donor negative pairing enabled: probability=%.3f entities=%d",
                    self.negative_cross_entity_pair_probability,
                    len(self.negative_entity_ids),
                )
        self.auxiliary_columns = tuple(auxiliary_columns)
        self.file_auxiliary_columns = tuple(
            column for column in self.auxiliary_columns if column != "rank_fraction"
        )
        embedding_dim = input_dim - len(self.auxiliary_columns)
        if embedding_dim < 0:
            raise ValueError("auxiliary_columns cannot exceed input_dim")
        if embedding_indices is None:
            selected_embedding_indices = tuple(range(embedding_dim))
        else:
            selected_embedding_indices = tuple(embedding_indices)
            if len(selected_embedding_indices) != embedding_dim:
                raise ValueError(
                    "embedding_indices length must equal input_dim minus auxiliary columns"
                )
            if any(index < 0 for index in selected_embedding_indices):
                raise ValueError("embedding_indices cannot contain negative values")
            if len(set(selected_embedding_indices)) != len(selected_embedding_indices):
                raise ValueError("embedding_indices cannot contain duplicates")
        self.embedding_indices = selected_embedding_indices
        self.columns = tuple(f"z{index}" for index in selected_embedding_indices)
        self.raw_embedding_cache_dir: Path | None = None
        self.require_raw_embedding_cache = require_raw_embedding_cache
        self.transformed_embedding_cache_dir: Path | None = None
        self.transformed_embedding_cache_dtype: np.dtype[Any] | None = None
        self.transformed_cached_auxiliary_columns: tuple[str, ...] = ()
        self.qc_score_cache = (
            QCScoreCache(qc_score_cache_dir) if qc_score_cache_dir is not None else None
        )
        self.qc_hard_filter_threshold = qc_hard_filter_threshold
        if qc_hard_filter_threshold is not None:
            if self.qc_score_cache is None:
                raise ValueError("QC hard filtering requires a QC score cache")
            if not 0.0 <= qc_hard_filter_threshold < 1.0:
                raise ValueError("QC hard-filter threshold must be in [0, 1)")
        if self.qc_score_cache is not None and top_rows_per_slide is not None:
            raise ValueError("QC score cache is incompatible with top_rows_per_slide")
        self.immune_score_hard_filter_threshold = immune_score_hard_filter_threshold
        self.immune_score_column = immune_score_column
        self._immune_keep_cache: dict[tuple[Path, str], torch.Tensor] = {}
        if immune_score_hard_filter_threshold is not None:
            if not np.isfinite(immune_score_hard_filter_threshold) or not 0 < immune_score_hard_filter_threshold <= 1:
                raise ValueError("immune score threshold must be in (0, 1]")
            if not immune_score_column:
                raise ValueError("immune score filtering requires an explicit column")
            if top_rows_per_slide is not None:
                raise ValueError("immune score filtering is incompatible with top rows")
        self.marker_mean_hard_filter_threshold = marker_mean_hard_filter_threshold
        self.marker_mean_hard_filter_columns = tuple(marker_mean_hard_filter_columns)
        self._marker_keep_cache: dict[tuple[Path, str], torch.Tensor] = {}
        if marker_mean_hard_filter_threshold is not None:
            if not np.isfinite(marker_mean_hard_filter_threshold):
                raise ValueError("marker mean hard-filter threshold must be finite")
            if marker_mean_hard_filter_threshold < 0.0:
                raise ValueError("marker mean hard-filter threshold must be non-negative")
            if not self.marker_mean_hard_filter_columns:
                raise ValueError("marker mean hard filtering requires marker columns")
            if top_rows_per_slide is not None:
                raise ValueError("marker mean hard filtering is incompatible with top rows")
        if transformed_embedding_cache_dir is not None:
            if embedding_normalization_path is not None:
                raise ValueError("transformed embedding cache already contains fold normalization")
            if top_rows_per_slide is not None:
                raise ValueError(
                    "transformed embedding cache is incompatible with top_rows_per_slide"
                )
            cache_dir = Path(transformed_embedding_cache_dir).expanduser().resolve(strict=True)
            metadata_path = cache_dir / "metadata.json"
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("format_version") != 1:
                raise ValueError("unsupported transformed embedding cache format")
            if metadata.get("complete") is not True:
                raise ValueError("transformed embedding cache is incomplete")
            if int(metadata.get("dimensions", -1)) != embedding_dim:
                raise ValueError("transformed embedding cache dimensions do not match")
            if tuple(metadata.get("embedding_indices", ())) != self.embedding_indices:
                raise ValueError("transformed embedding cache indices do not match")
            # Early embedding-only caches recorded JSON null; it is equivalent
            # to an empty auxiliary-column list and remains safe to reuse.
            cached_auxiliary = tuple(metadata.get("auxiliary_columns") or ())
            requested_cached_auxiliary = tuple(
                column for column in self.file_auxiliary_columns if column in cached_auxiliary
            )
            if cached_auxiliary != requested_cached_auxiliary:
                raise ValueError(
                    "transformed embedding cache contains unrequested or reordered "
                    "auxiliary columns"
                )
            cache_dtype = metadata.get("dtype")
            if cache_dtype not in {"float16", "float32"}:
                raise ValueError("transformed embedding cache dtype is invalid")
            self.transformed_embedding_cache_dir = cache_dir
            self.transformed_embedding_cache_dtype = np.dtype(cache_dtype)
            self.transformed_cached_auxiliary_columns = cached_auxiliary
            LOGGER.info(
                "Transformed embedding cache enabled: directory=%s dtype=%s",
                cache_dir,
                cache_dtype,
            )
        if raw_embedding_cache_dir is not None:
            if top_rows_per_slide is not None:
                raise ValueError("raw embedding cache is incompatible with top_rows_per_slide")
            cache_dir = Path(raw_embedding_cache_dir).expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.raw_embedding_cache_dir = cache_dir
            LOGGER.info("Raw embedding disk cache enabled: directory=%s", cache_dir)
        elif require_raw_embedding_cache:
            raise ValueError("required raw embedding cache directory was not supplied")
        self.embedding_mean: np.ndarray | None = None
        self.embedding_standard_deviation: np.ndarray | None = None
        if embedding_normalization_path is not None:
            normalization_path = (
                Path(embedding_normalization_path).expanduser().resolve(strict=True)
            )
            with np.load(normalization_path) as normalization:
                mean = normalization["mean"].astype(np.float32, copy=False)
                standard_deviation = normalization["standard_deviation"].astype(
                    np.float32, copy=False
                )
                normalization_indices = tuple(
                    int(index) for index in normalization["embedding_indices"]
                )
            expected_shape = (embedding_dim,)
            if mean.shape != expected_shape or standard_deviation.shape != expected_shape:
                raise ValueError(
                    "embedding normalization mean and standard_deviation must match "
                    "the selected embedding dimension"
                )
            if normalization_indices != self.embedding_indices:
                raise ValueError(
                    "embedding normalization indices do not match the selected z columns"
                )
            if not np.isfinite(mean).all() or not np.isfinite(standard_deviation).all():
                raise ValueError("embedding normalization statistics must be finite")
            if np.any(standard_deviation <= 0):
                raise ValueError("embedding normalization standard deviations must be positive")
            self.embedding_mean = mean
            self.embedding_standard_deviation = standard_deviation
            LOGGER.info(
                "Embedding normalization loaded: path=%s dimensions=%d",
                normalization_path,
                embedding_dim,
            )
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        if max_rows_per_slide is not None and max_rows_per_slide < 1:
            raise ValueError("max_rows_per_slide must be positive when set")
        self.max_rows_per_slide = max_rows_per_slide
        if top_rows_per_slide is not None and top_rows_per_slide < 1:
            raise ValueError("top_rows_per_slide must be positive when set")
        if (
            max_rows_per_slide is not None
            and top_rows_per_slide is not None
            and top_rows_per_slide > max_rows_per_slide
        ):
            raise ValueError("top_rows_per_slide cannot exceed max_rows_per_slide")
        self.top_rows_per_slide = top_rows_per_slide
        dtype_map = {"float16": torch.float16, "float32": torch.float32}
        if feature_dtype not in dtype_map:
            raise ValueError("feature_dtype must be 'float16' or 'float32'")
        self.feature_dtype = dtype_map[feature_dtype]

    def _raw_embedding_cache_path(self, path: Path, expected_slide_id: str) -> Path | None:
        if self.raw_embedding_cache_dir is None or not self.columns:
            return None
        return embedding_cache_path(
            self.raw_embedding_cache_dir, path, expected_slide_id, self.columns
        )

    def _transformed_embedding_cache_path(self, path: Path, expected_slide_id: str) -> Path | None:
        if self.transformed_embedding_cache_dir is None or not self.columns:
            return None
        return embedding_cache_path(
            self.transformed_embedding_cache_dir,
            path,
            expected_slide_id,
            self.columns,
        )

    def _raw_auxiliary_cache_path(self, path: Path, expected_slide_id: str) -> Path | None:
        if self.raw_embedding_cache_dir is None or not self.file_auxiliary_columns:
            return None
        return embedding_cache_path(
            self.raw_embedding_cache_dir,
            path,
            expected_slide_id,
            self.file_auxiliary_columns,
        )

    def _load_raw_embedding_cache(self, cache_path: Path) -> np.ndarray:
        values = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != len(self.columns) or values.shape[0] < 1:
            raise ValueError(f"invalid raw embedding cache artifact: {cache_path}")
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"non-finite or non-FP32 raw embedding cache: {cache_path}")
        return values

    def _load_raw_auxiliary_cache(self, cache_path: Path) -> np.ndarray:
        values = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if (
            values.ndim != 2
            or values.shape[1] != len(self.file_auxiliary_columns)
            or values.shape[0] < 1
        ):
            raise ValueError(f"invalid raw auxiliary cache artifact: {cache_path}")
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"non-finite or non-FP32 raw auxiliary cache: {cache_path}")
        return values

    def _load_transformed_embedding_cache(self, cache_path: Path) -> np.ndarray:
        if not cache_path.is_file():
            raise FileNotFoundError(f"transformed embedding cache is missing: {cache_path}")
        values = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        expected_columns = len(self.columns) + len(self.transformed_cached_auxiliary_columns)
        if values.ndim != 2 or values.shape[1] != expected_columns or values.shape[0] < 1:
            raise ValueError(f"invalid transformed embedding cache artifact: {cache_path}")
        if values.dtype != self.transformed_embedding_cache_dtype:
            raise ValueError(f"transformed embedding cache dtype mismatch: {cache_path}")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite transformed embedding cache: {cache_path}")
        return values

    @staticmethod
    def _write_raw_embedding_cache(cache_path: Path, values: np.ndarray) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.save(handle, values, allow_pickle=False)
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def _read_slide(self, path: Path, expected_slide_id: str) -> torch.Tensor:
        cache_key = (path, expected_slide_id)
        cached = self._slide_cache.get(cache_key)
        if cached is not None:
            self._slide_cache.move_to_end(cache_key)
            return cached
        raw_cache_path = self._raw_embedding_cache_path(path, expected_slide_id)
        raw_auxiliary_cache_path = self._raw_auxiliary_cache_path(path, expected_slide_id)
        transformed_cache_path = self._transformed_embedding_cache_path(path, expected_slide_id)
        embedding_values: np.ndarray | None = None
        cached_auxiliary_values: dict[str, np.ndarray] = {}
        table: pa.Table | None = None
        if self.require_raw_embedding_cache and (
            raw_cache_path is None or not raw_cache_path.is_file()
        ):
            raise FileNotFoundError(
                f"required raw embedding cache is missing for {expected_slide_id}: {raw_cache_path}"
            )
        if transformed_cache_path is not None:
            cached_values = self._load_transformed_embedding_cache(transformed_cache_path)
            embedding_values = cached_values[:, : len(self.columns)]
            cached_auxiliary_values = {
                column: cached_values[:, len(self.columns) + index]
                for index, column in enumerate(self.transformed_cached_auxiliary_columns)
            }
            source_row_count = embedding_values.shape[0]
            source_slide_ids = [expected_slide_id]
        elif raw_cache_path is not None and raw_cache_path.is_file():
            embedding_values = self._load_raw_embedding_cache(raw_cache_path)
            source_row_count = embedding_values.shape[0]
            source_slide_ids = [expected_slide_id]
        if raw_auxiliary_cache_path is not None and raw_auxiliary_cache_path.is_file():
            auxiliary_matrix = self._load_raw_auxiliary_cache(raw_auxiliary_cache_path)
            cached_auxiliary_values.update(
                {
                    column: auxiliary_matrix[:, index]
                    for index, column in enumerate(self.file_auxiliary_columns)
                }
            )
            if embedding_values is not None and len(auxiliary_matrix) != len(embedding_values):
                raise ValueError(
                    f"raw embedding and auxiliary cache row counts do not match: {path}"
                )
            source_row_count = len(auxiliary_matrix)
            source_slide_ids = [expected_slide_id]
        missing_auxiliary_columns = tuple(
            column
            for column in self.file_auxiliary_columns
            if column not in cached_auxiliary_values
        )
        try:
            if not missing_auxiliary_columns and (embedding_values is not None or not self.columns):
                pass
            elif self.top_rows_per_slide is None:
                table = pq.read_table(
                    path,
                    columns=[
                        *(self.columns if embedding_values is None else ()),
                        *missing_auxiliary_columns,
                        "slide_id",
                    ],
                    memory_map=True,
                )
                source_row_count = table.num_rows
                source_slide_ids = pc.unique(table.column("slide_id")).to_pylist()
            else:
                parquet_file = pq.ParquetFile(path, memory_map=True)
                source_row_count = parquet_file.metadata.num_rows
                if source_row_count < 1:
                    raise ValueError(f"{path} has no cells")
                # Validating the identifier column remains inexpensive, while reading
                # only the first record batch avoids decoding thousands of embedding
                # rows that rank-truncated training will never consume.
                id_table = pq.read_table(path, columns=["slide_id"], memory_map=True)
                source_slide_ids = pc.unique(id_table.column("slide_id")).to_pylist()
                first_batch = next(
                    parquet_file.iter_batches(
                        batch_size=self.top_rows_per_slide,
                        columns=[*self.columns, *missing_auxiliary_columns],
                    )
                )
                table = pa.Table.from_batches([first_batch])
        except pa.ArrowInvalid as exc:
            raise ValueError(
                f"{path} does not contain the configured features and slide_id column"
            ) from exc
        if self.max_rows_per_slide is not None and source_row_count > self.max_rows_per_slide:
            raise ValueError(
                f"{path} has {source_row_count} rows; configured maximum is "
                f"{self.max_rows_per_slide}"
            )
        if source_row_count < 1:
            raise ValueError(f"{path} has no cells")
        if embedding_values is not None and len(embedding_values) != source_row_count:
            raise ValueError(
                f"cached embeddings contain {len(embedding_values)} rows but {path} "
                f"contains {source_row_count} rows"
            )
        if source_slide_ids != [expected_slide_id]:
            raise ValueError(
                f"{path} contains slide_id values {source_slide_ids!r}; "
                f"manifest expects {expected_slide_id!r}"
            )
        numpy_dtype = np.float16 if self.feature_dtype == torch.float16 else np.float32
        feature_parts: list[np.ndarray] = []
        if self.columns:
            if embedding_values is None:
                assert table is not None
                embedding_values = np.column_stack(
                    [table.column(column).to_numpy(zero_copy_only=False) for column in self.columns]
                ).astype(np.float32, copy=False)
                if raw_cache_path is not None:
                    self._write_raw_embedding_cache(raw_cache_path, embedding_values)
            if self.embedding_mean is not None:
                assert self.embedding_standard_deviation is not None
                embedding_values = (
                    embedding_values - self.embedding_mean
                ) / self.embedding_standard_deviation
            feature_parts.append(embedding_values)
        for column in self.auxiliary_columns:
            if column == "rank_fraction":
                row_count = (
                    embedding_values.shape[0] if embedding_values is not None else source_row_count
                )
                denominator = max(row_count - 1, 1)
                transformed = np.arange(row_count, dtype=np.float32) / denominator
            else:
                if column in cached_auxiliary_values:
                    transformed = cached_auxiliary_values[column]
                else:
                    assert table is not None
                    transformed = transform_auxiliary_values(
                        column,
                        table.column(column).to_numpy(zero_copy_only=False),
                    )
                    cached_auxiliary_values[column] = transformed
            feature_parts.append(transformed[:, None])
        if raw_auxiliary_cache_path is not None and not raw_auxiliary_cache_path.is_file():
            auxiliary_matrix = np.column_stack(
                [cached_auxiliary_values[column] for column in self.file_auxiliary_columns]
            ).astype(np.float32, copy=False)
            self._write_raw_embedding_cache(raw_auxiliary_cache_path, auxiliary_matrix)
        # The common raw-Z512 path has one already contiguous feature block.
        # Avoid an unconditional column_stack copy of every cell on every epoch.
        values = (
            feature_parts[0]
            if len(feature_parts) == 1
            else np.column_stack(feature_parts)
        ).astype(numpy_dtype, copy=False)
        tensor = torch.from_numpy(values)
        if self.slide_cache_size:
            self._slide_cache[cache_key] = tensor
            self._slide_cache.move_to_end(cache_key)
            while len(self._slide_cache) > self.slide_cache_size:
                self._slide_cache.popitem(last=False)
        return tensor

    def _immune_keep_mask(
        self, path: Path, expected_slide_id: str, expected_rows: int
    ) -> torch.Tensor:
        key = (path, expected_slide_id)
        if key in self._immune_keep_cache:
            keep = self._immune_keep_cache[key]
            if len(keep) != expected_rows:
                raise ValueError(f"cached immune mask row mismatch for {expected_slide_id}")
            return keep
        if self.immune_score_hard_filter_threshold is None:
            return torch.ones(expected_rows, dtype=torch.bool)
        try:
            table = pq.read_table(path, columns=[self.immune_score_column, "slide_id"], memory_map=True)
        except pa.ArrowInvalid as exc:
            raise ValueError(f"{path} lacks configured immune score column {self.immune_score_column!r}") from exc
        if table.num_rows != expected_rows:
            raise ValueError(f"immune score/embedding row mismatch for {expected_slide_id}")
        if pc.unique(table.column("slide_id")).to_pylist() != [expected_slide_id]:
            raise ValueError(f"immune source slide identity mismatch for {expected_slide_id}")
        values = table.column(self.immune_score_column).to_numpy(zero_copy_only=False)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"immune scores must be finite probabilities in [0, 1]: {path}")
        keep = torch.from_numpy(values < self.immune_score_hard_filter_threshold)
        self._immune_keep_cache[key] = keep
        return keep

    def _marker_keep_mask(
        self, path: Path, expected_slide_id: str, expected_rows: int
    ) -> torch.Tensor:
        cache_key = (path, expected_slide_id)
        cached = self._marker_keep_cache.get(cache_key)
        if cached is not None:
            if len(cached) != expected_rows:
                raise ValueError(f"cached marker mask row mismatch for {expected_slide_id}")
            return cached
        if self.marker_mean_hard_filter_threshold is None:
            return torch.ones(expected_rows, dtype=torch.bool)
        try:
            table = pq.read_table(
                path,
                columns=[*self.marker_mean_hard_filter_columns, "slide_id"],
                memory_map=True,
            )
        except pa.ArrowInvalid as exc:
            raise ValueError(
                f"{path} lacks configured marker mean columns "
                f"{self.marker_mean_hard_filter_columns!r}"
            ) from exc
        if table.num_rows != expected_rows:
            raise ValueError(
                f"marker means contain {table.num_rows} rows but embeddings contain "
                f"{expected_rows} for {expected_slide_id}"
            )
        slide_ids = pc.unique(table.column("slide_id")).to_pylist()
        if slide_ids != [expected_slide_id]:
            raise ValueError(
                f"marker source contains slide IDs {slide_ids!r}; expected {expected_slide_id!r}"
            )
        marker_values = np.column_stack(
            [
                table.column(column).to_numpy(zero_copy_only=False)
                for column in self.marker_mean_hard_filter_columns
            ]
        )
        if not np.isfinite(marker_values).all():
            raise ValueError(f"non-finite marker means in {path}")
        # A DAPI-only event has every non-DAPI marker below the threshold.
        keep = torch.from_numpy(
            np.any(marker_values >= self.marker_mean_hard_filter_threshold, axis=1)
        )
        self._marker_keep_cache[cache_key] = keep
        return keep

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        selected_slides = tuple(zip(record.slide_ids, record.slide_paths))
        output_entity_id = record.entity_id
        if self.negative_entity_dynamic_slide_pairing and record.label < 0.5:
            entity_pool = self.negative_slide_pools[record.entity_id]
            cross_entity = (
                self.negative_cross_entity_pair_probability > 0.0
                and len(self.negative_entity_ids) > 1
                and float(torch.rand(())) < self.negative_cross_entity_pair_probability
            )
            if cross_entity:
                other_entities = tuple(
                    entity_id
                    for entity_id in self.negative_entity_ids
                    if entity_id != record.entity_id
                )
                other_entity = other_entities[int(torch.randint(len(other_entities), ()))]
                other_pool = self.negative_slide_pools[other_entity]
                selected_slides = (
                    entity_pool[int(torch.randint(len(entity_pool), ()))],
                    other_pool[int(torch.randint(len(other_pool), ()))],
                )
                output_entity_id = f"cross:{record.entity_id}+{other_entity}"
            else:
                selected_count = min(2, len(entity_pool))
                selected_indices = torch.randperm(len(entity_pool))[:selected_count].tolist()
                selected_slides = tuple(entity_pool[position] for position in selected_indices)
        slide_ids = tuple(slide_id for slide_id, _ in selected_slides)
        slide_paths = tuple(slide_path for _, slide_path in selected_slides)
        acquired_exposure = float(len(selected_slides))
        slides = [self._read_slide(path, slide_id) for slide_id, path in selected_slides]
        slide_row_indices = [torch.arange(slide.shape[0], dtype=torch.long) for slide in slides]
        marker_keep_slides: list[torch.Tensor | None] = (
            [
                self._marker_keep_mask(path, slide_id, slide.shape[0])
                for (slide_id, path), slide in zip(selected_slides, slides, strict=True)
            ]
            if self.marker_mean_hard_filter_threshold is not None
            else [None] * len(slides)
        )
        immune_keep_slides = (
            [self._immune_keep_mask(path, slide_id, slide.shape[0])
             for (slide_id, path), slide in zip(selected_slides, slides, strict=True)]
            if self.immune_score_hard_filter_threshold is not None else [None] * len(slides)
        )
        qc_slides = (
            [
                torch.from_numpy(
                    np.asarray(self.qc_score_cache.load(path, slide_id, slide.shape[0])).copy()
                )
                for (slide_id, path), slide in zip(selected_slides, slides, strict=True)
            ]
            if self.qc_score_cache is not None
            else None
        )
        original_cell_count = sum(slide.shape[0] for slide in slides)
        if (
            self.qc_hard_filter_threshold is not None
            or self.marker_mean_hard_filter_threshold is not None
            or self.immune_score_hard_filter_threshold is not None
        ):
            retained_slides: list[torch.Tensor] = []
            retained_qc_slides: list[torch.Tensor] = []
            retained_row_indices: list[torch.Tensor] = []
            retained_identifiers: list[tuple[str, Path]] = []
            for slide_index, (identifier, slide, row_indices, marker_keep, immune_keep) in enumerate(
                zip(selected_slides, slides, slide_row_indices, marker_keep_slides, immune_keep_slides, strict=True)
            ):
                keep = marker_keep
                if immune_keep is not None:
                    keep = immune_keep if keep is None else keep & immune_keep
                scores = qc_slides[slide_index] if qc_slides is not None else None
                if self.qc_hard_filter_threshold is not None:
                    assert scores is not None
                    qc_keep = scores <= self.qc_hard_filter_threshold
                    keep = qc_keep if keep is None else keep & qc_keep
                assert keep is not None
                if bool(keep.any()) or self.preserve_filtered_slide_exposure:
                    retained_identifiers.append(identifier)
                    retained_slides.append(slide[keep])
                    if scores is not None:
                        retained_qc_slides.append(scores[keep])
                    retained_row_indices.append(row_indices[keep])
            if not retained_slides:
                raise ValueError(
                    f"hard filtering removed every cell from group {record.group_id!r}"
                )
            selected_slides = tuple(retained_identifiers)
            slides = retained_slides
            qc_slides = retained_qc_slides if qc_slides is not None else None
            slide_row_indices = retained_row_indices
            slide_ids = tuple(slide_id for slide_id, _ in selected_slides)
            slide_paths = tuple(slide_path for _, slide_path in selected_slides)
        slide_lengths = torch.tensor([slide.shape[0] for slide in slides], dtype=torch.long)
        total_rows = int(slide_lengths.sum().item())
        features = torch.empty((total_rows, self.input_dim), dtype=self.feature_dtype)
        segment_ids = torch.empty(total_rows, dtype=torch.long)
        qc_unusable_scores = (
            torch.empty(total_rows, dtype=torch.float32) if qc_slides is not None else None
        )
        offset = 0
        for segment_id, slide in enumerate(slides):
            end = offset + slide.shape[0]
            features[offset:end].copy_(slide)
            segment_ids[offset:end] = segment_id
            if qc_unusable_scores is not None:
                assert qc_slides is not None
                qc_unusable_scores[offset:end].copy_(qc_slides[segment_id])
            offset = end
        return {
            "features": features,
            "attention_mask": torch.ones(total_rows, dtype=torch.bool),
            "segment_ids": segment_ids,
            "qc_unusable_scores": qc_unusable_scores,
            "label": torch.tensor(record.label, dtype=torch.float32),
            "sample_weight": torch.tensor(record.sample_weight, dtype=torch.float32),
            "group_id": record.group_id,
            "entity_id": output_entity_id,
            "sampling_stratum": record.sampling_stratum,
            "slide_ids": slide_ids,
            "slide_paths": tuple(str(path) for path in slide_paths),
            "slide_lengths": slide_lengths,
            "exposure_slide_equivalents": torch.tensor(acquired_exposure, dtype=torch.float32),
            "slide_row_indices": tuple(slide_row_indices),
            "qc_original_cell_count": original_cell_count,
            "qc_retained_cell_count": total_rows,
        }


def group_collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad only within a batch; attention_mask keeps padding invisible to the model."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    max_length = max(1, max(sample["features"].shape[0] for sample in samples))
    input_dim = samples[0]["features"].shape[1]
    features = torch.zeros((batch_size, max_length, input_dim), dtype=samples[0]["features"].dtype)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    segment_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    has_qc_scores = [sample["qc_unusable_scores"] is not None for sample in samples]
    if any(has_qc_scores) and not all(has_qc_scores):
        raise ValueError("cannot collate a mixture of QC-scored and unscored groups")
    qc_unusable_scores = (
        torch.zeros((batch_size, max_length), dtype=torch.float32) if all(has_qc_scores) else None
    )
    has_covariates = [sample.get("cell_covariates") is not None for sample in samples]
    if any(has_covariates) and not all(has_covariates):
        raise ValueError("cannot mix groups with and without cell covariates")
    cell_covariates = None
    if all(has_covariates):
        dimension = samples[0]["cell_covariates"].shape[-1]
        cell_covariates = torch.zeros(batch_size, max_length, dimension, dtype=torch.float32)
        for j, sample in enumerate(samples):
            values = sample["cell_covariates"]
            if values.shape != (len(sample["features"]), dimension):
                raise ValueError("cell covariate shape must align with retained cells")
            cell_covariates[j, :len(values)] = values
    for batch_index, sample in enumerate(samples):
        length = sample["features"].shape[0]
        features[batch_index, :length] = sample["features"]
        attention_mask[batch_index, :length] = True
        segment_ids[batch_index, :length] = sample["segment_ids"]
        if qc_unusable_scores is not None:
            qc_unusable_scores[batch_index, :length] = sample["qc_unusable_scores"]
    sample_weights = [
        torch.as_tensor(sample.get("sample_weight", 1.0), dtype=torch.float32)
        for sample in samples
    ]
    sampling_strata = [
        sample.get(
            "sampling_stratum",
            "control" if float(sample["label"].item()) < 0.5 else "other_cancer",
        )
        for sample in samples
    ]
    slide_row_indices = []
    for sample in samples:
        row_indices = sample.get("slide_row_indices")
        if row_indices is None:
            lengths = torch.as_tensor(sample["slide_lengths"]).tolist()
            row_indices = tuple(torch.arange(int(length), dtype=torch.long) for length in lengths)
        slide_row_indices.append(row_indices)
    return {
        "features": features,
        "cell_covariates": cell_covariates,
        "attention_mask": attention_mask,
        "segment_ids": segment_ids,
        "qc_unusable_scores": qc_unusable_scores,
        "label": torch.stack([sample["label"] for sample in samples]),
        "sample_weight": torch.stack(sample_weights),
        "group_id": [sample["group_id"] for sample in samples],
        "entity_id": [sample["entity_id"] for sample in samples],
        "sampling_stratum": sampling_strata,
        "slide_ids": [sample["slide_ids"] for sample in samples],
        "slide_paths": [sample["slide_paths"] for sample in samples],
        "slide_lengths": [sample["slide_lengths"] for sample in samples],
        "exposure_slide_equivalents": torch.stack([
            torch.as_tensor(sample.get("exposure_slide_equivalents", float(len(sample["slide_ids"]))), dtype=torch.float32)
            for sample in samples
        ]),
        # Preserve immutable source-row identity through padding. Attribution
        # exporters consume this nested structure directly rather than trying
        # to infer pre-QC row positions from the padded tensor.
        "slide_row_indices": slide_row_indices,
        "qc_original_cell_count": torch.tensor(
            [
                sample.get("qc_original_cell_count", sample["features"].shape[0])
                for sample in samples
            ],
            dtype=torch.long,
        ),
        "qc_retained_cell_count": torch.tensor(
            [
                sample.get("qc_retained_cell_count", sample["features"].shape[0])
                for sample in samples
            ],
            dtype=torch.long,
        ),
    }


def dataset_from_config(
    config: DataConfig,
    manifest_path: str | Path | None = None,
    negative_entity_dynamic_slide_pairing: bool = False,
    negative_cross_entity_pair_probability: float = 0.0,
    slide_cache_size: int | None = None,
) -> GroupedParquetDataset:
    selected_manifest = manifest_path if manifest_path is not None else config.manifest_path
    if selected_manifest is None:
        raise ValueError("a manifest path is required")
    LOGGER.info(
        "Creating grouped parquet dataset: manifest=%s input_dim=%d dtype=%s "
        "max_rows_per_slide=%s top_rows_per_slide=%s auxiliary_columns=%s "
        "embedding_indices=%s embedding_normalization_path=%s max_slides_per_group=%s "
        "slide_cache_size=%d raw_embedding_cache_dir=%s "
        "transformed_embedding_cache_dir=%s "
        "qc_score_cache_dir=%s "
        "marker_mean_hard_filter_threshold=%s marker_mean_hard_filter_columns=%s "
        "immune_score_hard_filter_threshold=%s immune_score_column=%s "
        "negative_entity_dynamic_slide_pairing=%s "
        "negative_cross_entity_pair_probability=%.3f",
        Path(selected_manifest).expanduser().resolve(),
        config.input_dim,
        config.feature_dtype,
        config.max_rows_per_slide,
        config.top_rows_per_slide,
        config.auxiliary_columns,
        config.embedding_indices,
        config.embedding_normalization_path,
        config.max_slides_per_group,
        config.slide_cache_size if slide_cache_size is None else slide_cache_size,
        config.raw_embedding_cache_dir,
        config.transformed_embedding_cache_dir,
        config.qc_score_cache_dir,
        config.marker_mean_hard_filter_threshold,
        config.marker_mean_hard_filter_columns,
        config.immune_score_hard_filter_threshold,
        config.immune_score_column,
        negative_entity_dynamic_slide_pairing,
        negative_cross_entity_pair_probability,
    )
    records = load_group_manifest(
        selected_manifest,
        source_roots=config.source_roots,
        slide_id_column=config.slide_id_column,
        slide_path_column=config.slide_path_column,
        group_id_column=config.group_id_column,
        label_column=config.label_column,
        entity_id_column=config.entity_id_column,
        sampling_stratum_column=config.sampling_stratum_column,
        sample_weight_column=config.sample_weight_column,
        allow_reused_slides=config.allow_reused_slides,
        max_slides_per_group=config.max_slides_per_group,
        canonicalize_entity_ids=config.canonicalize_entity_ids,
    )
    dataset = GroupedParquetDataset(
        records=records,
        preserve_filtered_slide_exposure=config.preserve_filtered_slide_exposure,
        input_dim=config.input_dim,
        max_rows_per_slide=config.max_rows_per_slide,
        top_rows_per_slide=config.top_rows_per_slide,
        embedding_indices=config.embedding_indices,
        embedding_normalization_path=config.embedding_normalization_path,
        auxiliary_columns=config.auxiliary_columns,
        feature_dtype=config.feature_dtype,
        slide_cache_size=(
            config.slide_cache_size if slide_cache_size is None else slide_cache_size
        ),
        raw_embedding_cache_dir=config.raw_embedding_cache_dir,
        require_raw_embedding_cache=config.require_raw_embedding_cache,
        transformed_embedding_cache_dir=config.transformed_embedding_cache_dir,
        qc_score_cache_dir=config.qc_score_cache_dir,
        qc_hard_filter_threshold=config.qc_hard_filter_threshold,
        immune_score_hard_filter_threshold=config.immune_score_hard_filter_threshold,
        immune_score_column=config.immune_score_column,
        marker_mean_hard_filter_threshold=config.marker_mean_hard_filter_threshold,
        marker_mean_hard_filter_columns=config.marker_mean_hard_filter_columns,
        negative_entity_dynamic_slide_pairing=(negative_entity_dynamic_slide_pairing),
        negative_cross_entity_pair_probability=(negative_cross_entity_pair_probability),
    )
    if config.cell_covariate_cache_dir is not None:
        from .cell_covariates import CellCovariateDataset
        dataset = CellCovariateDataset(dataset, config.cell_covariate_cache_dir)
    LOGGER.info("Dataset ready: groups=%d", len(dataset))
    return dataset


def make_dataloader(
    dataset: GroupedParquetDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    entity_balanced_sampling: bool = False,
    entity_class_balanced_sampling: bool = False,
    negative_entity_pair_sampling: bool = False,
    entity_structured_batch_sampling: bool = False,
    positive_entities_per_batch: int = 24,
    negative_entities_per_batch: int = 8,
    negative_entity_balance_exponent: float = 1.0,
    positive_stratum_balance_exponent: float = 0.0,
    entity_balance_exponent: float = 1.0,
    batch_schedule: Sequence[Sequence[Sequence[int]]] | None = None,
    batch_schedule_position: int = 0,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    sampler: Sampler[int] | None = None
    if (
        sum(
            (
                entity_balanced_sampling,
                entity_class_balanced_sampling,
                negative_entity_pair_sampling,
                entity_structured_batch_sampling,
            )
        )
        > 1
    ):
        raise ValueError("entity sampling modes are mutually exclusive")
    if not 0.0 <= entity_balance_exponent <= 1.0:
        raise ValueError("entity_balance_exponent must be in [0, 1]")
    if not 0.0 <= positive_stratum_balance_exponent <= 1.0:
        raise ValueError("positive_stratum_balance_exponent must be in [0, 1]")
    if not 0.0 <= negative_entity_balance_exponent <= 1.0:
        raise ValueError("negative_entity_balance_exponent must be in [0, 1]")
    batch_sampler: Sampler[list[int]] | None = None
    if batch_schedule is not None:
        if not shuffle:
            raise ValueError("materialized schedules are only valid for training loaders")
        batch_sampler = MaterializedBatchSampler(batch_schedule, batch_schedule_position)
    elif entity_structured_batch_sampling:
        if not shuffle:
            raise ValueError("structured entity batches require a shuffled loader")
        if batch_size != positive_entities_per_batch + negative_entities_per_batch:
            raise ValueError("batch size must match structured entity class counts")
        batch_sampler = EntityStructuredBatchSampler(
            dataset.records,
            positive_entities_per_batch,
            negative_entities_per_batch,
            seed,
            positive_stratum_balance_exponent=positive_stratum_balance_exponent,
        )
        LOGGER.info(
            "Structured entity batch sampler ready: batches=%d positives_per_batch=%d "
            "negatives_per_batch=%d positive_stratum_balance_exponent=%.3f",
            len(batch_sampler),
            positive_entities_per_batch,
            negative_entities_per_batch,
            positive_stratum_balance_exponent,
        )
    elif negative_entity_pair_sampling:
        if not shuffle:
            raise ValueError("negative entity-pair sampling requires a shuffled loader")
        if batch_size % 4:
            raise ValueError("negative entity-pair sampling requires batch_size divisible by four")
        sample_count = math.ceil(len(dataset) / batch_size) * batch_size
        sampler = NegativeEntityPairSampler(
            dataset.records,
            sample_count,
            seed,
            negative_entity_balance_exponent=negative_entity_balance_exponent,
            positive_stratum_balance_exponent=positive_stratum_balance_exponent,
        )
        LOGGER.info(
            "Negative entity-pair sampler ready: groups=%d sampled_groups=%d batch_size=%d "
            "negative_entity_balance_exponent=%.3f "
            "positive_stratum_balance_exponent=%.3f",
            len(dataset),
            sample_count,
            batch_size,
            negative_entity_balance_exponent,
            positive_stratum_balance_exponent,
        )
    elif entity_balanced_sampling or entity_class_balanced_sampling:
        if not shuffle:
            raise ValueError("entity-balanced sampling is only valid for a shuffled loader")
        class_counts = Counter(record.label for record in dataset.records)
        entity_counts = Counter((record.label, record.entity_id) for record in dataset.records)
        class_normalizers = {
            label: sum(
                count ** (1.0 - entity_balance_exponent)
                for (entity_label, _), count in entity_counts.items()
                if entity_label == label
            )
            for label in class_counts
        }
        if entity_class_balanced_sampling:
            weights = [
                entity_counts[(record.label, record.entity_id)] ** (-entity_balance_exponent)
                / class_normalizers[record.label]
                for record in dataset.records
            ]
        else:
            weights = [
                class_counts[record.label]
                * entity_counts[(record.label, record.entity_id)] ** (-entity_balance_exponent)
                / class_normalizers[record.label]
                for record in dataset.records
            ]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        LOGGER.info(
            "Entity-aware training sampler ready: groups=%d entities=%d classes=%d "
            "class_balanced=%s entity_balance_exponent=%.3f",
            len(dataset),
            len(entity_counts),
            len(class_counts),
            entity_class_balanced_sampling,
            entity_balance_exponent,
        )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "num_workers": num_workers,
        "collate_fn": group_collate,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "generator": generator,
    }
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs.update(
            {
                "batch_size": batch_size,
                "shuffle": shuffle if sampler is None else False,
                "sampler": sampler,
            }
        )
    loader = DataLoader(**loader_kwargs)
    LOGGER.info(
        "DataLoader ready: groups=%d batches=%d batch_size=%d shuffle=%s workers=%d "
        "entity_balanced_sampling=%s entity_class_balanced_sampling=%s "
        "negative_entity_pair_sampling=%s entity_structured_batch_sampling=%s "
        "negative_entity_balance_exponent=%.3f "
        "positive_stratum_balance_exponent=%.3f "
        "entity_balance_exponent=%.3f "
        "pin_memory=%s persistent_workers=%s",
        len(dataset),
        len(loader),
        batch_size,
        shuffle,
        num_workers,
        entity_balanced_sampling,
        entity_class_balanced_sampling,
        negative_entity_pair_sampling,
        entity_structured_batch_sampling,
        negative_entity_balance_exponent,
        positive_stratum_balance_exponent,
        entity_balance_exponent,
        torch.cuda.is_available(),
        num_workers > 0,
    )
    return loader


def load_metadata_for_positions(
    slide_paths: Sequence[str | Path],
    slide_ids: Sequence[str],
    slide_lengths: Sequence[int] | torch.Tensor,
    sequence_positions: Sequence[int],
    slide_row_indices: Sequence[Sequence[int] | torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    """Map selected group-sequence positions back to immutable parquet rows."""
    lengths = [int(value) for value in slide_lengths]
    offsets = np.cumsum([0, *lengths])
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    output: list[dict[str, Any] | None] = [None] * len(sequence_positions)
    for output_index, position in enumerate(sequence_positions):
        if position < 0 or position >= offsets[-1]:
            raise IndexError(f"sequence position {position} is outside [0, {offsets[-1]})")
        slide_index = int(np.searchsorted(offsets, position, side="right") - 1)
        local_row = int(position - offsets[slide_index])
        grouped[slide_index].append((output_index, local_row))

    for slide_index, requested in grouped.items():
        path = Path(slide_paths[slide_index])
        available = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [column for column in INSTANCE_METADATA_COLUMNS if column in available]
        table = pq.read_table(path, columns=columns, memory_map=True)
        source_rows = (
            [local_row for _, local_row in requested]
            if slide_row_indices is None
            else [int(slide_row_indices[slide_index][local_row]) for _, local_row in requested]
        )
        indices = pa.array(source_rows, type=pa.int64())
        rows = table.take(indices).to_pylist()
        for (output_index, _), source_row, metadata in zip(
            requested, source_rows, rows, strict=True
        ):
            output[output_index] = {
                "slide_id": slide_ids[slide_index],
                "slide_path": str(path),
                "slide_index": slide_index,
                "slide_row_index": source_row,
                **metadata,
            }
    return [row for row in output if row is not None]
