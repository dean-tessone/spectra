"""Strict source-aligned cell measurement caches and equal-patient normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .data import embedding_cache_path, transform_auxiliary_values

COLUMNS = ("DAPI_mean", "TRITC_mean", "FITC_mean", "CY5_mean", "area", "eccentricity")


def cache_path(root, source, slide):
    return embedding_cache_path(root, source, slide, COLUMNS).with_suffix(".npz")


def prepare_cache(root, source, slide):
    path = cache_path(root, source, slide)
    if path.exists():
        load_cache(root, source, slide)
        return path
    source = Path(source).resolve(strict=True)
    table = pq.read_table(source, columns=["slide_id", "cell_id", "frame_id", *COLUMNS]).to_pandas()
    if set(table.slide_id.astype(str)) != {str(slide)}:
        raise ValueError("covariate source slide identity mismatch")
    raw = table[list(COLUMNS)].to_numpy(dtype=np.float32)
    if (
        not np.isfinite(raw).all()
        or (raw[:, :5] < 0).any()
        or not ((raw[:, 5] >= 0) & (raw[:, 5] <= 1)).all()
    ):
        raise ValueError(f"invalid cell measurements: {slide}")
    values = np.column_stack(
        [transform_auxiliary_values(c, raw[:, i]) for i, c in enumerate(COLUMNS)]
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        np.savez(
            handle,
            values=values,
            columns=np.asarray(COLUMNS),
            slide_id=str(slide),
            source=str(source),
            row_indices=np.arange(len(values)),
            cell_id=table.cell_id.astype(str).to_numpy(dtype=str),
            frame_id=table.frame_id.astype(str).to_numpy(dtype=str),
        )
    temp.replace(path)
    return path


def load_cache(root, source, slide):
    path = cache_path(root, source, slide)
    if not path.exists():
        raise FileNotFoundError(f"required local cell covariate cache missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        values = payload["values"]
        if (
            tuple(payload["columns"]) != COLUMNS
            or str(payload["slide_id"]) != str(slide)
            or str(payload["source"]) != str(Path(source).resolve(strict=True))
            or values.ndim != 2
            or values.shape[1] != len(COLUMNS)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
            or not np.array_equal(payload["row_indices"], np.arange(len(values)))
            or len(payload["cell_id"]) != len(values)
            or len(payload["frame_id"]) != len(values)
        ):
            raise ValueError(f"cell covariate cache alignment/values invalid: {path}")
    return values


class CellCovariateDataset(Dataset):
    def __init__(self, base, root):
        self.base = base
        self.root = str(Path(root).resolve(strict=True))
        self._covariates = {}
        for record in base.records:
            for slide, path in zip(record.slide_ids, record.slide_paths, strict=True):
                key = (str(slide), str(path))
                if key not in self._covariates:
                    values = load_cache(root, path, slide)
                    # Raw row-count validation happens before QC/immune filtering.
                    if len(values) != pq.read_metadata(path).num_rows:
                        raise ValueError("covariate/source row count mismatch")
                    self._covariates[key] = torch.from_numpy(values)

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        pieces = []
        for slide, path, indices in zip(
            sample["slide_ids"], sample["slide_paths"], sample["slide_row_indices"], strict=True
        ):
            values = self._covariates[(str(slide), str(path))]
            indices = torch.as_tensor(indices, dtype=torch.long)
            if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= len(values)):
                raise ValueError("cell covariate source row out of bounds")
            pieces.append(values[indices])
        sample["cell_covariates"] = torch.cat(pieces) if pieces else torch.empty(0, len(COLUMNS))
        if len(sample["cell_covariates"]) != len(sample["features"]):
            raise ValueError("cell covariates and retained embeddings do not align")
        return sample


def fit_normalizer(dataset, output):
    means, seconds, patients = [], [], []
    for i, record in enumerate(dataset.records):
        if record.entity_id in patients:
            raise ValueError("normalization requires one joint bag per patient")
        x = dataset[i]["cell_covariates"].double().numpy()
        if len(x) == 0:
            raise ValueError("cannot estimate patient normalization from an empty bag")
        means.append(x.mean(0))
        seconds.append(np.square(x).mean(0))
        patients.append(record.entity_id)
    center = np.mean(means, axis=0)
    scale = np.sqrt(np.maximum(np.mean(seconds, axis=0) - center**2, 1e-12))
    np.savez(
        output,
        center=center.astype(np.float32),
        scale=scale.astype(np.float32),
        columns=np.asarray(COLUMNS),
        patients=np.asarray(patients),
        metadata=json.dumps({"weighting": "equal patients, equal cells within patient"}),
    )
