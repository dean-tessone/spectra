"""Explicit, reproducible intensity transforms for reviewing BLUE source cells."""

from __future__ import annotations

import numpy as np

CHANNELS = ("DAPI", "TRITC", "CY5", "FITC")


def acquisition_gains(text: str) -> np.ndarray:
    """Read enabled channel gains from slideinfo; never guess missing values."""
    fields = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key.strip().upper() in CHANNELS:
                key = key.strip().upper()
                if key in fields:
                    raise ValueError(f"duplicate acquisition channel: {key}")
                fields[key] = value
    gains = []
    for channel in CHANNELS:
        values = [float(x.strip()) for x in fields[channel].split(",")]
        if len(values) != 3 or values[0] != 1:
            raise ValueError(f"missing or disabled acquisition channel: {channel}")
        gains.append(values[2])
    result = np.asarray(gains, dtype=np.float64)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("acquisition gains must be finite and positive")
    return result


def apply_gains(images: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Match BLUE: multiply pixels, clip, truncate to uint16 BEFORE averaging."""
    gains = np.asarray(gains, dtype=np.float64)
    if images.dtype != np.uint16 or images.shape[-1] != 4:
        raise ValueError("expected uint16 crops with DAPI/TRITC/CY5/FITC channels")
    if gains.shape != (4,) or not np.isfinite(gains).all() or (gains <= 0).any():
        raise ValueError("expected four finite positive gains")
    return np.clip(images.astype(np.float64) * gains, 0, 65535).astype(np.uint16)


def segmented_means(images: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != images.shape[:2] or mask.dtype != bool or not mask.any():
        raise ValueError("nonempty boolean cell mask must align with crop")
    return images[mask].mean(axis=0).astype(np.uint16)
