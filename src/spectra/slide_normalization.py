"""Label-free normalization of cell features within acquisition slides."""

from __future__ import annotations

import torch


def _blocked_slide_sums(
    values: torch.Tensor,
    membership: torch.Tensor,
    block_cells: int = 4_096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate slide sums without a single long CUDA reduction."""
    if block_cells < 1:
        raise ValueError("block_cells must be positive")
    batch_size, _, dimensions = values.shape
    slides = membership.shape[-1]
    sums = torch.zeros(
        (batch_size, slides, dimensions), device=values.device, dtype=torch.float32
    )
    counts = torch.zeros(
        (batch_size, slides), device=values.device, dtype=torch.float32
    )
    for start in range(0, values.shape[1], block_cells):
        stop = min(start + block_cells, values.shape[1])
        block_membership = membership[:, start:stop]
        sums = sums + torch.bmm(
            block_membership.transpose(1, 2), values[:, start:stop]
        )
        counts = counts + block_membership.sum(dim=1)
    return sums, counts


def _segmented_slide_sums(
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    max_slides: int,
    square_values: bool = False,
    block_cells: int = 1_024,
) -> torch.Tensor:
    """Accumulate values by slide with bounded ordinary reductions.

    Some CUDA/cuBLAS versions can sporadically return non-finite results for
    the highly rectangular batched matrix multiplication used above when its
    right operand contains squared cell features.  CUDA atomic segmented-add
    kernels have shown the same problem on affected hardware.  Boolean
    selection followed by bounded ordinary sums is less clever, but each cell
    is reduced exactly once and it avoids both unstable kernel paths.
    """
    if block_cells < 1:
        raise ValueError("block_cells must be positive")
    batch_sums: list[torch.Tensor] = []
    for batch_index in range(values.shape[0]):
        slide_sums: list[torch.Tensor] = []
        for slide_index in range(max_slides):
            selected = values[
                batch_index,
                attention_mask[batch_index]
                & (segment_ids[batch_index] == slide_index),
            ]
            if square_values:
                selected = selected.square()
            slide_sum = torch.zeros(
                values.shape[-1], device=values.device, dtype=values.dtype
            )
            for start in range(0, selected.shape[0], block_cells):
                slide_sum = slide_sum + selected[start : start + block_cells].sum(dim=0)
            slide_sums.append(slide_sum)
        batch_sums.append(torch.stack(slide_sums))
    return torch.stack(batch_sums)


def _apply_segment_scale(
    centered: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    slide_variances: torch.Tensor,
) -> torch.Tensor:
    """Scale one physical slide at a time to bound CUDA kernel sizes."""
    scaled = torch.zeros_like(centered)
    for batch_index in range(centered.shape[0]):
        for slide_index in range(slide_variances.shape[1]):
            selected = attention_mask[batch_index] & (
                segment_ids[batch_index] == slide_index
            )
            scaled[batch_index, selected] = centered[
                batch_index, selected
            ] * torch.rsqrt(slide_variances[batch_index, slide_index])
    return scaled


def slide_relative_normalize(
    features: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    max_slides: int,
    scale_strength: float = 0.0,
    minimum_variance: float = 1e-4,
) -> torch.Tensor:
    """Center and optionally shrinkage-scale every slide independently.

    Fold-standardized inputs have unit feature variance in expectation.  The
    scale denominator interpolates in variance space between that training
    reference and the observed slide variance.  ``scale_strength=0`` is exact
    slide centering; ``scale_strength=1`` is per-slide standardization.

    This operation uses no labels or categorical batch identifier, so it also
    applies to staining batches absent during training.  All reductions are in
    FP32 for long bags and padded cells remain exactly zero.
    """
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, length, dimensions]")
    batch_size, sequence_length, dimension = features.shape
    if attention_mask.shape != (batch_size, sequence_length):
        raise ValueError("attention_mask shape does not match features")
    if segment_ids.shape != (batch_size, sequence_length):
        raise ValueError("segment_ids shape does not match features")
    if not bool(attention_mask.any(dim=1).all()):
        raise ValueError("every group must contain at least one unmasked cell")
    if max_slides < 1:
        raise ValueError("max_slides must be positive")
    if not 0.0 <= scale_strength <= 1.0:
        raise ValueError("scale_strength must be in [0, 1]")
    if minimum_variance <= 0:
        raise ValueError("minimum_variance must be positive")
    valid_segments = segment_ids[attention_mask]
    if int(valid_segments.min()) < 0 or int(valid_segments.max()) >= max_slides:
        raise ValueError("valid segment IDs are outside the configured slide range")

    slide_indices = torch.arange(max_slides, device=features.device).view(1, 1, -1)
    membership = (
        (segment_ids.unsqueeze(-1) == slide_indices) & attention_mask.unsqueeze(-1)
    ).float()
    values = features.float()
    # The previously used blocked BMM still presents a highly rectangular
    # CUDA kernel for long cell bags and can sporadically return a non-finite
    # slide mean on Blackwell.  Explicit per-slide bounded reductions are
    # slower in theory but stable and a small fraction of end-to-end I/O time.
    sums = _segmented_slide_sums(
        values,
        attention_mask,
        segment_ids,
        max_slides=max_slides,
    )
    counts = membership.sum(dim=1)
    means = sums / counts.clamp_min(1.0).unsqueeze(-1)
    gathered_means = means.gather(
        1, segment_ids.unsqueeze(-1).expand(-1, -1, dimension)
    )
    centered = values - gathered_means

    if scale_strength:
        squared_sums = _segmented_slide_sums(
            centered,
            attention_mask,
            segment_ids,
            max_slides=max_slides,
            square_values=True,
        )
        variances = squared_sums / counts.clamp_min(1.0).unsqueeze(-1)
        shrunk_variances = (
            (1.0 - scale_strength) + scale_strength * variances
        ).clamp_min(minimum_variance)
        centered = _apply_segment_scale(
            centered,
            attention_mask,
            segment_ids,
            shrunk_variances,
        )

    return centered.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
