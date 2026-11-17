"""Quality-aware ungated ABMIL for grouped cell embeddings."""

from __future__ import annotations

import math
from typing import Any, NotRequired, TypedDict

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .slide_normalization import slide_relative_normalize


class ModelOutput(TypedDict):
    logits: torch.Tensor
    cls_embedding: torch.Tensor
    cross_attentions: tuple[torch.Tensor, ...]
    instance_logits: NotRequired[torch.Tensor]
    cell_contributions: NotRequired[torch.Tensor]
    contribution_intercept: NotRequired[torch.Tensor]
    content_attention_logits: NotRequired[torch.Tensor]
    qc_attention_penalty: NotRequired[torch.Tensor]
    attention_support: NotRequired[torch.Tensor]
    attention_weights: NotRequired[torch.Tensor]
    clam_instance_logits: NotRequired[torch.Tensor]
    class_logits: NotRequired[torch.Tensor]
    class_attention_weights: NotRequired[torch.Tensor]
    class_pooling_weights: NotRequired[torch.Tensor]
    class_cell_contributions: NotRequired[torch.Tensor]
    class_contribution_intercepts: NotRequired[torch.Tensor]
    class_clam_instance_logits: NotRequired[torch.Tensor]
    pooling_weights: NotRequired[torch.Tensor]
    attention_mass: NotRequired[torch.Tensor]
    attention_fraction: NotRequired[torch.Tensor]
    attention_support_count: NotRequired[torch.Tensor]
    frequency_attention_logits: NotRequired[torch.Tensor]
    frequency_gates: NotRequired[torch.Tensor]
    attention_entmax_alpha: NotRequired[torch.Tensor]
    cell_predictor_scores: NotRequired[torch.Tensor]
    head_attention_weights: NotRequired[torch.Tensor]
    head_cell_contributions: NotRequired[torch.Tensor]
    head_cell_predictor_scores: NotRequired[torch.Tensor]
    head_contribution_intercepts: NotRequired[torch.Tensor]
    head_logits: NotRequired[torch.Tensor]
    component_logits: NotRequired[dict[str, torch.Tensor]]
    parent_logits: NotRequired[torch.Tensor]
    burden_logits: NotRequired[torch.Tensor]
    parent_cell_contributions: NotRequired[torch.Tensor]
    burden_cell_contributions: NotRequired[torch.Tensor]
    parent_intercept: NotRequired[torch.Tensor]
    burden_intercept: NotRequired[torch.Tensor]
    burden_activity: NotRequired[torch.Tensor]
    burden_activity_sum: NotRequired[torch.Tensor]
    burden_activity_rate: NotRequired[torch.Tensor]
    burden_selected_cells: NotRequired[torch.Tensor]
    burden_union_activity: NotRequired[torch.Tensor]
    burden_features: NotRequired[torch.Tensor]
    burden_local_evidence: NotRequired[torch.Tensor]
    exposure_slide_equivalents: NotRequired[torch.Tensor]


class FixedFeatureStandardization(nn.Module):
    """Persisted training-fold standardization that never tracks validation data."""

    def __init__(self, mean: torch.Tensor, standard_deviation: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("standard_deviation", standard_deviation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.mean) / self.standard_deviation


def _blocked_feature_standardize(
    inputs: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    block_cells: int = 4_096,
) -> torch.Tensor:
    """Apply a fixed affine transform without one enormous CUDA kernel."""
    if block_cells < 1:
        raise ValueError("block_cells must be positive")
    output = torch.empty(inputs.shape, dtype=torch.float32, device=inputs.device)
    for start in range(0, inputs.shape[1], block_cells):
        stop = min(start + block_cells, inputs.shape[1])
        output[:, start:stop] = (inputs[:, start:stop].float() - center) / scale
    return output


class FixedBatchResidualization(nn.Module):
    """Remove a fold-local nuisance offset without erasing within-slide biology."""

    def __init__(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
        basis: torch.Tensor,
        strength: float,
        max_slides: int,
        standardize_input: bool,
    ) -> None:
        super().__init__()
        if basis.ndim != 2 or basis.shape[0] < 1:
            raise ValueError("batch residualization basis must be a feature matrix")
        if center.shape != (basis.shape[0],) or scale.shape != center.shape:
            raise ValueError("batch residualization center and scale have the wrong shape")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("batch residualization center and scale must be finite")
        if bool((scale <= 0).any()):
            raise ValueError("batch residualization scale must be positive")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("batch residualization strength must be in [0, 1]")
        if max_slides < 1:
            raise ValueError("batch residualization max_slides must be positive")
        basis = basis.detach().to(dtype=torch.float32)
        if not torch.isfinite(basis).all():
            raise ValueError("batch residualization basis must be finite")
        if basis.shape[1] and not torch.allclose(
            basis.T @ basis,
            torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError("batch residualization basis must be orthonormal")
        self.register_buffer("center", center.detach().to(dtype=torch.float32))
        self.register_buffer("scale", scale.detach().to(dtype=torch.float32))
        self.register_buffer("basis", basis)
        self.register_buffer("strength", torch.tensor(float(strength), dtype=torch.float32))
        self.max_slides = max_slides
        self.standardize_input = standardize_input

    @property
    def feature_count(self) -> int:
        return int(self.basis.shape[0])

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("batch residualization expects [batch, length, features]")
        if inputs.shape[-1] < self.feature_count:
            raise ValueError("input has fewer dimensions than the nuisance basis")
        batch_size, sequence_length, _ = inputs.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length), device=inputs.device, dtype=torch.bool
            )
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("batch residualization attention mask has the wrong shape")
        if segment_ids is None:
            segment_ids = torch.zeros(
                (batch_size, sequence_length), device=inputs.device, dtype=torch.long
            )
        if segment_ids.shape != (batch_size, sequence_length):
            raise ValueError("batch residualization segment IDs have the wrong shape")
        if not segment_ids.is_cuda and (
            int(segment_ids.max()) >= self.max_slides or int(segment_ids.min()) < 0
        ):
            raise ValueError("batch residualization segment ID is out of range")

        raw_prefix = inputs[..., : self.feature_count]
        if self.standardize_input:
            prefix = _blocked_feature_standardize(
                raw_prefix,
                self.center,
                self.scale,
            )
        else:
            prefix = raw_prefix.float()
        basis = self.basis
        mask = attention_mask.unsqueeze(-1)
        if basis.shape[1]:
            slide_indices = torch.arange(self.max_slides, device=inputs.device).view(1, 1, -1)
            membership = (
                (segment_ids.unsqueeze(-1) == slide_indices) & attention_mask.unsqueeze(-1)
            ).to(dtype=prefix.dtype)
            sums = torch.bmm(membership.transpose(1, 2), prefix)
            counts = membership.sum(dim=1)
            means = sums / counts.clamp_min(1).unsqueeze(-1)
            projected_offsets = (means @ basis) @ basis.T
            scatter_indices = segment_ids.unsqueeze(-1).expand(-1, -1, self.feature_count)
            cell_offsets = projected_offsets.gather(1, scatter_indices)
            corrected = prefix - self.strength * cell_offsets
        else:
            # A zero-component artifact is used to carry fold-local input
            # standardization. Do not launch a large slide reduction whose
            # result would be discarded.
            corrected = prefix
        # Keep standardized values in FP32.  Casting a legitimate tail value
        # back to a float16 loader dtype can silently create an infinity.
        corrected = corrected.masked_fill(~mask, 0)
        if inputs.shape[-1] == self.feature_count:
            return corrected
        return torch.cat((corrected, inputs[..., self.feature_count :].float()), dim=-1)


class ABMILAttention(nn.Module):
    """Canonical Ilse et al. gated or ungated instance-attention scorer."""

    def __init__(self, instance_dim: int, attention_dim: int, gated: bool) -> None:
        super().__init__()
        if attention_dim < 1:
            raise ValueError("abmil_attention_dim must be positive")
        self.gated = gated
        self.value_projection = nn.Linear(instance_dim, attention_dim)
        self.gate_projection = nn.Linear(instance_dim, attention_dim) if gated else None
        self.score_projection = nn.Linear(attention_dim, 1)

    def forward(self, instances: torch.Tensor) -> torch.Tensor:
        values = torch.tanh(_stable_cellwise_linear(self.value_projection, instances))
        if self.gate_projection is not None:
            values = values * torch.sigmoid(
                _stable_cellwise_linear(self.gate_projection, instances)
            )
        return _stable_cellwise_linear(self.score_projection, values).squeeze(-1)


class _StableCellwiseLinearFunction(torch.autograd.Function):
    """Linear map with a bounded-reduction CUDA weight gradient."""

    @staticmethod
    def forward(
        ctx: object,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.save_for_backward(inputs, weight)
        ctx.has_bias = bias is not None
        # Very large [batch, cells, features] GEMMs have also produced rare
        # non-finite *forward* blocks on current Blackwell/CUDA builds. Smaller
        # independent GEMMs avoid that kernel path without changing the model.
        block_rows = 8_192
        return torch.cat(
            [
                F.linear(inputs[:, start : start + block_rows], weight, bias)
                for start in range(0, inputs.shape[1], block_rows)
            ],
            dim=1,
        )

    @staticmethod
    def backward(
        ctx: object,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        inputs, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        block_rows = 1_024
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            # Bound this GEMM too.  The phenotype detector has a small output
            # dimension but its [batch, 100k cells, heads] backward input is
            # still large enough to hit the same unreliable CUDA kernel class
            # as the weight-gradient reduction below.
            input_gradient_blocks: list[torch.Tensor] = []
            for start in range(0, grad_output.shape[1], block_rows):
                stop = min(start + block_rows, grad_output.shape[1])
                block = torch.matmul(grad_output[:, start:stop], weight)
                if not bool(torch.isfinite(block).all()):
                    if not bool(torch.isfinite(grad_output[:, start:stop]).all()) or not bool(
                        torch.isfinite(weight).all()
                    ):
                        raise FloatingPointError(
                            "cellwise linear received non-finite grad-input operands"
                        )
                    block = torch.matmul(grad_output[:, start:stop].double(), weight.double()).to(
                        dtype=grad_output.dtype
                    )
                    if not bool(torch.isfinite(block).all()):
                        raise FloatingPointError(
                            "cellwise linear input gradient remained non-finite "
                            "after FP64 recomputation"
                        )
                input_gradient_blocks.append(block)
            grad_inputs = torch.cat(input_gradient_blocks, dim=1)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            # A single GEMM over hundreds of thousands of cells can produce an
            # invalid weight gradient on current Blackwell/CUDA builds despite
            # finite O(1e-5) upstream gradients. Batched outer products over
            # bounded sequence blocks are algebraically identical.
            grad_weight = torch.zeros_like(weight)
            for start in range(0, inputs.shape[1], block_rows):
                stop = min(start + block_rows, inputs.shape[1])
                block_inputs = inputs[:, start:stop].reshape(-1, inputs.shape[-1])
                block_gradients = grad_output[:, start:stop].reshape(-1, grad_output.shape[-1])
                block = block_gradients.T @ block_inputs
                grad_weight.add_(block.to(dtype=grad_weight.dtype))
            if not bool(torch.isfinite(grad_weight).all()):
                # Rare Blackwell GEMM failures have returned an invalid FP32
                # reduction despite finite inputs and upstream gradients. Pay
                # for an FP64 recomputation only on that exceptional path; do
                # not silently sanitize a genuinely invalid gradient.
                inputs_finite = bool(torch.isfinite(inputs).all())
                grad_output_finite = bool(torch.isfinite(grad_output).all())
                if not inputs_finite or not grad_output_finite:
                    raise FloatingPointError(
                        "cellwise linear received non-finite backward operands; "
                        f"input_shape={tuple(inputs.shape)} "
                        f"grad_output_shape={tuple(grad_output.shape)} "
                        f"inputs_finite={inputs_finite} "
                        f"grad_output_finite={grad_output_finite}"
                    )
                grad_weight.zero_()
                for start in range(0, inputs.shape[1], block_rows):
                    stop = min(start + block_rows, inputs.shape[1])
                    block_inputs = inputs[:, start:stop].reshape(-1, inputs.shape[-1])
                    block_gradients = grad_output[:, start:stop].reshape(-1, grad_output.shape[-1])
                    block = block_gradients.T.double() @ block_inputs.double()
                    grad_weight.add_(block.to(dtype=grad_weight.dtype))
                if not bool(torch.isfinite(grad_weight).all()):
                    raise FloatingPointError(
                        "cellwise linear weight gradient remained non-finite "
                        "after FP64 recomputation"
                    )
        grad_bias = None
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = torch.zeros(
                grad_output.shape[-1],
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
            for start in range(0, grad_output.shape[1], block_rows):
                grad_bias.add_(grad_output[:, start : start + block_rows].sum(dim=(0, 1)))
            if not bool(torch.isfinite(grad_bias).all()):
                raise FloatingPointError("cellwise linear produced a non-finite bias gradient")
        return grad_inputs, grad_weight, grad_bias


def _stable_cellwise_linear(layer: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    if not inputs.is_cuda:
        return layer(inputs)
    return _StableCellwiseLinearFunction.apply(inputs, layer.weight, layer.bias)


def _stable_cellwise_sigmoid(inputs: torch.Tensor, block_rows: int = 8_192) -> torch.Tensor:
    """Apply sigmoid in bounded cell blocks for very long padded bags."""
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    if not inputs.is_cuda or inputs.ndim < 3:
        return torch.sigmoid(inputs)
    return torch.cat(
        [
            torch.sigmoid(inputs[:, start : start + block_rows])
            for start in range(0, inputs.shape[1], block_rows)
        ],
        dim=1,
    )


class VanillaGatedABMIL(nn.Module):
    """Minimal Ilse-style gated ABMIL with no feature or bag normalization.

    The complete learned path is ``Linear -> GELU -> gated attention ->
    softmax -> weighted sum -> Linear``. QC is handled only by the upstream
    hard filter; slide identifiers, QC scores, additive heads, residualizers,
    and normalization layers never enter this model.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.architecture != "abmil_vanilla":
            raise ValueError("VanillaGatedABMIL requires architecture='abmil_vanilla'")
        incompatible = {
            "abmil_input_activation": config.abmil_input_activation != "gelu",
            "dropout": config.dropout != 0.0,
            "abmil_attention_normalization": config.abmil_attention_normalization != "softmax",
            "abmil_stochastic_topk": config.abmil_stochastic_topk != 0,
            "abmil_stochastic_topk_drop_probability": (
                config.abmil_stochastic_topk_drop_probability != 0.0
            ),
            "abmil_additive_pooling": config.abmil_additive_pooling,
            "clam_instance_loss_weight": config.clam_instance_loss_weight != 0.0,
            "clam_bag_loss_weight": config.clam_bag_loss_weight is not None,
            "abmil_slide_center_input": config.abmil_slide_center_input,
            "abmil_slide_scale_strength": config.abmil_slide_scale_strength != 0.0,
            "qc_max_suppression": config.qc_max_suppression != 1.0,
            "batch_residualization_path": config.batch_residualization_path is not None,
            "batch_residualization_components": config.batch_residualization_components != 0,
            "batch_residualization_standardize_input": (
                config.batch_residualization_standardize_input
            ),
        }
        enabled = sorted(name for name, present in incompatible.items() if present)
        if enabled:
            raise ValueError("vanilla ABMIL forbids extra mechanisms: " + ", ".join(enabled))
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        self.input_activation = nn.GELU()
        self.attention_pool = ABMILAttention(
            instance_dim=config.model_dim,
            attention_dim=config.abmil_attention_dim,
            gated=True,
        )
        self.classifier = nn.Linear(config.model_dim, 1)

    @property
    def embedding_dim(self) -> int:
        return self.config.model_dim

    @staticmethod
    def _require_finite(values: torch.Tensor, stage: str) -> None:
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"vanilla ABMIL produced non-finite values after {stage}")

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        return_attention: bool = False,
        qc_unusable_scores: torch.Tensor | None = None,
    ) -> ModelOutput:
        del qc_unusable_scores
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError(f"features must have shape [batch, length, {self.config.input_dim}]")
        batch_size, sequence_length, _ = features.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length), device=features.device, dtype=torch.bool
            )
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask does not align with features")
        if not bool(attention_mask.any(dim=-1).all()):
            raise ValueError("every group must contain at least one unmasked instance")
        if segment_ids is not None and segment_ids.shape != attention_mask.shape:
            raise ValueError("segment_ids does not align with features")

        if not torch.is_autocast_enabled() and features.dtype != self.input_projection.weight.dtype:
            features = features.to(self.input_projection.weight.dtype)
        instances = self.input_activation(_stable_cellwise_linear(self.input_projection, features))
        self._require_finite(instances, "instance projection")
        content_attention_logits = self.attention_pool(instances).float()
        masked_attention_logits = content_attention_logits.masked_fill(
            ~attention_mask, torch.finfo(content_attention_logits.dtype).min
        )
        pooling_weights = torch.softmax(masked_attention_logits, dim=-1).masked_fill(
            ~attention_mask, 0.0
        )
        self._require_finite(pooling_weights, "attention softmax")
        bag_embedding = (pooling_weights.unsqueeze(-1) * instances.float()).sum(dim=1)
        self._require_finite(bag_embedding, "weighted-sum pooling")
        logits = self.classifier(bag_embedding).squeeze(-1)
        self._require_finite(logits, "classification")

        classifier_weight = self.classifier.weight[0].float()
        cell_contributions = pooling_weights * (instances.float() * classifier_weight).sum(dim=-1)
        cell_contributions = cell_contributions.masked_fill(~attention_mask, 0.0)
        contribution_intercept = self.classifier.bias[0].float().expand(batch_size)
        attention_support = (pooling_weights > 0.0) & attention_mask
        valid_count = attention_mask.sum(dim=-1).clamp_min(1).float()
        attention_mass = pooling_weights.sum(dim=-1)
        zero_cell_values = torch.zeros_like(pooling_weights)
        return {
            "logits": logits,
            "cls_embedding": bag_embedding,
            "cross_attentions": (pooling_weights[:, None, None, :],) if return_attention else (),
            "instance_logits": cell_contributions,
            "cell_contributions": cell_contributions,
            "contribution_intercept": contribution_intercept,
            "content_attention_logits": content_attention_logits,
            "qc_attention_penalty": zero_cell_values,
            "attention_support": attention_support,
            "attention_weights": pooling_weights,
            "clam_instance_logits": zero_cell_values,
            "pooling_weights": pooling_weights,
            "attention_mass": attention_mass,
            "attention_fraction": attention_mass / valid_count,
            "attention_support_count": attention_support.sum(dim=-1).float(),
            "frequency_attention_logits": content_attention_logits,
            "frequency_gates": pooling_weights,
        }


def _sparsemax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact sparsemax with FP32 support selection for mixed-precision training."""
    original_dtype = values.dtype
    values = values.float()
    shifted = values - values.amax(dim=dim, keepdim=True)
    ordered = shifted.sort(dim=dim, descending=True).values
    cumulative = ordered.cumsum(dim)
    support_index = torch.arange(
        1,
        ordered.shape[dim] + 1,
        device=values.device,
        dtype=values.dtype,
    )
    shape = [1] * values.ndim
    shape[dim] = -1
    support_index = support_index.view(shape)
    support = 1 + support_index * ordered > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = (cumulative.gather(dim, support_size - 1) - 1) / support_size.to(values.dtype)
    return (shifted - threshold).clamp_min(0).to(original_dtype)


class _Entmax15Function(torch.autograd.Function):
    """Exact alpha=1.5 entmax with a stable analytical backward pass."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, dim: int) -> torch.Tensor:
        shifted = (values - values.amax(dim=dim, keepdim=True)) / 2
        # Padding uses the lowest finite FP32 value. Values this far below the
        # maximum cannot enter the support; clipping avoids overflow when the
        # support variance squares every sorted value.
        shifted = shifted.clamp_min(-2.0)
        ordered = shifted.sort(dim=dim, descending=True).values
        support_index = torch.arange(
            1,
            ordered.shape[dim] + 1,
            device=values.device,
            dtype=values.dtype,
        )
        shape = [1] * values.ndim
        shape[dim] = -1
        support_index = support_index.view(shape)
        mean = ordered.cumsum(dim) / support_index
        mean_square = ordered.square().cumsum(dim) / support_index
        support_variance = support_index * (mean_square - mean.square())
        delta = ((1 - support_variance) / support_index).clamp_min(0)
        thresholds = mean - delta.sqrt()
        support_size = (thresholds <= ordered).sum(dim=dim, keepdim=True).clamp_min(1)
        threshold = thresholds.gather(dim, support_size - 1)
        probabilities = (shifted - threshold).clamp_min(0).square()
        ctx.dim = dim
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (probabilities,) = ctx.saved_tensors
        generalized_derivative = probabilities.sqrt()
        grad_input = grad_output * generalized_derivative
        normalizer = generalized_derivative.sum(dim=ctx.dim, keepdim=True).clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        correction = grad_input.sum(dim=ctx.dim, keepdim=True) / normalizer
        return grad_input - correction * generalized_derivative, None


def _entmax15(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact alpha=1.5 entmax in FP32, including for mixed-precision inputs."""
    original_dtype = values.dtype
    probabilities = _Entmax15Function.apply(values.float(), dim)
    return probabilities.to(original_dtype)

class _EntmaxBisectFunction(torch.autograd.Function):
    """Arbitrary-alpha entmax for fixed alpha in (1, 2), solved in FP32."""

    @staticmethod
    def forward(
        ctx: Any,
        values: torch.Tensor,
        alpha: float,
        dim: int,
        iterations: int,
    ) -> torch.Tensor:
        values = values.float()
        scaled = (values - values.amax(dim=dim, keepdim=True)) * (alpha - 1.0)
        maximum = scaled.amax(dim=dim, keepdim=True)
        lower = maximum - 1.0
        upper = maximum
        exponent = 1.0 / (alpha - 1.0)
        for _ in range(iterations):
            threshold = (lower + upper) / 2.0
            probabilities = (scaled - threshold).clamp_min(0).pow(exponent)
            mass = probabilities.sum(dim=dim, keepdim=True)
            lower = torch.where(mass > 1.0, threshold, lower)
            upper = torch.where(mass <= 1.0, threshold, upper)
        threshold = (lower + upper) / 2.0
        probabilities = (scaled - threshold).clamp_min(0).pow(exponent)
        probabilities = probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        ctx.alpha = alpha
        ctx.dim = dim
        ctx.save_for_backward(probabilities)
        return probabilities

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (probabilities,) = ctx.saved_tensors
        generalized_derivative = probabilities.pow(2.0 - ctx.alpha)
        grad_input = grad_output.float() * generalized_derivative
        normalizer = generalized_derivative.sum(dim=ctx.dim, keepdim=True).clamp_min(
            torch.finfo(probabilities.dtype).tiny
        )
        correction = grad_input.sum(dim=ctx.dim, keepdim=True) / normalizer
        return grad_input - correction * generalized_derivative, None, None, None


def _entmax_bisect(
    values: torch.Tensor,
    alpha: float,
    dim: int = -1,
    iterations: int = 50,
) -> torch.Tensor:
    """Entmax over alpha in [1, 2], with exact softmax/sparsemax endpoints."""
    if not math.isfinite(alpha) or alpha < 1.0 or alpha > 2.0:
        raise ValueError("entmax alpha must be finite and in [1, 2]")
    if alpha <= 1.0 + 1e-7:
        return values.float().softmax(dim=dim).to(values.dtype)
    if alpha >= 2.0 - 1e-7:
        return _sparsemax(values, dim=dim)
    if abs(alpha - 1.5) <= 1e-7:
        return _entmax15(values, dim=dim)
    original_dtype = values.dtype
    probabilities = _EntmaxBisectFunction.apply(values.float(), float(alpha), dim, iterations)
    return probabilities.to(original_dtype)


class SpectraABMIL(nn.Module):
    """Permutation-invariant ABMIL/CLAM pooling with an optional QC routing prior."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.architecture not in {
            "abmil_ungated",
            "abmil_gated",
            "clam_gated",
            "clam_gated_multihead",
            "gated_frequency",
        }:
            raise ValueError("SpectraABMIL requires an ABMIL or CLAM architecture")
        self.config = config
        fixed_auxiliary_normalizations = {
            "fixed_uniform",
            "fixed_rank",
            "fixed_outlier",
            "ecod_monotone",
        }
        self.z_dim = config.input_dim - (
            1 if config.abmil_attention_normalization in fixed_auxiliary_normalizations else 0
        )
        if self.z_dim < 1:
            raise ValueError("ABMIL requires at least one Z feature")
        if config.abmil_attention_normalization not in {
            "fixed_uniform",
            "fixed_rank",
            "ecod_monotone",
            "fixed_outlier",
            "softmax",
            "entmax15",
            "entmax_annealed",
            "sparsemax",
            "sigmoid_mass",
            "sparse_sigmoid_mass",
            "dual_sparse_sigmoid_mass",
            "shared_sparse_sigmoid_mass",
            "independent_relu",
        }:
            raise ValueError("unsupported ABMIL attention normalization")
        self.frequency_aware = config.architecture == "gated_frequency"
        self.class_multihead = config.architecture == "clam_gated_multihead"
        if self.class_multihead and (
            config.abmil_attention_normalization != "sparsemax" or not config.abmil_additive_pooling
        ):
            raise ValueError("multihead CLAM requires sparsemax Additive MIL")
        self.input_projection = nn.Linear(self.z_dim, config.model_dim)
        self.input_activation = (
            nn.Identity() if config.abmil_input_activation == "identity" else nn.GELU()
        )
        self.input_dropout = nn.Dropout(config.dropout)
        self.attention_pool = (
            None
            if self.class_multihead
            or config.abmil_attention_normalization in fixed_auxiliary_normalizations
            else ABMILAttention(
                instance_dim=config.model_dim,
                attention_dim=config.abmil_attention_dim,
                gated=config.architecture
                in {
                    "abmil_gated",
                    "clam_gated",
                    "gated_frequency",
                },
            )
        )
        self.class_attention_pools = (
            nn.ModuleList(
                [
                    ABMILAttention(
                        instance_dim=config.model_dim,
                        attention_dim=config.abmil_attention_dim,
                        gated=True,
                    )
                    for _ in range(2)
                ]
            )
            if self.class_multihead
            else None
        )
        self.frequency_attention_pool = (
            ABMILAttention(
                instance_dim=config.model_dim,
                attention_dim=config.abmil_attention_dim,
                gated=True,
            )
            if config.abmil_attention_normalization == "dual_sparse_sigmoid_mass"
            else None
        )
        self.clam_instance_classifier = (
            nn.Linear(config.model_dim, 1) if config.architecture == "clam_gated" else None
        )
        self.class_clam_instance_classifier = (
            nn.Linear(config.model_dim, 2) if self.class_multihead else None
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.frequency_feature_dim = 3 if self.frequency_aware else 0
        self.classifier = nn.Linear(
            config.model_dim + self.frequency_feature_dim,
            2 if self.class_multihead else 1,
        )
        if config.abmil_attention_normalization == "ecod_monotone":
            knots = torch.tensor([0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0])
            initial_increments = 3.0 * torch.diff(knots)
            inverse_softplus = torch.log(torch.expm1(initial_increments))
            self.register_buffer("ecod_rank_knots", knots)
            self.ecod_monotone_raw_increments = nn.Parameter(inverse_softplus)
        else:
            self.register_buffer("ecod_rank_knots", torch.empty(0), persistent=False)
            self.register_parameter("ecod_monotone_raw_increments", None)

    def _ecod_monotone_logits(self, rank_percentile: torch.Tensor) -> torch.Tensor:
        """Evaluate a positive-slope linear spline that cannot reorder ECOD ranks."""
        if self.ecod_monotone_raw_increments is None or self.ecod_rank_knots.numel() < 2:
            raise RuntimeError("ECOD monotone calibration is not configured")
        increments = F.softplus(self.ecod_monotone_raw_increments)
        values = torch.cat((increments.new_zeros(1), increments.cumsum(dim=0)))
        positions = torch.searchsorted(
            self.ecod_rank_knots, rank_percentile.contiguous(), right=True
        ).clamp(1, len(self.ecod_rank_knots) - 1)
        lower = positions - 1
        upper = positions
        fraction = (rank_percentile - self.ecod_rank_knots[lower]) / (
            self.ecod_rank_knots[upper] - self.ecod_rank_knots[lower]
        ).clamp_min(1e-12)
        return values[lower] + fraction * (values[upper] - values[lower])

    @property
    def embedding_dim(self) -> int:
        multiplier = 2 if self.class_multihead else 1
        return multiplier * (self.config.model_dim + self.frequency_feature_dim)

    def _slide_center(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Remove slide location and configured scale nuisance."""
        return slide_relative_normalize(
            features,
            attention_mask,
            segment_ids,
            max_slides=self.config.max_slides,
            scale_strength=self.config.abmil_slide_scale_strength,
            minimum_variance=self.config.abmil_slide_scale_minimum_variance,
        )

    @staticmethod
    def _require_finite(values: torch.Tensor, stage: str) -> None:
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"ABMIL produced non-finite values after {stage}")

    def _forward_ragged_batch(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        return_attention: bool,
        qc_unusable_scores: torch.Tensor | None,
        exposure_slide_equivalents: torch.Tensor | None,
    ) -> ModelOutput:
        """Evaluate each true bag independently, then restore padded cell outputs."""
        sequence_length = features.shape[1]
        outputs: list[ModelOutput] = []
        valid_indices: list[torch.Tensor] = []
        for batch_index in range(features.shape[0]):
            indices = attention_mask[batch_index].nonzero(as_tuple=False).squeeze(-1)
            valid_indices.append(indices)
            outputs.append(
                self.forward(
                    features[batch_index, indices].unsqueeze(0),
                    torch.ones(
                        (1, indices.numel()),
                        device=attention_mask.device,
                        dtype=torch.bool,
                    ),
                    segment_ids[batch_index, indices].unsqueeze(0),
                    return_attention=return_attention,
                    exposure_slide_equivalents=(
                        exposure_slide_equivalents[batch_index:batch_index + 1]
                        if exposure_slide_equivalents is not None else None
                    ),
                    qc_unusable_scores=(
                        qc_unusable_scores[batch_index, indices].unsqueeze(0)
                        if qc_unusable_scores is not None
                        else None
                    ),
                )
            )

        def padded_cell_values(name: str) -> torch.Tensor:
            return torch.cat(
                [
                    output[name]
                    .new_zeros((1, sequence_length))
                    .index_copy(1, indices, output[name])
                    for output, indices in zip(outputs, valid_indices, strict=True)
                ],
                dim=0,
            )

        def padded_class_cell_values(name: str) -> torch.Tensor:
            return torch.cat(
                [
                    output[name]
                    .new_zeros((1, output[name].shape[1], sequence_length))
                    .index_copy(2, indices, output[name])
                    for output, indices in zip(outputs, valid_indices, strict=True)
                ],
                dim=0,
            )

        cross_attentions: tuple[torch.Tensor, ...] = ()
        if return_attention:
            cross_attentions = tuple(
                torch.cat(
                    [
                        output["cross_attentions"][layer_index]
                        .new_zeros(
                            (*output["cross_attentions"][layer_index].shape[:-1], sequence_length)
                        )
                        .index_copy(
                            -1,
                            indices,
                            output["cross_attentions"][layer_index],
                        )
                        for output, indices in zip(outputs, valid_indices, strict=True)
                    ],
                    dim=0,
                )
                for layer_index in range(len(outputs[0]["cross_attentions"]))
            )
        result: ModelOutput = {
            "logits": torch.cat([output["logits"] for output in outputs]),
            "cls_embedding": torch.cat([output["cls_embedding"] for output in outputs]),
            "cross_attentions": cross_attentions,
            "instance_logits": padded_cell_values("instance_logits"),
            "cell_contributions": padded_cell_values("cell_contributions"),
            "cell_predictor_scores": (
                padded_cell_values("cell_predictor_scores")
                if "cell_predictor_scores" in outputs[0]
                else padded_cell_values("instance_logits")
            ),
            "contribution_intercept": torch.cat(
                [output["contribution_intercept"] for output in outputs]
            ),
            "content_attention_logits": padded_cell_values("content_attention_logits"),
            "qc_attention_penalty": padded_cell_values("qc_attention_penalty"),
            "attention_support": padded_cell_values("attention_support").bool(),
            "attention_weights": padded_cell_values("attention_weights"),
            "clam_instance_logits": padded_cell_values("clam_instance_logits"),
            "pooling_weights": padded_cell_values("pooling_weights"),
            "frequency_attention_logits": padded_cell_values("frequency_attention_logits"),
            "frequency_gates": padded_cell_values("frequency_gates"),
            "attention_mass": torch.cat([output["attention_mass"] for output in outputs]),
            "attention_fraction": torch.cat([output["attention_fraction"] for output in outputs]),
            "attention_support_count": torch.cat(
                [output["attention_support_count"] for output in outputs]
            ),
        }
        if "class_logits" in outputs[0]:
            result["class_logits"] = torch.cat([output["class_logits"] for output in outputs])
            for name in (
                "class_attention_weights",
                "class_pooling_weights",
                "class_cell_contributions",
                "class_clam_instance_logits",
            ):
                result[name] = padded_class_cell_values(name)
            result["class_contribution_intercepts"] = torch.cat(
                [output["class_contribution_intercepts"] for output in outputs]
            )
        return result

    def _qc_attention_penalty(
        self,
        qc_unusable_scores: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a fixed bounded penalty without exposing QC to cell values."""
        if qc_unusable_scores is None:
            if self.config.qc_max_suppression > 1.0:
                raise ValueError("QC-aware ABMIL requires qc_unusable_scores")
            return torch.zeros_like(attention_mask, dtype=torch.float32)
        if qc_unusable_scores.shape != attention_mask.shape:
            raise ValueError("qc_unusable_scores must align with the attention mask")
        valid_scores = qc_unusable_scores[attention_mask].float()
        if not bool(torch.isfinite(valid_scores).all()) or bool(
            ((valid_scores < 0.0) | (valid_scores > 1.0)).any()
        ):
            raise ValueError("QC unusability scores must be finite probabilities")
        threshold = float(self.config.qc_threshold)
        excess = ((qc_unusable_scores.float() - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
        penalty = -math.log(float(self.config.qc_max_suppression)) * excess.pow(
            float(self.config.qc_penalty_exponent)
        )
        return penalty.masked_fill(~attention_mask, 0.0)

    def _stochastic_topk_mask(
        self,
        attention_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Training-only AC-MIL-style stochastic masking of top attention logits."""
        topk = self.config.abmil_stochastic_topk
        drop_probability = self.config.abmil_stochastic_topk_drop_probability
        if not self.training or topk == 0 or drop_probability == 0.0:
            return attention_logits
        if attention_logits.shape[0] != attention_mask.shape[0]:
            raise ValueError("attention logits and mask batch dimensions must match")
        if attention_logits.shape[-1] != attention_mask.shape[-1]:
            raise ValueError("attention logits and mask sequence dimensions must match")

        masked_logits = attention_logits.clone()
        flattened = masked_logits.reshape(masked_logits.shape[0], -1, masked_logits.shape[-1])
        sentinel = torch.finfo(masked_logits.dtype).min
        for bag_index in range(flattened.shape[0]):
            valid_count = int(attention_mask[bag_index].sum())
            # Always retain at least one instance even for synthetic tiny bags.
            selected_count = min(topk, max(valid_count - 1, 0))
            if selected_count == 0:
                continue
            for head_index in range(flattened.shape[1]):
                top_indices = torch.topk(
                    flattened[bag_index, head_index],
                    k=selected_count,
                ).indices
                drop = torch.rand(selected_count, device=masked_logits.device) < drop_probability
                flattened[bag_index, head_index, top_indices[drop]] = sentinel
        return masked_logits

    def _forward_class_multihead(
        self,
        instances: torch.Tensor,
        attention_mask: torch.Tensor,
        qc_attention_penalty: torch.Tensor,
        return_attention: bool,
    ) -> ModelOutput:
        """Two class-specific CLAM-MB heads with additive per-class evidence."""
        if self.class_attention_pools is None or self.class_clam_instance_classifier is None:
            raise RuntimeError("multihead CLAM modules were not initialized")
        batch_size = instances.shape[0]
        content_logits = torch.stack(
            [head(instances) for head in self.class_attention_pools], dim=1
        )
        attention_logits = content_logits.float() + qc_attention_penalty[:, None, :].float()
        attention_logits = attention_logits.masked_fill(
            ~attention_mask[:, None, :], torch.finfo(attention_logits.dtype).min
        )
        attention_logits = self._stochastic_topk_mask(attention_logits, attention_mask)
        class_attention = _sparsemax(attention_logits, dim=-1).float()
        class_attention = class_attention.masked_fill(~attention_mask[:, None, :], 0.0)
        self._require_finite(class_attention, "class-specific sparsemax normalization")

        class_instance_logits = _stable_cellwise_linear(
            self.class_clam_instance_classifier, instances
        ).permute(0, 2, 1)
        self._require_finite(class_instance_logits, "class-specific instance classification")

        composition_embeddings: list[torch.Tensor] = []
        contribution_rows: list[torch.Tensor] = []
        intercept_rows: list[torch.Tensor] = []
        class_logits: list[torch.Tensor] = []
        assert self.final_norm.weight is not None
        assert self.final_norm.bias is not None
        for class_index in range(2):
            weights = class_attention[:, class_index]
            bag_embedding = (weights.unsqueeze(-1) * instances.float()).sum(dim=1)
            centered = bag_embedding - bag_embedding.mean(dim=-1, keepdim=True)
            variance = centered.square().mean(dim=-1, keepdim=True)
            normalized = centered * torch.rsqrt(variance + self.final_norm.eps)
            composition_embeddings.append(
                normalized * self.final_norm.weight.float() + self.final_norm.bias.float()
            )

            attended_instances = weights.unsqueeze(-1) * instances.float()
            additive_centered = attended_instances - attended_instances.mean(dim=-1, keepdim=True)
            additive_variance = additive_centered.square().mean(dim=-1, keepdim=True)
            additive_normalized = additive_centered * torch.rsqrt(
                additive_variance + self.final_norm.eps
            )
            additive_embeddings = (
                additive_normalized * self.final_norm.weight.float() + self.final_norm.bias.float()
            )
            classifier_weight = self.classifier.weight[class_index].float()
            zero_baseline = (classifier_weight * self.final_norm.bias.float()).sum()
            contributions = (
                (additive_embeddings - self.final_norm.bias.float()) * classifier_weight
            ).sum(dim=-1)
            contributions = contributions.masked_fill(~attention_mask, 0.0)
            intercept = (self.classifier.bias[class_index].float() + zero_baseline).expand(
                batch_size
            )
            contribution_rows.append(contributions)
            intercept_rows.append(intercept)
            class_logits.append(intercept + contributions.sum(dim=-1))

        class_logit_tensor = torch.stack(class_logits, dim=1)
        class_contributions = torch.stack(contribution_rows, dim=1)
        class_intercepts = torch.stack(intercept_rows, dim=1)
        binary_logits = class_logit_tensor[:, 1] - class_logit_tensor[:, 0]
        binary_contributions = class_contributions[:, 1] - class_contributions[:, 0]
        binary_intercept = class_intercepts[:, 1] - class_intercepts[:, 0]
        positive_attention = class_attention[:, 1]
        support = (positive_attention > 0.0) & attention_mask
        valid_count = attention_mask.sum(dim=-1).clamp_min(1).float()
        mass = positive_attention.sum(dim=-1)
        cls_embedding = torch.cat(composition_embeddings, dim=-1)
        self._require_finite(binary_logits, "multihead additive classification")
        return {
            "logits": binary_logits,
            "class_logits": class_logit_tensor,
            "cls_embedding": cls_embedding,
            "cross_attentions": (class_attention[:, :, None, :],) if return_attention else (),
            "instance_logits": binary_contributions,
            "cell_contributions": binary_contributions,
            "contribution_intercept": binary_intercept,
            "content_attention_logits": content_logits[:, 1],
            "qc_attention_penalty": qc_attention_penalty,
            "attention_support": support,
            "attention_weights": positive_attention,
            "clam_instance_logits": class_instance_logits[:, 1],
            "pooling_weights": positive_attention,
            "attention_mass": mass,
            "attention_fraction": mass / valid_count,
            "attention_support_count": support.sum(dim=-1).float(),
            "frequency_attention_logits": content_logits[:, 1],
            "frequency_gates": positive_attention,
            "class_attention_weights": class_attention,
            "class_pooling_weights": class_attention,
            "class_cell_contributions": class_contributions,
            "class_contribution_intercepts": class_intercepts,
            "class_clam_instance_logits": class_instance_logits,
        }

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        return_attention: bool = False,
        qc_unusable_scores: torch.Tensor | None = None,
        exposure_slide_equivalents: torch.Tensor | None = None,
    ) -> ModelOutput:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError(f"features must have shape [batch, length, {self.config.input_dim}]")
        batch_size, sequence_length, _ = features.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length),
                device=features.device,
                dtype=torch.bool,
            )
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask shape must match the first two feature dimensions")
        if not bool(attention_mask.any(dim=1).all()):
            raise ValueError("every group must contain at least one unmasked instance")
        if segment_ids is not None and segment_ids.shape != (
            batch_size,
            sequence_length,
        ):
            raise ValueError("segment_ids shape must match the first two feature dimensions")

        if batch_size > 1 and sequence_length > 32_768:
            if segment_ids is None:
                segment_ids = torch.zeros(
                    (batch_size, sequence_length),
                    device=features.device,
                    dtype=torch.long,
                )
            return self._forward_ragged_batch(
                features,
                attention_mask,
                segment_ids,
                return_attention,
                qc_unusable_scores,
                exposure_slide_equivalents,
            )

        self._require_finite(features, "input preprocessing")

        z_features = features[..., : self.z_dim]
        if self.config.abmil_slide_center_input:
            if segment_ids is None:
                segment_ids = torch.zeros(
                    (batch_size, sequence_length),
                    device=features.device,
                    dtype=torch.long,
                )
            z_features = self._slide_center(z_features, attention_mask, segment_ids)
            self._require_finite(z_features, "slide-relative normalization")

        # Canonical ABMIL remains permutation invariant. Slide segments are used
        # only for optional label-free centering, never as learned identifiers.
        if (
            not torch.is_autocast_enabled()
            and z_features.dtype != self.input_projection.weight.dtype
        ):
            z_features = z_features.to(self.input_projection.weight.dtype)
        projected_instances = _stable_cellwise_linear(self.input_projection, z_features)
        if not bool(torch.isfinite(projected_instances).all()):
            raise FloatingPointError(
                "ABMIL produced non-finite values after instance projection; "
                f"input_max_abs={float(z_features.abs().amax()):.6g} "
                f"weight_max_abs={float(self.input_projection.weight.abs().amax()):.6g} "
                f"bias_max_abs={float(self.input_projection.bias.abs().amax()):.6g}"
            )
        activated_instances = self.input_activation(projected_instances)
        self._require_finite(activated_instances, "instance activation")
        instances = self.input_dropout(activated_instances)
        self._require_finite(instances, "instance dropout")
        qc_attention_penalty = self._qc_attention_penalty(qc_unusable_scores, attention_mask)
        if self.class_multihead:
            return self._forward_class_multihead(
                instances,
                attention_mask,
                qc_attention_penalty,
                return_attention,
            )
        clam_instance_logits = (
            self.clam_instance_classifier(instances).squeeze(-1)
            if self.clam_instance_classifier is not None
            else torch.zeros_like(attention_mask, dtype=instances.dtype)
        )
        self._require_finite(clam_instance_logits, "CLAM instance classification")
        padding_mask = attention_mask[:, None, None, :]
        normalization = self.config.abmil_attention_normalization
        if normalization in {
            "fixed_uniform",
            "fixed_rank",
            "fixed_outlier",
            "ecod_monotone",
        }:
            # The final feature is an auxiliary ECOD quantity and is excluded
            # from the biological instance projection. Only the legacy
            # fixed_outlier arm uses the transformed raw score; the registered
            # baselines below use the pre-QC within-slide rank fraction.
            uniform = attention_mask.float() / attention_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            if normalization == "fixed_uniform":
                content_attention_logits = torch.zeros_like(attention_mask, dtype=torch.float32)
                fixed_weights = uniform
            elif normalization == "fixed_outlier":
                content_attention_logits = features[..., -1].float().clamp_min(0.0)
                outlier_strength = content_attention_logits.masked_fill(~attention_mask, 0.0)
                total_strength = outlier_strength.sum(dim=-1, keepdim=True)
                fixed_weights = torch.where(
                    total_strength > 0,
                    outlier_strength / total_strength.clamp_min(1e-12),
                    uniform,
                )
            else:
                rank_percentile = (1.0 - features[..., -1].float()).clamp(0.0, 1.0)
                if normalization == "fixed_rank":
                    content_attention_logits = rank_percentile
                    strength = rank_percentile.masked_fill(~attention_mask, 0.0)
                    fixed_weights = strength / strength.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                else:
                    content_attention_logits = self._ecod_monotone_logits(rank_percentile)
                    masked_logits = content_attention_logits.masked_fill(
                        ~attention_mask, torch.finfo(torch.float32).min
                    )
                    fixed_weights = _sparsemax(masked_logits, dim=-1).float()
            attention_logits = content_attention_logits[:, None, None, :]
        else:
            assert self.attention_pool is not None
            content_attention_logits = self.attention_pool(instances)
            attention_logits = (content_attention_logits.float() + qc_attention_penalty.float())[
                :, None, None, :
            ]
            if self.config.abmil_attention_temperature != 1.0:
                attention_logits = attention_logits / self.config.abmil_attention_temperature
            self._require_finite(attention_logits, "attention scoring")
            attention_logits = attention_logits.masked_fill(
                ~padding_mask, torch.finfo(attention_logits.dtype).min
            )
            attention_logits = self._stochastic_topk_mask(attention_logits, attention_mask)
        frequency_content_logits = (
            self.frequency_attention_pool(instances)
            if self.frequency_attention_pool is not None
            else content_attention_logits
        )
        frequency_attention_logits = (
            frequency_content_logits.float() + qc_attention_penalty.float()
        )[:, None, None, :]
        frequency_attention_logits = frequency_attention_logits.masked_fill(
            ~padding_mask, torch.finfo(frequency_attention_logits.dtype).min
        )
        if normalization in {
            "fixed_uniform",
            "fixed_rank",
            "fixed_outlier",
            "ecod_monotone",
        }:
            attention = fixed_weights[:, None, None, :]
            pooling_weights = fixed_weights
        elif normalization == "softmax":
            # Very long bags make reduced-precision softmax backward fragile even
            # when its forward values are finite. Keep this small scalar path and
            # the weighted reduction in FP32 while projections remain autocast.
            attention = attention_logits.float().softmax(dim=-1)
            pooling_weights = attention[:, 0, 0]
        elif normalization == "entmax15":
            attention = _entmax15(attention_logits, dim=-1).float()
            pooling_weights = attention[:, 0, 0]
        elif normalization == "sparsemax":
            attention = _sparsemax(attention_logits, dim=-1).float()
            pooling_weights = attention[:, 0, 0]
        elif normalization == "independent_relu":
            if exposure_slide_equivalents is None:
                raise ValueError("independent ReLU requires explicit acquired slide exposure")
            exposure = exposure_slide_equivalents.to(device=features.device, dtype=torch.float32)
            if (exposure.shape != (batch_size,) or not bool(torch.isfinite(exposure).all())
                    or not bool((exposure > 0).all())):
                raise ValueError("exposure must contain one finite positive value per patient")
            attention = torch.relu(attention_logits.float() - self.config.abmil_independent_threshold)
            attention = attention.masked_fill(~padding_mask, 0.0)
            pooling_weights = (
                attention[:, 0, 0] * self.config.abmil_independent_scale / exposure[:, None]
            )
        elif normalization in {
            "dual_sparse_sigmoid_mass",
            "shared_sparse_sigmoid_mass",
        }:
            pooling_weights = _sparsemax(attention_logits, dim=-1).float()[:, 0, 0]
            attention = torch.sigmoid(frequency_attention_logits.float())
        else:
            independent_gates = torch.sigmoid(attention_logits.float())
            if normalization == "sparse_sigmoid_mass":
                sparse_support = _sparsemax(attention_logits, dim=-1) > 0
                independent_gates = independent_gates.masked_fill(~sparse_support, 0.0)
            attention = independent_gates
        # Keep padding impossible even under reduced-precision finite sentinels.
        attention = attention.masked_fill(~padding_mask, 0)
        attention_support = (
            (pooling_weights > 0.0) & attention_mask
            if normalization in {"dual_sparse_sigmoid_mass", "shared_sparse_sigmoid_mass"}
            else (attention[:, 0, 0] > 0.0) & attention_mask
        )
        attention_mass = attention[:, 0, 0].sum(dim=-1)
        valid_count = attention_mask.sum(dim=-1).clamp_min(1).float()
        attention_fraction = attention_mass / valid_count
        attention_support_count = attention_support.sum(dim=-1).float()
        if normalization in {"sigmoid_mass", "sparse_sigmoid_mass"}:
            pooling_weights = attention[:, 0, 0] / attention_mass.unsqueeze(-1).clamp_min(1e-12)
        self._require_finite(attention, "attention normalization")
        # An explicit reduction is equivalent to the [1,L] @ [L,D] batched
        # matmul, but avoids a cuBLAS long-thin backward path that can return
        # non-finite gradients on very long, heavily padded bags.
        bag_embedding = (pooling_weights.unsqueeze(-1) * instances.float()).sum(dim=1)
        self._require_finite(bag_embedding, "attention pooling")
        centered_embedding = bag_embedding - bag_embedding.mean(dim=-1, keepdim=True)
        variance = centered_embedding.square().mean(dim=-1, keepdim=True)
        normalized_embedding = centered_embedding * torch.rsqrt(variance + self.final_norm.eps)
        assert self.final_norm.weight is not None
        assert self.final_norm.bias is not None
        composition_embedding = (
            normalized_embedding * self.final_norm.weight.float() + self.final_norm.bias.float()
        )
        if normalization == "shared_sparse_sigmoid_mass":
            frequency_features = torch.stack(
                (
                    torch.log1p(attention_mass),
                    torch.log1p(valid_count),
                    20.0 * (attention_fraction - 0.5),
                ),
                dim=-1,
            )
        else:
            frequency_features = torch.stack(
                (
                    torch.log1p(attention_mass) / 10.0,
                    torch.log1p(valid_count) / 10.0,
                    attention_fraction,
                ),
                dim=-1,
            )
        cls_embedding = (
            torch.cat((composition_embedding, frequency_features), dim=-1)
            if self.frequency_aware
            else composition_embedding
        )
        self._require_finite(cls_embedding, "final normalization")
        composition_weight = self.classifier.weight[0, : self.config.model_dim].float()
        cell_predictor_scores = torch.zeros_like(attention_mask, dtype=torch.float32)
        if self.config.abmil_additive_pooling:
            # Additive MIL (Javed et al., 2022): move the predictor inside the
            # instance sum. Centering at psi(0) makes zero-attention cells
            # contribute exactly zero and adds the learned intercept once.
            additive_normalization = self.config.abmil_additive_normalization
            if additive_normalization == "post_attention":
                additive_inputs = pooling_weights.unsqueeze(-1) * instances.float()
                additive_centered = additive_inputs - additive_inputs.mean(dim=-1, keepdim=True)
                additive_variance = additive_centered.square().mean(dim=-1, keepdim=True)
                additive_normalized = additive_centered * torch.rsqrt(
                    additive_variance + self.final_norm.eps
                )
                unweighted_contributions = (
                    additive_normalized * self.final_norm.weight.float() * composition_weight
                ).sum(dim=-1)
                cell_contributions = unweighted_contributions
                zero_baseline = (composition_weight * self.final_norm.bias.float()).sum()
            elif additive_normalization == "pre_attention":
                additive_inputs = instances.float()
                additive_centered = additive_inputs - additive_inputs.mean(dim=-1, keepdim=True)
                additive_variance = additive_centered.square().mean(dim=-1, keepdim=True)
                additive_normalized = additive_centered * torch.rsqrt(
                    additive_variance + self.final_norm.eps
                )
                unweighted_contributions = (
                    additive_normalized * self.final_norm.weight.float() * composition_weight
                ).sum(dim=-1)
                cell_predictor_scores = unweighted_contributions.masked_fill(~attention_mask, 0.0)
                cell_contributions = pooling_weights * unweighted_contributions
                zero_baseline = (composition_weight * self.final_norm.bias.float()).sum()
            else:
                cell_contributions = pooling_weights * (instances.float() * composition_weight).sum(
                    dim=-1
                )
                zero_baseline = instances.new_zeros((), dtype=torch.float32)
            cell_contributions = cell_contributions.masked_fill(~attention_mask, 0.0)
            contribution_intercept = (self.classifier.bias[0].float() + zero_baseline).expand(
                batch_size
            )
            logits = contribution_intercept + cell_contributions.sum(dim=-1)
        else:
            logits = self.classifier(cls_embedding).squeeze(-1)
            inverse_scale = torch.rsqrt(variance + self.final_norm.eps)
            weighted_norm = composition_weight * self.final_norm.weight.float()
            effective_weight = inverse_scale * (weighted_norm - weighted_norm.mean()).unsqueeze(0)
            cell_contributions = pooling_weights * (
                instances.float() * effective_weight.unsqueeze(1)
            ).sum(dim=-1)
            frequency_contribution = (
                (
                    self.classifier.weight[0, self.config.model_dim :].float() * frequency_features
                ).sum(dim=-1)
                if self.frequency_aware
                else torch.zeros(batch_size, device=features.device)
            )
            contribution_intercept = (
                self.classifier.bias[0].float()
                + (composition_weight * self.final_norm.bias.float()).sum()
                + frequency_contribution
            )
        return {
            "logits": logits,
            "cls_embedding": cls_embedding,
            "cross_attentions": (attention,) if return_attention else (),
            "instance_logits": cell_contributions,
            "cell_contributions": cell_contributions,
            "cell_predictor_scores": cell_predictor_scores,
            "contribution_intercept": contribution_intercept,
            "content_attention_logits": content_attention_logits,
            "qc_attention_penalty": qc_attention_penalty,
            "attention_support": attention_support,
            "attention_weights": attention[:, 0, 0],
            "clam_instance_logits": clam_instance_logits,
            "pooling_weights": pooling_weights,
            "attention_mass": attention_mass,
            "attention_fraction": attention_fraction,
            "attention_support_count": attention_support_count,
            "frequency_attention_logits": frequency_content_logits,
            "frequency_gates": attention[:, 0, 0],
        }


SpectraModel = VanillaGatedABMIL | SpectraABMIL


def _load_batch_residualization_state(
    config: ModelConfig,
) -> dict[str, torch.Tensor] | None:
    if config.batch_residualization_path is None:
        return None
    from .batch_effects import load_paired_batch_residualizer

    residualizer, _ = load_paired_batch_residualizer(config.batch_residualization_path)
    if config.batch_residualization_components > residualizer.maximum_components:
        raise ValueError("requested batch residualization components exceed the artifact")
    if residualizer.dimensions > config.input_dim:
        raise ValueError("batch residualizer has more dimensions than the model input")
    return {
        "center": torch.from_numpy(residualizer.center.copy()),
        "scale": torch.from_numpy(residualizer.scale.copy()),
        "basis": torch.from_numpy(
            residualizer.basis[:, : config.batch_residualization_components].copy()
        ),
    }


def _attach_batch_residualization(
    model: SpectraModel,
    config: ModelConfig,
    state: dict[str, torch.Tensor] | None,
) -> SpectraModel:
    if state is None:
        return model
    if set(state) != {"center", "scale", "basis"}:
        raise ValueError("batch residualization state is incomplete")
    basis = state["basis"]
    expected = config.batch_residualization_components
    if basis.ndim != 2 or basis.shape[1] != expected or basis.shape[0] > config.input_dim:
        raise ValueError("checkpoint batch residualization basis has an invalid shape")
    residualization = FixedBatchResidualization(
        center=state["center"],
        scale=state["scale"],
        basis=basis,
        strength=config.batch_residualization_strength,
        max_slides=config.max_slides,
        standardize_input=config.batch_residualization_standardize_input,
    )
    model.add_module("batch_residualization", residualization)

    def apply_before_forward(
        module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[tuple[object, ...], dict[str, object]]:
        if not args or not isinstance(args[0], torch.Tensor):
            raise ValueError("model features must be the first positional argument")
        transform = module.get_submodule("batch_residualization")
        attention_mask = args[1] if len(args) > 1 else kwargs.get("attention_mask")
        segment_ids = args[2] if len(args) > 2 else kwargs.get("segment_ids")
        features = args[0]
        transform_dimensions = int(transform.center.numel())
        biological = transform(features[..., :transform_dimensions], attention_mask, segment_ids)
        transformed = (
            torch.cat((biological, features[..., transform_dimensions:]), dim=-1)
            if features.shape[-1] > transform_dimensions
            else biological
        )
        transformed_args = (transformed, *args[1:])
        return transformed_args, kwargs

    model.register_forward_pre_hook(apply_before_forward, with_kwargs=True)
    return model


def build_model(
    config: ModelConfig,
    batch_residualization_state: dict[str, torch.Tensor] | None = None,
    covariate_normalization_state: dict[str, torch.Tensor] | None = None,
) -> SpectraModel:
    residualization_state = (
        batch_residualization_state
        if batch_residualization_state is not None
        else _load_batch_residualization_state(config)
    )
    if config.architecture == "population_abundance":
        from .population_abundance import PopulationAbundance
        if residualization_state is not None:
            raise ValueError("population abundance requires data-level fixed normalization")
        return PopulationAbundance(config)
    if config.architecture == "additive_burden":
        from .additive_burden import AdditiveBurden
        model = AdditiveBurden(config)
        return _attach_batch_residualization(model, config, residualization_state)
    if config.architecture == "additive_experts":
        from .additive_experts import AdditiveExperts
        model = AdditiveExperts(config, covariate_normalization_state)
        return _attach_batch_residualization(model, config, residualization_state)
    if config.architecture == "abmil_vanilla":
        if residualization_state is not None:
            raise ValueError("vanilla ABMIL does not accept residualization state")
        return VanillaGatedABMIL(config)
    if config.architecture in {
        "abmil_ungated",
        "abmil_gated",
        "clam_gated",
        "clam_gated_multihead",
        "gated_frequency",
    }:
        model: SpectraModel = SpectraABMIL(config)
        return _attach_batch_residualization(model, config, residualization_state)
    raise ValueError(f"unsupported model architecture: {config.architecture!r}")


def forward_spectra_model(
    model: SpectraModel,
    features: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    cell_covariates: torch.Tensor | None = None,
    exposure_slide_equivalents: torch.Tensor | None = None,
    return_attention: bool = False,
    qc_unusable_scores: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Route inference while keeping QC outside biological discovery models."""
    if model.config.architecture in {"additive_experts", "additive_burden"}:
        return model(features, attention_mask, segment_ids,
                     cell_covariates=cell_covariates, return_attention=return_attention,
                     qc_unusable_scores=qc_unusable_scores,
                     exposure_slide_equivalents=exposure_slide_equivalents)
    if cell_covariates is not None:
        raise ValueError("this architecture does not consume cell covariates")
    if isinstance(model, SpectraABMIL):
        return model(
            features, attention_mask, segment_ids,
            return_attention=return_attention,
            qc_unusable_scores=qc_unusable_scores,
            exposure_slide_equivalents=exposure_slide_equivalents,
        )
    return model(
        features,
        attention_mask,
        segment_ids,
        return_attention=return_attention,
        qc_unusable_scores=qc_unusable_scores,
    )
