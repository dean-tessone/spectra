"""Corrected additive experts with optional, independently normalized cell measurements."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn

from .config import ModelConfig
from .model import (
    ABMILAttention,
    SpectraABMIL,
    _entmax15,
    _entmax_bisect,
    _sparsemax,
    _stable_cellwise_linear,
)


class AdditiveExperts(SpectraABMIL):
    def __init__(self, config: ModelConfig, covariate_state=None):
        if config.architecture != "additive_experts":
            raise ValueError("expected additive_experts architecture")
        if (
            config.abmil_attention_normalization
            not in {"sparsemax", "softmax", "entmax15", "entmax_annealed"}
            or not config.abmil_additive_pooling
            or config.abmil_additive_normalization != "pre_attention"
            or config.abmil_stochastic_topk
            or config.clam_instance_loss_weight
        ):
            raise ValueError(
                "experts require corrected Additive with sparsemax, softmax or entmax15 without instance supervision"
            )
        if config.additive_head_count < 1 or config.additive_cell_chunk_size < 1:
            raise ValueError("expert head count and cell chunk size must be positive")
        super().__init__(replace(config, architecture="abmil_gated"))
        self.config = config
        # Legacy checkpoints default to the registered sparsemax endpoint.
        self.training_epoch = config.abmil_entmax_anneal_epochs
        self.attention_alpha_override = None
        self.extra_attention_pools = nn.ModuleList(
            [
                ABMILAttention(config.model_dim, config.abmil_attention_dim, gated=True)
                for _ in range(config.additive_head_count - 1)
            ]
        )
        self.extra_norms = nn.ModuleList(
            [nn.LayerNorm(config.model_dim) for _ in self.extra_attention_pools]
        )
        self.extra_classifiers = nn.ModuleList(
            [nn.Linear(config.model_dim, 1) for _ in self.extra_attention_pools]
        )
        dim = config.cell_covariate_dim
        self.covariate_projection = None
        if dim:
            if covariate_state is None:
                if not config.cell_covariate_normalization_path:
                    raise ValueError(
                        "cell measurements require a training-fold normalization artifact"
                    )
                with np.load(
                    config.cell_covariate_normalization_path, allow_pickle=False
                ) as payload:
                    covariate_state = {
                        k: torch.from_numpy(payload[k].astype(np.float32))
                        for k in ("center", "scale")
                    }
            center = torch.as_tensor(covariate_state["center"]).detach().float().cpu()
            scale = torch.as_tensor(covariate_state["scale"]).detach().float().cpu()
            if (
                center.shape != (dim,)
                or scale.shape != (dim,)
                or not torch.isfinite(center).all()
                or not torch.isfinite(scale).all()
                or not (scale > 0).all()
            ):
                raise ValueError("invalid cell covariate normalization")
            self.register_buffer("covariate_center", center)
            self.register_buffer("covariate_scale", scale)
            # Preserve the caller's RNG stream and exact initial biological parameters.
            with torch.random.fork_rng(devices=[]):
                self.covariate_projection = nn.Linear(dim, config.model_dim, bias=False)
            nn.init.zeros_(self.covariate_projection.weight)

    def set_training_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("training epoch must be positive")
        self.training_epoch = int(epoch)

    def set_attention_alpha(self, alpha: float | None) -> None:
        if alpha is not None and (not np.isfinite(alpha) or not 1 <= alpha <= 2):
            raise ValueError("attention alpha must be finite and within [1, 2]")
        self.attention_alpha_override = None if alpha is None else float(alpha)

    def attention_entmax_alpha(self) -> float:
        if self.attention_alpha_override is not None:
            return self.attention_alpha_override
        if self.config.abmil_attention_normalization != "entmax_annealed":
            if self.config.abmil_attention_normalization == "softmax":
                return 1.0
            if self.config.abmil_attention_normalization == "entmax15":
                return 1.5
            return 2.0
        progress = min(
            max(self.training_epoch - 1, 0)
            / (self.config.abmil_entmax_anneal_epochs - 1),
            1.0,
        )
        return (
            self.config.abmil_entmax_alpha_start
            + progress
            * (
                self.config.abmil_entmax_alpha_end
                - self.config.abmil_entmax_alpha_start
            )
        )

    def copy_single_head_reference(self, reference):
        if self.config.additive_head_count != 1:
            raise ValueError("compatibility copy requires one head")
        own = self.state_dict()
        incoming = reference.state_dict()
        if not set(incoming).issubset(own):
            raise ValueError("reference has incompatible state")
        own.update(incoming)
        self.load_state_dict(own, strict=True)

    def forward(
        self,
        features,
        attention_mask=None,
        segment_ids=None,
        return_attention=False,
        qc_unusable_scores=None,
        exposure_slide_equivalents=None,
        cell_covariates=None,
    ):
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("expert features have wrong dimensions")
        b, n, _ = features.shape
        mask = (
            torch.ones((b, n), dtype=torch.bool, device=features.device)
            if attention_mask is None
            else attention_mask
        )
        if mask.shape != (b, n):
            raise ValueError("attention mask must align with cells")
        segments = torch.zeros_like(mask, dtype=torch.long) if segment_ids is None else segment_ids
        if segments.shape != mask.shape:
            raise ValueError("segments must align with cells")
        if n == 0:
            raise ValueError("use padded masked cells for an empty group")
        x = features.float().masked_fill(~mask[..., None], 0)
        self._require_finite(x, "expert input")
        if self.config.abmil_slide_center_input:
            present = mask.any(-1)
            if bool(present.all()):
                x = self._slide_center(x, mask, segments)
            elif bool(present.any()):
                centered = torch.zeros_like(x)
                centered[present] = self._slide_center(x[present], mask[present], segments[present])
                x = centered
        covariates = None
        if self.covariate_projection is not None:
            if cell_covariates is None or cell_covariates.shape != (
                b,
                n,
                self.config.cell_covariate_dim,
            ):
                raise ValueError("aligned cell_covariates are required")
            covariates = cell_covariates.float().masked_fill(~mask[..., None], 0)
            self._require_finite(covariates, "cell measurements")
            covariates = (covariates - self.covariate_center) / self.covariate_scale
        elif cell_covariates is not None:
            raise ValueError("unexpected cell covariates for a Z-only model")
        parts = []
        for start in range(0, n, self.config.additive_cell_chunk_size):
            stop = start + self.config.additive_cell_chunk_size
            part = _stable_cellwise_linear(self.input_projection, x[:, start:stop])
            if covariates is not None:
                part = part + _stable_cellwise_linear(
                    self.covariate_projection, covariates[:, start:stop]
                )
            parts.append(self.input_activation(part))
        # Dropout applied once so changing chunk size does not change training masks.
        instances = self.input_dropout(torch.cat(parts, dim=1))
        penalty = self._qc_attention_penalty(qc_unusable_scores, mask)
        centered = instances.float() - instances.float().mean(-1, keepdim=True)
        variance = centered.square().mean(-1, keepdim=True)
        attention, contributions, predictors, intercepts, contents, embeddings = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for head, norm, classifier in zip(
            [self.attention_pool, *self.extra_attention_pools],
            [self.final_norm, *self.extra_norms],
            [self.classifier, *self.extra_classifiers],
            strict=True,
        ):
            content = torch.cat(
                [
                    head(instances[:, j : j + self.config.additive_cell_chunk_size])
                    for j in range(0, n, self.config.additive_cell_chunk_size)
                ],
                dim=1,
            ).float()
            scores = (content + penalty) / self.config.abmil_attention_temperature
            masked_scores = scores.masked_fill(~mask, torch.finfo(torch.float32).min)
            normalization = self.config.abmil_attention_normalization
            if normalization == "sparsemax":
                weights = _sparsemax(masked_scores, dim=-1)
            elif normalization == "entmax15":
                weights = _entmax15(masked_scores, dim=-1)
            elif normalization == "entmax_annealed":
                weights = _entmax_bisect(
                    masked_scores, self.attention_entmax_alpha(), dim=-1
                )
            else:  # softmax; the constructor validates the closed set above.
                weights = masked_scores.softmax(dim=-1)
            # Fully masked patients contribute only the learned intercept. Keep
            # the scalar normalization and reductions in FP32 for every method.
            weights = weights.float().masked_fill(~mask, 0)
            normalized = centered * torch.rsqrt(variance + norm.eps)
            predictor = (
                (normalized * norm.weight.float() * classifier.weight[0].float())
                .sum(-1)
                .masked_fill(~mask, 0)
            )
            intercept = (
                classifier.bias[0].float()
                + (classifier.weight[0].float() * norm.bias.float()).sum()
            ).expand(b)
            attention.append(weights)
            predictors.append(predictor)
            contributions.append(weights * predictor)
            intercepts.append(intercept)
            contents.append(content.masked_fill(~mask, 0))
            embeddings.append(norm((weights[..., None] * instances).sum(1)))
        head_attention = torch.stack(attention, 1)
        head_contributions = torch.stack(contributions, 1)
        head_intercepts = torch.stack(intercepts, 1)
        cell_contributions = head_contributions.mean(1)
        intercept = head_intercepts.mean(1)
        logits = intercept + cell_contributions.sum(-1)
        self._require_finite(logits, "expert additive readout")
        support = (head_attention > 0).any(1)
        routing = head_attention.mean(1)
        count = mask.sum(-1).clamp_min(1).float()
        return {
            "logits": logits,
            "cls_embedding": torch.stack(embeddings, 1).mean(1),
            "cell_contributions": cell_contributions,
            "instance_logits": cell_contributions,
            "contribution_intercept": intercept,
            "head_attention_weights": head_attention,
            "head_cell_contributions": head_contributions,
            "head_contribution_intercepts": head_intercepts,
            "head_logits": head_intercepts + head_contributions.sum(-1),
            "head_cell_predictor_scores": torch.stack(predictors, 1),
            "cell_predictor_scores": torch.stack(predictors, 1).mean(1),
            "content_attention_logits": torch.stack(contents, 1).mean(1),
            "attention_weights": routing,
            "pooling_weights": routing,
            "attention_support": support,
            "attention_mass": routing.sum(-1),
            "attention_fraction": routing.sum(-1) / count,
            "attention_support_count": support.sum(-1).float(),
            "qc_attention_penalty": penalty,
            "clam_instance_logits": torch.zeros_like(routing),
            "frequency_attention_logits": torch.stack(contents, 1).mean(1),
            "frequency_gates": routing,
            "cross_attentions": (head_attention[:, :, None, :],) if return_attention else (),
            "attention_entmax_alpha": torch.full(
                (b,), self.attention_entmax_alpha(), device=logits.device, dtype=torch.float32
            ),
        }
