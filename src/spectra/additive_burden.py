"""Frozen corrected Additive MIL plus cell-local, exposure-aware residual evidence."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .additive_experts import AdditiveExperts
from .config import ModelConfig
from .model import _stable_cellwise_linear

BURDEN_MODES = ("intercept", "count_rate", "total_rate", "phenotype_rate", "phenotype_frequency")


class AdditiveBurden(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.burden_mode not in BURDEN_MODES:
            raise ValueError("unknown burden mode")
        if config.cell_covariate_dim or config.batch_residualization_components:
            raise ValueError("burden requires global-only normalization and Z512 inputs")
        self.config = config
        self.parent = AdditiveExperts(replace(config, architecture="additive_experts"))
        self.projection = nn.Linear(config.input_dim, config.model_dim)
        self.projection.load_state_dict(self.parent.input_projection.state_dict())
        self.dropout = nn.Dropout(config.dropout)
        self.activity_head = nn.Linear(config.model_dim, 4)
        nn.init.normal_(self.activity_head.weight, std=0.01)
        nn.init.constant_(self.activity_head.bias, -4.0)
        dim = {
            "intercept": 0,
            "count_rate": 1,
            "total_rate": 1,
            "phenotype_rate": 4,
            "phenotype_frequency": 4,
        }[config.burden_mode]
        self.readout_weight = nn.Parameter(torch.zeros(dim))
        self.readout_intercept = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("feature_scale", torch.ones(dim))
        self.register_buffer("scale_initialized", torch.tensor(False))
        self.register_buffer("parent_initialized", torch.tensor(False))
        self._last_gradients = {}
        self._last_activity = {}
        self.freeze_parent()

    @property
    def learned_activity(self):
        return self.config.burden_mode not in ("intercept", "count_rate")

    def freeze_parent(self):
        self.parent.requires_grad_(False)
        self.parent.eval()
        for module in (self.projection, self.activity_head):
            module.requires_grad_(self.learned_activity)
        self.readout_weight.requires_grad_(self.readout_weight.numel() > 0)
        self.readout_intercept.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def initialize_parent(self, model):
        """Copy biological weights; global transform is attached only to the wrapper."""
        state = {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith("batch_residualization.")
        }
        self.parent.load_state_dict(state, strict=True)
        self.projection.load_state_dict(self.parent.input_projection.state_dict())
        if hasattr(model, "batch_residualization"):
            if not hasattr(self, "batch_residualization"):
                raise ValueError("wrapper is missing parent normalization")
            self.batch_residualization.load_state_dict(model.batch_residualization.state_dict())
        self.parent_initialized.fill_(True)
        self.freeze_parent()

    def residual(self, features, mask, exposure):
        clean = features.float().masked_fill(~mask[..., None], 0)
        if not torch.isfinite(clean).all():
            raise ValueError("nonfinite burden features")
        n = mask.sum(-1).float()
        if self.learned_activity:
            hidden = torch.cat(
                [
                    _stable_cellwise_linear(self.projection, block)
                    for block in clean.split(self.config.additive_cell_chunk_size, dim=1)
                ],
                dim=1,
            )
            hidden = self.dropout(F.gelu(hidden))
            q = _stable_cellwise_linear(self.activity_head, hidden).float().sigmoid()
            q = q.masked_fill(~mask[..., None], 0)
        else:
            q = clean.new_zeros((*mask.shape, 4))
        union = 1 - (1 - q).prod(-1)
        mode = self.config.burden_mode
        if mode == "intercept":
            local = clean.new_zeros((*mask.shape, 0))
        elif mode == "count_rate":
            local = mask.float()[..., None]
        elif mode == "total_rate":
            local = union[..., None]
        else:
            local = q
        denominator = n.clamp_min(1) if mode == "phenotype_frequency" else exposure
        pooled = local.sum(1) / denominator[:, None]
        if not torch.isfinite(self.feature_scale).all() or (self.feature_scale <= 0).any():
            raise ValueError("invalid burden feature scale")
        local_evidence = (local * (self.readout_weight / self.feature_scale)).sum(-1)
        contribution = local_evidence / denominator[:, None]
        return q, union, pooled, local_evidence, contribution

    def forward(
        self,
        features,
        attention_mask,
        segment_ids=None,
        *,
        exposure_slide_equivalents=None,
        qc_unusable_scores=None,
        return_attention=False,
        cell_covariates=None,
    ):
        if cell_covariates is not None:
            raise ValueError("burden does not consume cell covariates")
        if (
            features.ndim != 3
            or attention_mask.shape != features.shape[:2]
            or attention_mask.dtype != torch.bool
        ):
            raise ValueError("invalid burden features/mask")
        if exposure_slide_equivalents is None:
            raise ValueError("burden requires acquired slide exposure")
        exposure = exposure_slide_equivalents.to(features.device, dtype=torch.float32)
        if (
            exposure.shape != (len(features),)
            or not torch.isfinite(exposure).all()
            or (exposure <= 0).any()
        ):
            raise ValueError("exposure must be finite, positive and patient-aligned")
        if features.shape[1] == 0:
            features = features.new_zeros((len(features), 1, features.shape[-1]))
            attention_mask = torch.zeros(
                (len(features), 1), device=features.device, dtype=torch.bool
            )
            segment_ids = torch.zeros_like(attention_mask, dtype=torch.long)
            qc_unusable_scores = None
        with torch.no_grad():
            parent = self.parent(
                features,
                attention_mask,
                segment_ids,
                exposure_slide_equivalents=exposure,
                qc_unusable_scores=qc_unusable_scores,
                return_attention=return_attention,
            )
        q, union, pooled, raw, c = self.residual(features, attention_mask, exposure)
        residual = self.readout_intercept + c.sum(-1)
        result = dict(parent)
        result.update(
            logits=parent["logits"] + residual,
            component_logits={"parent": parent["logits"], "burden": residual},
            parent_logits=parent["logits"],
            burden_logits=residual,
            parent_cell_contributions=parent["cell_contributions"],
            burden_cell_contributions=c,
            cell_contributions=parent["cell_contributions"] + c,
            instance_logits=parent["cell_contributions"] + c,
            contribution_intercept=parent["contribution_intercept"] + self.readout_intercept,
            parent_intercept=parent["contribution_intercept"],
            burden_intercept=self.readout_intercept.expand(len(features)),
            burden_activity=q,
            burden_union_activity=union,
            burden_features=pooled,
            burden_local_evidence=raw,
            exposure_slide_equivalents=exposure,
            burden_activity_sum=q.sum(1),
            burden_activity_rate=q.sum(1) / exposure[:, None],
            burden_selected_cells=((q >= 0.5).any(-1) & attention_mask).sum(-1),
        )
        if self.training:
            with torch.no_grad():
                valid = q[attention_mask]
                self._last_activity = {
                    "last_microbatch_mean_activity": valid.mean(0).tolist()
                    if len(valid)
                    else [0.0] * 4,
                    "last_microbatch_mass_features": pooled.mean(0).tolist(),
                }
        return result

    def record_gradients(self):
        for name, module in [("projection", self.projection), ("activity", self.activity_head)]:
            grads = [
                p.grad.detach().float().square().sum()
                for p in module.parameters()
                if p.grad is not None
            ]
            self._last_gradients[name] = float(torch.stack(grads).sum().sqrt()) if grads else 0.0
        self._last_gradients["readout"] = (
            float(self.readout_weight.grad.norm()) if self.readout_weight.grad is not None else 0.0
        )

    def diagnostics(self):
        return {
            "readout_coefficients": self.readout_weight.detach().tolist(),
            "residual_intercept": float(self.readout_intercept.detach()),
            "feature_scale": self.feature_scale.tolist(),
            "gradients_before_clip": self._last_gradients,
            **self._last_activity,
        }
