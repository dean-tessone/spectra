"""Jointly learned cell populations with an exactly decomposable quadratic readout.

Membership is a phenotype assignment, not a cell malignancy probability. Pairwise
credit is split equally between the two participating populations. Frequencies
refer to the observed selected-cell bag, not a concentration in whole blood.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class PopulationAbundance(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.population_count < 1 or config.population_frequency_scale <= 0:
            raise ValueError("positive population count and frequency scale required")
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        # Last category is background; all categories remain label-agnostic phenotypes.
        self.population_detector = nn.Linear(config.model_dim, config.population_count + 1)
        nn.init.normal_(self.population_detector.weight, std=0.02)
        nn.init.zeros_(self.population_detector.bias)
        with torch.no_grad():
            self.population_detector.bias[-1] = 3.0
        self.population_weight = nn.Parameter(torch.empty(config.population_count))
        nn.init.normal_(self.population_weight, std=0.02)
        pairs = torch.triu_indices(config.population_count, config.population_count, offset=1)
        self.register_buffer("pair_indices", pairs)
        self.interaction_weight = nn.Parameter(
            torch.zeros(pairs.shape[1]), requires_grad=config.population_interactions
        )
        self.intercept = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, attention_mask=None, segment_ids=None, *,
                return_attention=False, qc_unusable_scores=None):
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("population model expects [batch, cells, input_dim]")
        if attention_mask is None:
            attention_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if attention_mask.shape != features.shape[:2]:
            raise ValueError("population mask shape mismatch")
        n = attention_mask.sum(1)
        if bool((n == 0).any()):
            raise ValueError("population model requires nonempty bags")
        clean = features.float().masked_fill(~attention_mask[..., None], 0.0)
        if not torch.isfinite(clean).all():
            raise ValueError("nonfinite population features")
        membership = torch.cat([
            self.population_detector(self.dropout(F.gelu(self.input_projection(block)))).softmax(-1)
            for block in clean.split(self.config.additive_cell_chunk_size, dim=1)
        ], dim=1).masked_fill(~attention_mask[..., None], 0.0)
        local_frequency = membership[..., :-1] * (
            self.config.population_frequency_scale / n.float()
        )[:, None, None]
        abundance = local_frequency.sum(1)
        main = abundance * self.population_weight
        left, right = self.pair_indices
        weight = self.interaction_weight if self.config.population_interactions else self.interaction_weight * 0
        interactions = abundance[:, left] * abundance[:, right] * weight
        context = self.population_weight.expand_as(abundance).clone()
        context.index_add_(1, left, 0.5 * abundance[:, right] * weight)
        context.index_add_(1, right, 0.5 * abundance[:, left] * weight)
        population_cell_credit = local_frequency * context[:, None, :]
        cell_credit = population_cell_credit.sum(-1)
        logits = self.intercept + main.sum(-1) + interactions.sum(-1)
        output = {
            "logits": logits,
            "cls_embedding": abundance,
            "cross_attentions": (),
            "cell_contributions": cell_credit,
            "instance_logits": cell_credit,
            "contribution_intercept": self.intercept.expand_as(logits),
            "population_abundance": abundance,
            "population_counts": membership[..., :-1].sum(1),
            "population_main_effects": main,
            "population_interaction_effects": interactions,
            "population_total_effects": population_cell_credit.sum(1),
            "population_pair_indices": self.pair_indices,
        }
        if return_attention:
            output["population_membership"] = membership
            output["population_cell_contributions"] = population_cell_credit
        return output
