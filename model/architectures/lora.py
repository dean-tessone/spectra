"""Query/value LoRA for the existing fused-QKV SPECTRA transformer."""
import torch
from torch import nn
from torch.nn import functional as F


class QueryValueLoRA(nn.Module):
    def __init__(self, base, rank=4, alpha=8):
        super().__init__()
        if base.out_features != 3 * base.in_features or rank < 1:
            raise ValueError('Expected fused square query/key/value projections')
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        width = base.in_features
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.q_a = nn.Parameter(base.weight.new_empty(rank, width))
        self.q_b = nn.Parameter(base.weight.new_zeros(width, rank))
        self.v_a = nn.Parameter(base.weight.new_empty(rank, width))
        self.v_b = nn.Parameter(base.weight.new_zeros(width, rank))
        nn.init.normal_(self.q_a, std=.02)
        nn.init.normal_(self.v_a, std=.02)

    def forward(self, x):
        q = F.linear(F.linear(x, self.q_a), self.q_b) * self.scale
        v = F.linear(F.linear(x, self.v_a), self.v_b) * self.scale
        return self.base(x) + torch.cat([q, torch.zeros_like(q), v], dim=-1)


def install(model, rank=4, alpha=8):
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.blocks:
        block.qkv = QueryValueLoRA(block.qkv, rank=rank, alpha=alpha)
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    return model


def training_mode(model, arm):
    if arm == 'lora':
        # Preserve the frozen source computation; adapters have no dropout.
        model.eval()
        model.classifier.train()
    else:
        model.train()
        if arm == 'adapt':
            for block in model.blocks[:-1]:
                block.eval()
