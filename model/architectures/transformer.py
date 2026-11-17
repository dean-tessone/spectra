"""Permutation-invariant full-context transformer with supervised CLS readout."""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

CONFIG = dict(input_dim=512, width=256, depth=3, heads=4, expansion=4, dropout=.1)


class Block(nn.Module):
    def __init__(self, width, heads, expansion, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, expansion * width), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(expansion * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, diagnostics=False, fp32_attention=False):
        batch, length, width = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized = self.norm1(x.float())
        qkv = self.qkv(normalized).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        allowed = mask[:, None, None, :]
        backend = (sdpa_kernel(SDPBackend.MATH) if fp32_attention else
            sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])) if x.is_cuda else nullcontext()
        with backend:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        x = x + self.dropout(self.out(attended))
        with torch.autocast(device_type=x.device.type, enabled=False):
            normalized = self.norm2(x.float())
        x = x + self.dropout(self.ffn(normalized))
        attention = None
        if diagnostics:
            with torch.autocast(device_type=x.device.type, enabled=False):
                scores = (q[:, :, :1].float() @ k.float().transpose(-1, -2)) / (width // self.heads) ** .5
                attention = scores.masked_fill(~allowed, -torch.inf).softmax(-1).squeeze(-2)
        return x, attention


class CLSTransformer(nn.Module):
    def __init__(self, input_dim=512, width=256, depth=3, heads=4, expansion=4, dropout=.1):
        super().__init__()
        if width % heads:
            raise ValueError('Width must divide into attention heads')
        self.projection = nn.Linear(input_dim, width)
        self.cls = nn.Parameter(torch.empty(1, 1, width))
        nn.init.normal_(self.cls, std=.02)
        self.blocks = nn.ModuleList([Block(width, heads, expansion, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 1)

    def forward(self, features, mask, diagnostics=False, fp32_attention=False):
        if mask.dtype != torch.bool or mask.shape != features.shape[:2]:
            raise ValueError('Boolean cell mask must match features')
        h = self.projection(features.float())
        cls = self.cls.expand(len(features), -1, -1).to(h.dtype)
        h = torch.cat([cls, h], dim=1)
        full_mask = torch.cat([torch.ones(len(features), 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        attention = None
        for i, block in enumerate(self.blocks):
            h, attention = block(h, full_mask, diagnostics and i == len(self.blocks) - 1, fp32_attention)
        with torch.autocast(device_type=features.device.type, enabled=False):
            z = self.norm(h[:, 0].float())
            logits = self.classifier(z).flatten()
        result = dict(logits=logits, cls_embedding=z)
        if attention is not None:
            result['cls_self_attention'] = attention[:, :, 0]
            result['cls_cell_attention'] = attention[:, :, 1:].masked_fill(~mask[:, None, :], 0.)
        return result
