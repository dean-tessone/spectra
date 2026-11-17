"""Cell scores and perturbations; inputs are standardized event embeddings.

No training or model parameter mutation. Efficient LRP is an independent
architecture-specific implementation of the rules described by Achtibat et al.
(ICML2024) and their efficient LXT formulation.
"""
import math
import time

import numpy as np
import torch
from torch.nn import functional as F


def _norm(x, layer, relevance=False):
    if not relevance:
        return layer(x.float())
    x = x.float()
    centered = x - x.mean(-1, keepdim=True)
    inverse = torch.rsqrt(centered.square().mean(-1, keepdim=True) + layer.eps).detach()
    return centered * inverse * layer.weight + layer.bias


class RelevanceGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = F.gelu(x)
        # GELU(x)/x is Phi(x), with a well-defined limit at x=0.
        ctx.save_for_backward(.5 * (1 + torch.erf(x / math.sqrt(2))))
        return y

    @staticmethod
    def backward(ctx, dy):
        return dy * ctx.saved_tensors[0]


def _gradient_fraction(x, fraction):
    return x * fraction + x.detach() * (1 - fraction)


def explanatory_forward(model, x, mask, relevance=False, rollout=False):
    """Float32 replay with optional relevance rules/full attention rollout.

    Rule changes affect derivatives only. Rollout includes every attention layer,
    residual identities and row normalization; it is not final-layer CLS attention.
    Chunk queries when materializing attention to bound temporary GPU memory.
    """
    if model.training:
        raise ValueError('Explanations require eval mode')
    h = torch.cat([model.cls.expand(len(x), -1, -1), model.projection(x.float())], 1)
    full = torch.cat([torch.ones(len(x), 1, device=x.device, dtype=torch.bool), mask], 1)
    matrices = []
    for block in model.blocks:
        b, n, w = h.shape
        qkv = block.qkv(_norm(h, block.norm1, relevance))
        q, k, v = qkv.reshape(b, n, 3, block.heads, w // block.heads).permute(2, 0, 3, 1, 4).unbind(0)
        if relevance:
            q, k, v = _gradient_fraction(q, .25), _gradient_fraction(k, .25), _gradient_fraction(v, .5)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=full[:, None, None, :], dropout_p=0.)
        if rollout:
            if b != 1:
                raise ValueError('Rollout is one bag at a time')
            rows = []
            for start in range(0, n, 256):
                s = q[:, :, start:start+256] @ k.transpose(-1, -2) / math.sqrt(w // block.heads)
                p = s.masked_fill(~full[:, None, None, :], -torch.inf).softmax(-1).mean(1)[0]
                p[torch.arange(len(p), device=x.device), torch.arange(start, start+len(p), device=x.device)] += 1
                rows.append(p / p.sum(-1, keepdim=True))
            matrices.append(torch.cat(rows))
        h = h + block.out(a.transpose(1, 2).reshape(b, n, w))
        z = block.ffn[0](_norm(h, block.norm2, relevance))
        z = RelevanceGELU.apply(z) if relevance else block.ffn[1](z)
        h = h + block.ffn[3](z)
    z = _norm(h[:, 0], model.norm, relevance)
    logit = model.classifier(z).flatten()
    if not rollout:
        return logit
    weights = torch.zeros(h.shape[1], device=x.device); weights[0] = 1
    for matrix in reversed(matrices):
        weights = weights @ matrix
    return logit, weights[1:].masked_fill(~mask[0], 0.)


def scalar(model, x, mask, target='probability', reference_logit=None, relevance=False):
    logit = explanatory_forward(model, x, mask, relevance=relevance).reshape(())
    if target == 'probability':
        return logit.double().sigmoid()
    if target == 'logit':
        return logit.double()
    if target == 'contrast':
        return (logit.double() - reference_logit).square()
    raise ValueError(target)


def integrated(model, x, mask, reference, nodes=32, target='probability'):
    delta = (x - reference).detach().masked_fill(~mask.unsqueeze(-1), 0.)
    with torch.no_grad():
        ref_logit = scalar(model, reference, mask, 'logit')
        start = scalar(model, reference, mask, target, ref_logit)
        end = scalar(model, x, mask, target, ref_logit)
    points, weights = np.polynomial.legendre.leggauss(nodes)
    credit = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
    for point, weight in zip((points+1)/2, weights/2):
        value = (reference + float(point)*delta).detach().requires_grad_(True)
        score = scalar(model, value, mask, target, ref_logit)
        gradient, = torch.autograd.grad(score, value)
        credit += float(weight)*(gradient*delta).sum(-1)[0].double()
    residual = float(end-start-credit.sum())
    return credit.detach().cpu().numpy(), dict(nodes=nodes, target=target,
        score=float(end), reference_score=float(start), residual=residual,
        completeness_passed=abs(residual) <= (0.001 if target == 'probability' else max(.01,.01*abs(float(end-start)))))


def top_overlap(a, b, k=10):
    k = min(k, len(a))
    return len(set(np.argsort(-a, kind='stable')[:k]) & set(np.argsort(-b, kind='stable')[:k]))/k


def adaptive_ig(model, x, mask, reference, target='probability'):
    previous, diagnostics = integrated(model, x, mask, reference, 32, target)
    attempts = [diagnostics]
    for nodes in (64, 128):
        current, diagnostics = integrated(model, x, mask, reference, nodes, target)
        diagnostics['top10_overlap_previous'] = top_overlap(previous, current)
        attempts.append(diagnostics)
        if diagnostics['completeness_passed'] and diagnostics['top10_overlap_previous'] >= .8:
            return current, dict(converged=True, attempts=attempts)
        previous = current
    return current, dict(converged=False, attempts=attempts)


def gradient_scores(model, x, mask, relevance=False):
    value = x.detach().requires_grad_(True)
    target = 'logit' if relevance else 'probability'
    score = scalar(model, value, mask, target, relevance=relevance)
    gradient, = torch.autograd.grad(score, value)
    credit = (gradient*value).sum(-1)[0].detach().double().cpu().numpy()
    return credit, dict(target=target, output=float(score.detach()),
                        cell_sum=float(credit.sum()), unallocated_output=float(score.detach())-float(credit.sum()))


@torch.no_grad()
def singleton_scores(model, x, chunk=128):
    result = []
    for part in x[0].split(chunk):
        values = explanatory_forward(model, part[:, None], torch.ones(len(part), 1, device=x.device, dtype=torch.bool))
        result.append(values.double().sigmoid().cpu().numpy())
    return np.concatenate(result)


@torch.no_grad()
def cell_effects(model, x, mask, replacements, chunk=8):
    """Every cell, no candidate preselection. Replace with four draws or remove."""
    full = float(scalar(model, x, mask))
    removed, replaced = [], []
    for i in range(x.shape[1]):
        keep = torch.arange(x.shape[1], device=x.device) != i
        removed.append(full-float(scalar(model, x[:, keep], mask[:, keep])))
        samples = []
        for reference in replacements:
            modified = x.clone(); modified[:, i] = reference[:, i]
            samples.append(float(scalar(model, modified, mask)))
        replaced.append(full-float(np.mean(samples)))
    return np.asarray(removed), np.asarray(replaced)


@torch.no_grad()
def perturb(model, x, mask, indices, replacement=None, retain=False):
    if retain:
        return float(scalar(model, x[:, indices], mask[:, indices]))
    if replacement is None:
        keep = torch.ones(x.shape[1], device=x.device, dtype=torch.bool); keep[indices] = False
        return float(scalar(model, x[:, keep], mask[:, keep]))
    modified = x.clone(); modified[:, indices] = replacement[:, indices]
    return float(scalar(model, modified, mask))
