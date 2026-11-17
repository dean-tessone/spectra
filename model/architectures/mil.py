"""Matched gated ABMIL/Additive MIL with an exact signed logit decomposition.

This experiment does not modify frozen historical SPECTRA model implementations.
"""
import torch
from torch import nn
from spectra.model import ABMILAttention, _stable_cellwise_linear, _entmax_bisect


class InterpretableMIL(nn.Module):
    def __init__(self, additive=False, sparse=False):
        super().__init__()
        self.additive=additive;self.sparse=sparse;self.epoch=1
        self.projection=nn.Linear(512,128)
        self.activation=nn.GELU();self.dropout=nn.Dropout(.2)
        self.attention=ABMILAttention(128,64,gated=True)
        self.norm=nn.LayerNorm(128);self.classifier=nn.Linear(128,1)
    def set_training_epoch(self,epoch):self.epoch=int(epoch)
    @property
    def alpha(self):return 1+.5*min(max((self.epoch-1)/9,0),1) if self.sparse else 1.
    def encode_cells(self,x):
        return self.activation(_stable_cellwise_linear(self.projection,x.float()))
    def forward(self,x,mask,segments=None):
        # An empty filtered bag has zero attention/evidence and the existing
        # classifier intercept. Never inject a fake retained cell or drop a donor.
        h=self.dropout(self.encode_cells(x))
        scores=self.attention(h).float().masked_fill(~mask,torch.finfo(torch.float32).min)
        a=_entmax_bisect(scores,alpha=self.alpha,dim=-1).float().masked_fill(~mask,0)
        # All attention, pooling, normalization and credit computations are FP32.
        bag=(a.unsqueeze(-1)*h.float()).sum(1)
        if self.additive:
            normalized=self.norm(h.float())
            cell_score=_stable_cellwise_linear(self.classifier,normalized).squeeze(-1)
            # The classifier bias is separate and appears exactly once.
            credit=a*(cell_score-self.classifier.bias[0])
            intercept=self.classifier.bias[0].expand(len(x))
            z=(a.unsqueeze(-1)*normalized).sum(1)
            logits=_stable_cellwise_linear(self.classifier,z[:,None,:]).flatten()
        else:
            z=self.norm(bag)
            logits=self.classifier(z).flatten()
            # LayerNorm's effective linear map depends on the complete bag.
            std=(bag.var(-1,unbiased=False)+self.norm.eps).sqrt()
            w=self.classifier.weight[0]*self.norm.weight
            centered_w=w-w.mean()
            credit=a*(h.float()*centered_w[None,None,:]).sum(-1)/std[:,None]
            intercept=(self.classifier.weight[0]*self.norm.bias).sum()+self.classifier.bias[0]
            intercept=intercept.expand(len(x))
        credit=credit.masked_fill(~mask,0)
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite MIL logit')
        return dict(logits=logits,cls_embedding=z,attention_weights=a,
            cell_contributions=credit,contribution_intercept=intercept,
            attention_support_count=(a>0).sum(1),alpha=self.alpha)
