"""Masked mean of frozen cell embeddings, followed by a small MLP head."""
import torch
from torch import nn

class MeanPoolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(512,128)
        self.activation=nn.GELU()
        self.dropout=nn.Dropout(.2)
        self.norm=nn.LayerNorm(128)
        self.classifier=nn.Linear(128,1)

    def set_training_epoch(self,epoch):
        pass

    def forward(self,x,mask,segments=None):
        if x.ndim!=3 or mask.shape!=x.shape[:2]:
            raise ValueError('Expected B x cells x 512 and matching cell mask')
        values=x.float().masked_fill(~mask.unsqueeze(-1),0.)
        counts=mask.sum(1,keepdim=True).clamp_min(1)
        pooled=values.sum(1)/counts
        h=self.norm(self.dropout(self.activation(self.projection(pooled))))
        logits=self.classifier(h).flatten()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite mean-pooling logit')
        return {'logits':logits,'cls_embedding':h}
