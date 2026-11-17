"""Batched CLAM-SB with class-specific instance clustering.

Algorithm/architecture reference (independently implemented for padded cell bags):
https://github.com/mahmoodlab/CLAM/blob/master/models/model_clam.py
https://github.com/mahmoodlab/CLAM/blob/master/main.py
Small architecture: embed_dim->512->256 gated attention, ReLU, dropout .25,
2-way bag classifier and two class-specific 2-way instance classifiers.
"""
import torch
from torch import nn
from torch.nn import functional as F

class CLAMSingleBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(512,512)
        self.activation=nn.ReLU()
        self.dropout=nn.Dropout(.25)
        self.attention_a=nn.Sequential(nn.Linear(512,256),nn.Tanh(),nn.Dropout(.25))
        self.attention_b=nn.Sequential(nn.Linear(512,256),nn.Sigmoid(),nn.Dropout(.25))
        self.attention_c=nn.Linear(256,1)
        self.classifier=nn.Linear(512,2)
        self.instance_classifiers=nn.ModuleList([nn.Linear(512,2),nn.Linear(512,2)])
        self.k_sample=8

    def set_training_epoch(self,epoch):
        pass

    def forward(self,x,mask,segments=None):
        h=self.dropout(self.activation(self.projection(x.float())))
        raw=self.attention_c(self.attention_a(h)*self.attention_b(h)).squeeze(-1)
        raw=raw.masked_fill(~mask,torch.finfo(raw.dtype).min)
        attention=raw.softmax(-1).masked_fill(~mask,0.)
        pooled=(attention.unsqueeze(-1)*h).sum(1)
        two_logits=self.classifier(pooled)
        # sigmoid(logit1-logit0) == two-class softmax(logits)[1].
        logits=two_logits[:,1]-two_logits[:,0]
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite CLAM bag score')
        return dict(logits=logits,bag_logits=two_logits,attention_weights=attention,instance_features=h,cls_embedding=pooled)

    def instance_losses(self,out,labels,mask):
        losses=[]
        for b in range(len(labels)):
            valid=mask[b].bool();h=out['instance_features'][b,valid]
            scores=out['attention_weights'][b,valid].detach()
            n=len(h)
            if not n:
                losses.append(out['logits'][b]*0.)
                continue
            # k=8 when N>=16, matching the authors' standard top/bottom selection.
            # Small bags reduce k to avoid overlapping positive/negative pseudo-labels.
            k=min(self.k_sample,max(n//2,1))
            order=torch.argsort(scores,descending=True,stable=True)
            positive=order[:k]
            negative=order[-k:] if n>1 else order[:0]
            selected=torch.cat([positive,negative])
            target=torch.cat([torch.ones(len(positive),device=h.device,dtype=torch.long),
                              torch.zeros(len(negative),device=h.device,dtype=torch.long)])
            branch=int(labels[b].detach().item())
            logits=self.instance_classifiers[branch](h[selected])
            losses.append(F.cross_entropy(logits,target))
        return torch.stack(losses)


def training_losses(model,out,labels,mask):
    bag=F.binary_cross_entropy_with_logits(out['logits'],labels.float(),reduction='none')
    instance=model.instance_losses(out,labels,mask)
    return .7*bag+.3*instance
