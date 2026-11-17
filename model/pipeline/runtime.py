"""Shared cache, donor sampling and models; no historical cohort controllers."""
import importlib.util
import math
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader
from spectra.data import group_collate
from common import *

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
MODEL=module('revised_transformer_model',REPO/'model/architectures/transformer.py')
MIL=module('revised_mil_model',REPO/'model/architectures/mil.py')
MEAN=module('revised_mean_model',REPO/'model/architectures/mean_pool.py')
SB=module('revised_clam_sb',REPO/'model/architectures/clam_sb.py')
MB=module('revised_clam_mb',REPO/'model/architectures/clam_mb.py')
RECIPES={f'd{d}_do{int(p*10):02d}_lr{tag}':dict(depth=d,dropout=p,lr=lr) for d in (3,6) for p in (.1,.4) for lr,tag in ((2e-4,'2e4'),(5e-5,'5e5'))}
TYPES=['Bladder','Bone','Breast','Lung','Multiple Myeloma','Pancreatic','Prostate','Upper Tract Urothelial Carcinoma','Uveal Melanoma']

def model_for(family,recipe):
    if family=='transformer':
        cfg=RECIPES[recipe];return MODEL.CLSTransformer(input_dim=512,width=256,heads=4,expansion=4,depth=cfg['depth'],dropout=cfg['dropout'])
    return {'abmil':lambda:MIL.InterpretableMIL(additive=False,sparse=False),'additive':lambda:MIL.InterpretableMIL(additive=True,sparse=False),'mean':MEAN.MeanPoolClassifier,'clam_sb':SB.CLAMSingleBranch,'clam_mb':MB.CLAMMultiBranch}[family]()

def lr_factor(step):
    if step<64:return (step+1)/64
    return .05+.95*.5*(1+math.cos(math.pi*min(max((step-64)/(30*64-64),0),1)))

class Cache:
    def __init__(self,donors):
        donors=set(donors)
        manifest=pd.read_csv(ROOT/'cohort/all_groups.csv',dtype={'slide_id':str,'donor_id':str})
        manifest=manifest[manifest.donor_id.isin(donors)]
        assert set(manifest.donor_id)==donors
        receipt=read(ROOT/'data/cache_receipt.json')['slides']
        self.entries={};self.indices={};self.raw={};self.sources={}
        for gid,g in manifest.groupby('group_id',sort=True):
            e=g.iloc[0]
            self.entries[gid]=dict(donor_id=e.donor_id,label=int(e.label),cancer_types=e.cancer_types,study=e.study,primary_evaluation=bool(e.primary_evaluation),slide_ids=g.slide_id.tolist(),lengths=[receipt[s]['retained_cells'] for s in g.slide_id])
            self.indices[gid]=[np.load(receipt[s]['indices'],allow_pickle=False).astype(np.int64) for s in g.slide_id]
            for s in g.slide_id:
                r=receipt[s];self.raw[s]=r['raw_cache'];self.sources[s]=r['source']
                for field in ['source','raw_cache','qc_cache']:
                    st=local(r[field]).stat();assert [st.st_size,st.st_mtime_ns]==r[field+'_stat']
                assert sha(r['indices'])==r['indices_sha256']
        self.meta=pd.DataFrame.from_dict(self.entries,orient='index')
    def sample(self,gid):
        e=self.entries[gid]
        parts=[np.load(self.raw[s],mmap_mode='r',allow_pickle=False)[idx] for s,idx in zip(e['slide_ids'],self.indices[gid])]
        x=np.concatenate(parts)
        if not len(x) or not np.isfinite(x).all():raise ValueError('Empty/nonfinite bag')
        return dict(features=torch.from_numpy(x),attention_mask=torch.ones(len(x),dtype=torch.bool),segment_ids=torch.repeat_interleave(torch.arange(len(parts)),torch.tensor(e['lengths'])),slide_lengths=torch.tensor(e['lengths']),label=torch.tensor(float(e['label'])),group_id=gid,entity_id=e['donor_id'],slide_ids=e['slide_ids'],slide_paths=[self.sources[s] for s in e['slide_ids']],slide_row_indices=tuple(torch.from_numpy(i.copy()) for i in self.indices[gid]),qc_unusable_scores=None)

class Bags(Dataset):
    def __init__(self,cache,items):self.cache=cache;self.items=items
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        g,w=self.items[i];d=self.cache.sample(g);d['sample_weight']=w;return d

def loader(cache,batches=None,ids=None):
    kwargs=dict(num_workers=2,collate_fn=group_collate,pin_memory=True,persistent_workers=True,generator=torch.Generator().manual_seed(20261109))
    if batches is not None:
        items=[v for b in batches for v in b];ends=np.cumsum([0]+[len(b) for b in batches]);sampler=[list(range(int(a),int(b))) for a,b in zip(ends[:-1],ends[1:])]
        return DataLoader(Bags(cache,items),batch_sampler=sampler,**kwargs)
    return DataLoader(Bags(cache,[(g,1.) for g in ids]),batch_size=1,shuffle=False,**kwargs)

class Queue:
    def __init__(self,ids,rng):self.ids=sorted(ids);self.rng=rng;self.pending=[]
    def draw(self,n):
        assert n<=len(self.ids)
        chosen=[];deferred=[]
        while len(chosen)<n:
            if not self.pending:self.pending=self.rng.permutation(self.ids).tolist()
            v=self.pending.pop(0)
            if v in chosen:deferred.append(v)
            else:chosen.append(v)
        self.pending=deferred+self.pending;return chosen

def schedule(cache,seed,epochs):
    people=cache.meta.drop_duplicates('donor_id');negative=people[people.label.eq(0)];positive=people[people.label.eq(1)]
    assert len(negative)>=32 and len(positive)>=32
    rng=np.random.default_rng(seed);neg=Queue(negative.donor_id,np.random.default_rng([seed,9]))
    groups={p:Queue(g.index,np.random.default_rng([seed,535+i])) for i,(p,g) in enumerate(cache.meta.groupby('donor_id',sort=True))}
    strata=list(positive.groupby('cancer_types',sort=True));sizes=np.array([len(g) for _,g in strata]);mass=sizes/np.sqrt(np.maximum(sizes,20));mass/=mass.sum();credit=np.zeros(len(mass));queues=[Queue(g.donor_id,np.random.default_rng([seed,100+i])) for i,(_,g) in enumerate(strata)]
    output=[]
    for epoch in range(epochs):
        batches=[]
        for _ in range(64):
            ns=neg.draw(32);credit+=32*mass;quota=np.zeros(len(mass),int);tie=rng.permutation(len(mass))
            for _ in range(32):
                allowed=tie[quota[tie]<sizes[tie]];i=allowed[np.argmax(credit[allowed])];quota[i]+=1;credit[i]-=1
            ps=[p for q,n in zip(queues,quota) for p in q.draw(int(n))];rng.shuffle(ps)
            assert len(set(ns+ps))==64
            for i in range(0,32,4):
                ids=ns[i:i+4]+ps[i:i+4];rng.shuffle(ids)
                batches.append([(groups[p].draw(1)[0],1/64) for p in ids])
        output.append(batches)
    return output

def forward(model,family,batch,mean,scale):
    mask=batch['attention_mask'].cuda(non_blocking=True)
    x=((batch['features'].cuda(non_blocking=True).float()-mean)/scale).half().float().masked_fill(~mask.unsqueeze(-1),0.)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=family=='transformer'):out=model(x,mask)
    return out,mask

def metrics(d):
    from sklearn.metrics import roc_auc_score,log_loss
    result=dict(pooled_auroc=float(roc_auc_score(d.label,d.probability)))
    result['per_cancer_auroc']={t:float(roc_auc_score(s.label,s.probability)) for t in sorted(d[d.label.eq(1)].cancer_types.unique()) if (s:=d[d.label.eq(0)|d.cancer_types.eq(t)]).label.nunique()==2}
    result['macro']=float(np.mean([v for k,v in result['per_cancer_auroc'].items() if k in TYPES]))
    result['balanced_log_loss']=float(np.mean([log_loss(g.label,g.probability,labels=[0,1]) for _,g in d.groupby('label')]))
    return result

@torch.no_grad()
def score(model,family,cache,mean,scale,primary=True):
    model.eval();ids=[g for g,e in cache.entries.items() if not primary or e['primary_evaluation']];rows=[]
    dl=loader(cache,ids=ids)
    for batch in dl:
        out,_=forward(model,family,batch,mean,scale);gid=batch['group_id'][0];e=cache.entries[gid]
        rows.append(dict(group_id=gid,**{k:e[k] for k in ['donor_id','label','cancer_types','study','primary_evaluation']},retained_cells=sum(e['lengths']),logit=float(out['logits'][0]),probability=float(out['logits'].sigmoid()[0])))
    del dl
    return pd.DataFrame(rows)
