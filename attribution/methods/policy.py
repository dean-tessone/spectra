"""Physical-cell aggregation, positive relevance selection and exemplar policies."""
import numpy as np
import pandas as pd
from scipy.special import expit


def probability_relevance(logit_relevance,logit):
    # Stable sigmoid derivative avoids cancellation near one.
    return np.asarray(logit_relevance,dtype=float)*expit(logit)*expit(-logit)


def select_cells(frame,top_n=50,mass=.9):
    if not 0<mass<=1:raise ValueError('Invalid relevance mass target')
    d=frame.copy()
    if not np.isfinite(d.relevance).all():raise ValueError('Nonfinite relevance')
    d=d.sort_values(['relevance','slide_id','source_row'],ascending=[False,True,True]).reset_index(drop=True)
    positive=d.relevance.gt(0);total=float(d.loc[positive,'relevance'].sum())
    d['positive_rank']=0;d.loc[positive,'positive_rank']=np.arange(1,int(positive.sum())+1)
    d['positive_fraction']=np.maximum(d.relevance.to_numpy(),0)/total if total>0 else 0.
    cumulative=np.cumsum(np.maximum(d.relevance.to_numpy(),0),dtype=float)
    d['cumulative_positive_fraction']=cumulative/total if total>0 else 0.
    n90=min(int(positive.sum()),int(np.searchsorted(cumulative[:int(positive.sum())],np.nextafter(mass*total,-np.inf),side='left'))+1) if total>0 else 0
    d['top50']=positive & d.positive_rank.le(top_n)
    d['mass90']=positive & d.positive_rank.le(n90)
    return d,dict(cells=len(d),positive_cells=int(positive.sum()),top50_cells=int(d.top50.sum()),mass90_cells=int(d.mass90.sum()),
        positive_relevance_total=total,top50_mass=float(d.loc[d.top50,'relevance'].sum()/total) if total>0 else 0.,
        mass90_covered=float(d.loc[d.mass90,'relevance'].sum()/total) if total>0 else 0.)


def relabel_by_size(labels):
    labels=np.asarray(labels);values,counts=np.unique(labels[labels>=0],return_counts=True)
    ordered=sorted(zip(values,counts),key=lambda q:(-q[1],q[0]));mapping={int(v):i for i,(v,_) in enumerate(ordered)}
    return np.array([mapping[int(v)] if v>=0 else -1 for v in labels],dtype=int)


def exemplars(frame,pca,seed=20261020):
    output=[]
    for cluster,group in frame.groupby('cluster',sort=True):
        ids=group.index.to_numpy();rng=np.random.default_rng([seed,int(cluster)+1])
        if cluster==-1:
            order=rng.permutation(ids);modes=['random_noise']*5
        else:
            center=np.median(pca[ids],axis=0);order=ids[np.argsort(((pca[ids]-center)**2).sum(axis=1),kind='stable')]
            modes=['near_PCA_median']*3+['random']*2
        chosen=[];donors=set()
        for rank,mode in enumerate(modes,1):
            pool=order if mode!='random' else rng.permutation(ids)
            eligible=[i for i in pool if i not in chosen]
            if not eligible:break
            novel=[i for i in eligible if frame.loc[i,'donor_id'] not in donors]
            i=(novel or eligible)[0];chosen.append(i);donors.add(frame.loc[i,'donor_id'])
            row=frame.loc[i].to_dict();row.update(exemplar_rank=rank,exemplar_rule=mode)
            output.append(row)
    return pd.DataFrame(output)
