"""Summaries of how adaptation changes cell attribution (descriptive).

Per test patient and head (source / probe / adapt): positive logit relevance per cell -> share per Leiden
cluster (cells assigned to the Figure 3 atlas clusters by 15-NN in the atlas PCA50 space); top-50 sets
and their overlap (Jaccard) between source and adapted heads.
"""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read  # noqa: E402
RUN = REPO / 'runs/specific_adaptation_20261111/attribution'
W = REPO / 'runs/figure3_locked6_attribution_20261111/mass90_leiden'
OUT = REPO / 'analyses/specific_adaptation_20261111/attribution'; OUT.mkdir(parents=True, exist_ok=True)


def assign(cells):
    from sklearn.neighbors import KNeighborsClassifier
    receipt = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
    lock = read(REPO / 'runs/reference_control_full_refit_20261110/model_lock.json')
    with np.load(lock['normalizer']) as z:
        mu, sd = z['mean'], z['scale']
    with np.load(W / 'pca_transform.npz') as z:
        pm, pc = z['mean'], z['components']
    knn = KNeighborsClassifier(15).fit(np.load(W / 'mass90_pca50.npy'), np.load(W / 'cluster_primary.npy'))
    out = np.empty(len(cells), int)
    for sid, g in cells.groupby('slide_id'):
        raw = np.load(receipt[sid]['raw_cache'], mmap_mode='r')
        x = ((np.asarray(raw[g.source_row.to_numpy()], np.float32) - mu) / sd - pm) @ pc.T
        out[g.index.to_numpy()] = knn.predict(x)
    return out


def main():
    summary = {}
    for task in ['lung', 'breast']:
        c = pd.read_parquet(RUN / f'{task}_cells_private.parquet').reset_index(drop=True)
        c['cluster'] = assign(c)
        heads = ['source', 'probe', 'adapt']
        rows = []; jac = []
        for d, g in c.groupby('donor_id'):
            lab = int(g.label.iloc[0])
            for h in heads:
                pos = g[h].clip(lower=0); tot = pos.sum()
                share = (pos.groupby(g.cluster).sum() / tot) if tot > 0 else pos.groupby(g.cluster).sum() * 0
                for k, v in share.items():
                    rows.append(dict(donor_id=d, label=lab, head=h, cluster=int(k), share=float(v)))
            top = {h: set(g.nlargest(50, h).index) for h in heads}
            for h in ['probe', 'adapt']:
                jac.append(dict(donor_id=d, label=lab, head=h, jaccard_top50=len(top['source'] & top[h]) / len(top['source'] | top[h])))
        s = pd.DataFrame(rows).pivot_table(index=['donor_id', 'label', 'cluster'], columns='head', values='share', fill_value=0).reset_index()
        s.to_csv(OUT / f'{task}_cluster_share_private.csv', index=False)
        agg = s.groupby(['label', 'cluster'])[heads].mean().reset_index()
        for h in ['probe', 'adapt']:
            agg[f'delta_{h}'] = agg[h] - agg['source']
        agg.to_csv(OUT / f'{task}_cluster_share_shift.csv', index=False)
        pd.DataFrame(jac).to_csv(OUT / f'{task}_top50_jaccard.csv', index=False)
        summary[task] = dict(patients=int(c.donor_id.nunique()), cells=len(c), median_jaccard={h: float(pd.DataFrame(jac).query('head==@h').jaccard_top50.median()) for h in ['probe', 'adapt']})
        print(task, summary[task]); print(agg.sort_values('delta_probe').round(4).to_string())
    json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
