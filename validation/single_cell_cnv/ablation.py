"""Copy-number ablation on the locked model: among high-attention cells that were sequenced, does
deleting copy-number-ALTERED cells lower the patient score more than deleting FLAT cells? Inference only; locked
three-seed ensemble, BF16 autocast and FP16 input rounding exactly as the official replay (figure3 removal.py).

Score = log-odds of the equal-weight three-seed probability mean (as Fig. 2c).
  1. Leave-one-out: for every sequenced model-input cell, delete that one cell from its patient's bag; delta = intact
     score minus score without it (positive = the cell pushed the score up).
  2. Group deletion, patients with >=1 altered AND >=1 flat sequenced high-attention (mass-90) cell: delete all
     altered vs all flat high-attention sequenced cells, each against 20 draws of the same number of random cells
     from that patient's bag.
Copy-number calls: caller.py (first pass). Picked cells are a biased sample; exploratory, no statistics claimed."""
from pathlib import Path
import sys
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read  # noqa: E402
from runtime import Cache, model_for  # noqa: E402
FULL = REPO / 'runs/reference_control_full_refit_20261110'; SEEDS = [17, 29, 43]
OUT = REPO / 'analyses/cnv_attribution_20261112'; DRAWS = 20


def main():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.set_per_process_memory_fraction(.20)
    lock = read(FULL / 'model_lock.json'); models = []
    for s in SEEDS:
        m = model_for('transformer', 'd6_do04_lr2e4').cuda().eval()
        m.load_state_dict(torch.load(lock['models'][str(s)]['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True); models.append(m)
    with np.load(lock['normalizer']) as z:
        mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
    T = pd.read_csv(OUT / 'test_sequenced_cells_private.csv', dtype={'slide_id': str})
    T = T[T.relevance.notna()].copy(); T['source_row'] = T.source_row.astype(int)
    cache = Cache(set(T.donor_id)); primary = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}

    @torch.no_grad()
    def score(x):
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device='cuda'); p = []
        for m in models:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p.append(float(m(x, mask)['logits'].float().sigmoid()[0]))
        p = float(np.clip(np.mean(p), 1e-7, 1 - 1e-7)); return np.log(p / (1 - p))

    rng = np.random.default_rng(20261112); loo, grp = [], []
    for d, g in T.groupby('donor_id'):
        b = cache.sample(primary[d]); x = ((b['features'].cuda().float()[None] - mean) / scale).half().float()
        key = pd.DataFrame(dict(slide_id=np.repeat(b['slide_ids'], [len(i) for i in b['slide_row_indices']]),
                                source_row=np.concatenate([i.numpy() for i in b['slide_row_indices']])))
        pos = {(s, r): i for i, (s, r) in enumerate(zip(key.slide_id, key.source_row))}
        g = g.assign(idx=[pos.get((s, r), -1) for s, r in zip(g.slide_id, g.source_row)]); g = g[g.idx >= 0]
        base = score(x); n = x.shape[1]
        drop = lambda idx: score(x[:, np.setdiff1d(np.arange(n), idx)])
        for r in g.itertuples():
            loo.append(dict(donor_id=d, cell_name=r.cell_name, altered=r.altered, altered_fraction=r.altered_fraction, group=r.group,
                            positive_rank=r.positive_rank, display_cluster=r.display_cluster, base=base, delta=base - drop([r.idx])))
        h = g[g.mass90 == True]
        if h.altered.any() and (~h.altered).any():
            for lab, sel in [('altered', h[h.altered]), ('flat', h[~h.altered])]:
                k = len(sel); dd = base - drop(sel.idx.to_numpy())
                rnd = [base - drop(rng.choice(n, k, replace=False)) for _ in range(DRAWS)]
                grp.append(dict(donor_id=d, removed=lab, cells=k, base=base, delta=dd, random_delta_mean=float(np.mean(rnd)), random_delta_sd=float(np.std(rnd))))
        print(d, 'bag', n, 'sequenced model-input', len(g), flush=True)
    L = pd.DataFrame(loo); G = pd.DataFrame(grp)
    L.to_csv(OUT / 'ablation_leave_one_out_private.csv', index=False); G.to_csv(OUT / 'ablation_group_private.csv', index=False)
    print(L.groupby(['group', 'altered']).delta.agg(['size', 'mean', 'median']).round(3))
    if len(G):
        print(G.groupby('removed')[['cells', 'delta', 'random_delta_mean']].agg(['mean', 'median']).round(3))
        w = G.pivot(index='donor_id', columns='removed', values='delta'); print('patients', len(w), 'altered drop > flat drop in', int((w.altered > w.flat).sum()))


if __name__ == '__main__':
    main()
