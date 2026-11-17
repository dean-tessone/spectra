"""Cluster removal and pairwise interaction test with the locked SPECTRA ensemble (inference only, post hoc, descriptive).

Question (group meeting 2026-11-12): several Figure 3 clusters are less common in cancers than in controls yet receive more
than their share of attribution. Is their contribution context dependent, i.e. does it interact with other cell populations?

For each of the 757 primary test patients: base logit (mean over seeds 17/29/43, BF16 replay as in removal.py), the logit after
removing every cell of one Leiden cluster (23 conditions), after removing a random matched number of other cells (null), and after
removing pairs of clusters from a fixed set PAIR_SET (45 pairs). Interaction for a pair (a, b) = D(a+b) - D(a) - D(b), where D is
the change in ensemble-mean logit. Logits, not probabilities, because ensemble probabilities saturate.
"""
from pathlib import Path
import itertools, json, sys
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read, utc  # noqa: E402
from runtime import Cache, model_for  # noqa: E402

RUN = REPO / 'runs/figure3_locked6_attribution_20261111'
FULL = REPO / 'runs/reference_control_full_refit_20261110'
OUT = REPO / 'runs/population_analyses'; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [17, 29, 43]
# internal Leiden ids; display numbers 10, 18, 17, 4, 12, 20, 21, 5, 1, 23
PAIR_SET = [2, 12, 13, 10, 21, 8, 5, 0, 9, 3]


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.20)
    lock = read(FULL / 'model_lock.json'); models = []
    for s in SEEDS:
        m = model_for('transformer', 'd6_do04_lr2e4').cuda().eval()
        m.load_state_dict(torch.load(lock['models'][str(s)]['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True)
        models.append(m)
    with np.load(lock['normalizer']) as z:
        mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
    donors = pd.read_csv(RUN / 'donors.csv', dtype={'donor_id': str}, float_precision='round_trip')
    cells = pd.read_parquet(RUN / 'all_cells.parquet', columns=['donor_id', 'slide_id', 'source_row'])
    cells['k'] = np.load(RUN / 'mass90_leiden/pool_cluster_all_cells.npy')
    cache = Cache(set(donors.donor_id))
    primary = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}

    @torch.no_grad()
    def score(x, keep=None):
        if keep is not None:
            x = x[:, torch.as_tensor(keep, device='cuda')]
        lo = []
        for m in models:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lo.append(float(m(x, mask)['logits'].float()[0]))
        lo = np.array(lo); return float(lo.mean()), float((1 / (1 + np.exp(-lo))).mean())

    rows = []
    for i, d in enumerate(donors.itertuples(), 1):
        b = cache.sample(primary[d.donor_id])
        x = ((b['features'].cuda().float()[None] - mean) / scale).half().float()
        key = pd.DataFrame(dict(slide_id=np.repeat(b['slide_ids'], [len(j) for j in b['slide_row_indices']]),
                                source_row=np.concatenate([j.numpy() for j in b['slide_row_indices']])))
        c = key.merge(cells[cells.donor_id == d.donor_id], on=['slide_id', 'source_row'], how='left', validate='one_to_one')
        assert c.k.notna().all() and len(c) == x.shape[1]
        k = c.k.to_numpy().astype(int); n = len(k)
        base, bp = score(x); assert abs(bp - d.probability) <= 1e-6, (d.donor_id, bp, d.probability)
        row = dict(donor_id=d.donor_id, label=int(d.label), cancer_types=d.cancer_types, n_cells=n, base_logit=base)
        rng = np.random.default_rng([20261112, i])
        for a in range(23):
            sel = k == a; row[f'n_{a}'] = int(sel.sum())
            if 0 < sel.sum() < n:
                row[f'rm_{a}'] = score(x, np.flatnonzero(~sel))[0]
                drop = rng.choice(np.flatnonzero(~sel), size=min(int(sel.sum()), int((~sel).sum()) - 1), replace=False)
                row[f'rnd_{a}'] = score(x, np.setdiff1d(np.arange(n), drop))[0]
            else:
                row[f'rm_{a}'] = row[f'rnd_{a}'] = base if sel.sum() == 0 else np.nan
        for a, b_ in itertools.combinations(PAIR_SET, 2):
            sel = (k == a) | (k == b_)
            if (k == a).sum() and (k == b_).sum() and sel.sum() < n:
                row[f'rm_{a}_{b_}'] = score(x, np.flatnonzero(~sel))[0]
        rows.append(row)
        if i % 50 == 0:
            print('patients', i, utc(), flush=True)
            pd.DataFrame(rows).to_parquet(OUT / 'cluster_removal_private.partial.parquet')
    pd.DataFrame(rows).to_parquet(OUT / 'cluster_removal_private.parquet')
    json.dump(dict(completed_utc=utc(), patients=len(rows), seeds=SEEDS, pair_set=PAIR_SET, scale='ensemble-mean logit',
                   note='inference only; post hoc descriptive; locked model, normalizer and bags unchanged'),
              open(OUT / 'cluster_removal_receipt.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
