"""Fig 5e examples consistent with Fig 5c/d: cells from the clusters that lose / gain the most attribution
after adaptation (probe), from test cancer patients. Per cluster, the cell with the largest within-patient rank
change (top 10% before and moved down >=50 percentile points for 'lost'; top 10% after and moved up >=50 for 'gained'); 40x if a cached re-image exists, else 10x."""
from pathlib import Path
import importlib.util, json, sys
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('aa', REPO / 'adaptation/disease_specific/analyze_attribution.py'); AA = importlib.util.module_from_spec(spec); spec.loader.exec_module(AA)
A3 = REPO / 'analyses/figure3_locked6_attribution_20261111'
ok = pd.read_csv(A3 / '40x_extraction_status.csv'); ok = set(ok.cell_key[ok.ok])
rows = []
for task in ['lung', 'breast']:
    sh = pd.read_csv(REPO / f'analyses/specific_adaptation_20261111/attribution/{task}_cluster_share_shift.csv')
    both = sh.groupby('cluster').delta_probe.mean()
    lost, gained = list(both.nsmallest(2).index), list(both.nlargest(4).index)
    t = pd.read_parquet(REPO / f'runs/specific_adaptation_20261111/attribution/{task}_cells_private.parquet')
    t = t[t.label == 1].reset_index(drop=True)
    t['cluster'] = AA.assign(t)
    for h in ['source', 'probe']:
        t[h + '_rk'] = t.groupby('donor_id')[h].rank(ascending=False, pct=True)
    t['shift'] = t.source_rk - t.probe_rk
    t['cell_key'] = [f'{s}_r{int(r):05d}' for s, r in zip(t.slide_id, t.source_row)]
    t['has40'] = t.cell_key.isin(ok)
    def pick(pool, n, asc):
        pool = pool.sort_values(['has40', 'shift'], ascending=[False, asc]).drop_duplicates('donor_id')
        return pool.head(n)
    out_l = pd.concat([pick(t[(t.cluster == k) & (t.source_rk <= .10) & (t['shift'] <= -.5)], 3 if i == 0 else 2, True).assign(kind='lost') for i, k in enumerate(lost)])
    out_g = pd.concat([pick(t[(t.cluster == k) & (t.probe_rk <= .10) & (t['shift'] >= .5)], 2 if i == 0 else 1, False).assign(kind='gained') for i, k in enumerate(gained)])
    rows.append(pd.concat([out_l, out_g]).assign(task=task))
sel = pd.concat(rows, ignore_index=True)
sel.to_csv(REPO / 'analyses/specific_adaptation_20261111/attribution/fig5e_cluster_examples_private.csv', index=False)
print(sel[['task', 'kind', 'cluster', 'donor_id', 'has40', 'source_rk', 'probe_rk']].to_string())
