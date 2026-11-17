"""Fig 5e: 40x only, 5 'less influence' + 5 'more influence' cells per task.
Only the six clusters drawn in 5c/5d are eligible (40x supply decides which of them can contribute), taken in order of their attribution-share change (5c/5d; negative for less, positive for more);
within each, test-cancer cells with a cached 40x re-image that moved >=30 percentile points in the matching
direction, largest move first, at most two per patient; continue to the next cluster until 5 cells. Candidates (up to
12 per row) are written for visual review of image quality."""
from pathlib import Path
import importlib.util
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('aa', REPO / 'adaptation/disease_specific/analyze_attribution.py'); AA = importlib.util.module_from_spec(spec); spec.loader.exec_module(AA)
A3 = REPO / 'analyses/figure3_locked6_attribution_20261111'
ok = pd.read_csv(A3 / '40x_extraction_status.csv'); ok = set(ok.cell_key[ok.ok])
rows = []
for task in ['lung', 'breast']:
    sh = pd.read_csv(REPO / f'analyses/specific_adaptation_20261111/attribution/{task}_cluster_share_shift.csv')
    both = sh.pivot_table(index='cluster', columns='label', values='delta_probe').fillna(0).mean(axis=1)
    both = both[both.abs().sort_values(ascending=False).index[:6]]   # exactly the six clusters drawn in Fig 5c/5d
    t = pd.read_parquet(REPO / f'runs/specific_adaptation_20261111/attribution/{task}_cells_private.parquet'); t = t[t.label == 1].reset_index(drop=True)
    t['cell_key'] = [f'{s}_r{int(r):05d}' for s, r in zip(t.slide_id, t.source_row)]
    t = t[t.cell_key.isin(ok)].reset_index(drop=True)
    t['cluster'] = AA.assign(t)
    full = pd.read_parquet(REPO / f'runs/specific_adaptation_20261111/attribution/{task}_cells_private.parquet', columns=['donor_id', 'slide_id', 'source_row', 'source', 'probe'])
    for h in ['source', 'probe']:
        full[h + '_rk'] = full.groupby('donor_id')[h].rank(ascending=False, pct=True)
    t = t.merge(full[['slide_id', 'source_row', 'source_rk', 'probe_rk']], on=['slide_id', 'source_row'])
    t['shift'] = t.source_rk - t.probe_rk
    for kind, clusters, cond, asc in [('lost', both[both < 0].sort_values().index, lambda d: d['shift'] <= -.3, True),
                                      ('gained', both[both > 0].sort_values(ascending=False).index, lambda d: (d['shift'] > 0) & (d['probe_rk'] <= .25), False)]:
        picked = []; seen_n = {}
        for k in clusters:
            g = t[(t.cluster == k) & cond(t)].sort_values('shift', ascending=asc)
            for r in g.itertuples():
                if seen_n.get(r.donor_id, 0) >= 2: continue
                picked.append(dict(task=task, kind=kind, cluster=int(k), cluster_delta=float(both[k]), **{c: getattr(r, c) for c in ['donor_id', 'slide_id', 'source_row', 'cell_key', 'source_rk', 'probe_rk', 'shift']}))
                seen_n[r.donor_id] = seen_n.get(r.donor_id, 0) + 1
                if sum(1 for p in picked if p['cluster'] == k) >= 6: break
            if len(picked) >= 12: break
        rows += picked
c = pd.DataFrame(rows)
c.to_csv(REPO / 'analyses/specific_adaptation_20261111/attribution/fig5e_40x_candidates_private.csv', index=False)
print(c.groupby(['task', 'kind']).size()); print(c.groupby(['task', 'kind']).cluster.apply(list))
