"""Revision step 8.8: patient-bootstrap CIs for the Fig. 6c,d attribution-share shifts (probe head minus source head), inference-free.
Shift per cluster = mean over held-out test task patients of (probe share - source share), in percentage points; equal-weight
mean of the cancer-group and control-group means, as in Fig. 6c,d."""
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
A = REPO / 'analyses/specific_adaptation_20261111/attribution'
OUT = REPO / 'analyses/revision_20261117/adaptation'; OUT.mkdir(parents=True, exist_ok=True)
disp = pd.read_csv(REPO / 'analyses/figure3_locked6_attribution_20261111/leiden/cluster_display_numbers.csv').set_index('cluster').number
rng = np.random.default_rng(20261117); B = 4000; rows = []
for task in ('lung', 'breast'):
    s = pd.read_csv(A / f'{task}_cluster_share_private.csv', dtype={'donor_id': str})
    s['d'] = 100 * (s.probe - s.source)
    w = s.pivot_table(index=['donor_id', 'label'], columns='cluster', values='d').reset_index()  # no fill: matches Fig. 6 group means
    cl = [c for c in w.columns if c not in ('donor_id', 'label')]
    ca, co = w[w.label == 1][cl].to_numpy(), w[w.label == 0][cl].to_numpy()
    est = (np.nanmean(ca, 0) + np.nanmean(co, 0)) / 2
    reps = np.array([(np.nanmean(ca[rng.integers(0, len(ca), len(ca))], 0) + np.nanmean(co[rng.integers(0, len(co), len(co))], 0)) / 2 for _ in range(B)])
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    for i, c in enumerate(cl):
        rows.append(dict(task=task, cluster=int(disp[c]), shift_pp=est[i], ci_low=lo[i], ci_high=hi[i], cancers=len(ca), controls=len(co),
                         cancer_mean_pp=np.nanmean(ca, 0)[i], control_mean_pp=np.nanmean(co, 0)[i]))
r = pd.DataFrame(rows).sort_values(['task', 'shift_pp']); r.to_csv(OUT / 'attribution_shift_ci.csv', index=False)
print(r[r.cluster.isin([10, 5, 8, 21, 18, 19])].round(2).to_string())
