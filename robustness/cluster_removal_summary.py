"""Summarise cluster removal (step 2.3) with the same reporting as step 3."""
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'analyses/revision_20261117/ablation'; OUT.mkdir(parents=True, exist_ok=True)
ops = pd.read_csv(REPO / 'analyses/reference_control_test_20261110_v1a/operating_points.csv')
ops = ops[(ops.endpoint == 'primary') & (ops.cancer_type == 'Overall') & (ops.comparison == 'Cancer vs primary reference')].set_index('target_specificity').threshold
r = pd.read_csv(REPO / 'runs/revision_20261117/cluster_removal/cluster_removal_per_patient_private.csv')
y = r.label.to_numpy(); P, N = np.where(y == 1)[0], np.where(y == 0)[0]
rng = np.random.default_rng(20261117); B = 4000; PI, NI = rng.integers(0, len(P), (B, len(P))), rng.integers(0, len(N), (B, len(N)))
def auc(pos, neg):
    allv = np.concatenate([pos, neg]); rk = pd.Series(allv).rank().to_numpy()
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def boot(v): return np.array([auc(v[P][PI[b]], v[N][NI[b]]) for b in range(B)])
def refit(v, k): t = np.sort(v[N])[k - 1]; return float((v[P] > t).mean())
rows = []
for c in ('cluster10', 'cluster20', 'cluster21'):
    a, b = r[f'{c}_removed'].to_numpy(), r[f'{c}_random'].to_numpy(); ra, rb = boot(a), boot(b); d = ra - rb
    for arm, v, rr in (('removed', a, ra), ('matched random', b, rb)):
        rows.append(dict(condition=c.replace('cluster', 'Cluster '), arm=arm, AUROC=auc(v[P], v[N]), ci_low=np.percentile(rr, 2.5), ci_high=np.percentile(rr, 97.5),
                         sens90_locked=float((v[P] >= ops[.9]).mean()), spec90_locked=f'{int((v[N] < ops[.9]).sum())}/44', sens90_recomputed=refit(v, 40),
                         median_removed_cancers=float(np.median(r.loc[y == 1, f'{c}_n'])), median_removed_controls=float(np.median(r.loc[y == 0, f'{c}_n']))))
    rows[-2].update(diff_vs_random=rows[-2]['AUROC'] - rows[-1]['AUROC'], diff_ci_low=np.percentile(d, 2.5), diff_ci_high=np.percentile(d, 97.5),
                    p=float(2 * min((d <= 0).mean(), (d >= 0).mean())))
o = pd.DataFrame(rows); o.to_csv(OUT / 'cluster_removal_reporting.csv', index=False)
pd.set_option('display.width', 250); print(o.round(3).to_string())
