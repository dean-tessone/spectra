"""Summarise the leukocyte negative control (step 1.8): bootstrap AUROC, paired difference vs real bags, per control source."""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'figures/code')); import sx  # noqa: E402
OUT = REPO / 'analyses/revision_20261117/leukocyte_control'; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261117); B = 4000
def auc(pos, neg):
    allv = np.concatenate([pos, neg]); rk = pd.Series(allv).rank().to_numpy()
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
p = pd.read_csv(REPO / 'runs/revision_20261117/leukocyte_control/leukocyte_bag_predictions_private.csv', dtype={'donor_id': str})
t = sx.primary_predictions()[['donor_id', 'probability', 'study']].rename(columns={'probability': 'real'})
m = p.merge(t, on='donor_id', validate='one_to_one'); y = m.label.to_numpy(); P, N = np.where(y == 1)[0], np.where(y == 0)[0]
L, R = m.probability.to_numpy(), m.real.to_numpy()
PI, NI = rng.integers(0, len(P), (B, len(P))), rng.integers(0, len(N), (B, len(N)))
bl = np.array([auc(L[P][PI[b]], L[N][NI[b]]) for b in range(B)]); br = np.array([auc(R[P][PI[b]], R[N][NI[b]]) for b in range(B)])
res = dict(patients=len(m), cancers=len(P), controls=len(N),
           leukocyte_AUROC=auc(L[P], L[N]), leukocyte_ci=np.percentile(bl, [2.5, 97.5]).tolist(),
           real_AUROC=auc(R[P], R[N]), real_ci=np.percentile(br, [2.5, 97.5]).tolist(),
           diff_real_minus_leukocyte=auc(R[P], R[N]) - auc(L[P], L[N]), diff_ci=np.percentile(br - bl, [2.5, 97.5]).tolist(),
           median_bag_cells=dict(cancers=float(m.leukocyte_bag[y == 1].median()), controls=float(m.leukocyte_bag[y == 0].median())),
           median_probability_leukocyte=dict(cancers=float(np.median(L[P])), controls=float(np.median(L[N]))),
           spearman_leukocyte_vs_real=float(m[['probability', 'real']].corr('spearman').iloc[0, 1]),
           per_control_source={s: auc(L[P], L[N][m.study.to_numpy()[N] == s]) for s in sorted(m.study[N].unique())})
json.dump(res, open(OUT / 'summary.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
