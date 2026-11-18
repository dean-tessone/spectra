"""Revision step 6.2: CK-gate robustness on per-patient counts of model-input cells (inference-free, 2026-11-17).
Nine predeclared gates CK (TRITC_mean) > 30k/40k/50k AND CD45/CD31 (CY5_mean) < 5k/10k/15k; the 40k/10k gate is primary."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'figures/code')); import sx  # noqa: E402
OUT = REPO / 'analyses/revision_20261117/ck_gate_grid'; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261117); B = 4000
def auc(pos, neg):
    allv = np.concatenate([pos, neg]); rk = pd.Series(allv).rank().to_numpy()
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
c = pd.read_parquet(REPO / 'runs/figure3_locked6_attribution_20261111/all_cells.parquet', columns=['donor_id', 'label', 'TRITC_mean', 'CY5_mean', 'CK_gate'])
p = sx.primary_predictions()[['donor_id', 'label']]
rows = []
for ck in (30000, 40000, 50000):
    for cy in (5000, 10000, 15000):
        g = ((c.TRITC_mean > ck) & (c.CY5_mean < cy)).groupby(c.donor_id).sum().reindex(p.donor_id, fill_value=0).to_numpy()
        y = p.label.to_numpy(); P, N = g[y == 1], g[y == 0]
        reps = [auc(P[rng.integers(0, len(P), len(P))], N[rng.integers(0, len(N), len(N))]) for _ in range(B)]
        rows.append(dict(CK_gt=ck, CD45CD31_lt=cy, primary=(ck, cy) == (40000, 10000), AUROC=auc(P, N), ci_low=np.percentile(reps, 2.5), ci_high=np.percentile(reps, 97.5),
                         median_cancer=np.median(P), median_control=np.median(N), controls_with_any=int((N > 0).sum())))
# consistency with the stored CK_gate column
chk = c.CK_gate.groupby(c.donor_id).sum().reindex(p.donor_id, fill_value=0).to_numpy()
own = ((c.TRITC_mean > 40000) & (c.CY5_mean < 10000)).groupby(c.donor_id).sum().reindex(p.donor_id, fill_value=0).to_numpy()
assert (chk == own).all()
r = pd.DataFrame(rows); r.to_csv(OUT / 'ck_gate_grid_per_patient.csv', index=False); print(r.round(3).to_string())
