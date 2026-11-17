"""Summaries of cluster_removal.py (post hoc, descriptive). All changes are in ensemble-mean logit."""
from pathlib import Path
import itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / 'runs/population_analyses'; OUT = REPO / 'analyses/population_analyses'
L = REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
NUM = pd.read_csv(L / 'cluster_display_numbers.csv', index_col=0).number; T = pd.read_csv(L / 'cluster_table.csv', index_col=0)
PAIR_SET = [2, 12, 13, 10, 21, 8, 5, 0, 9, 3]


def main():
    r = pd.read_parquet(RUN / 'cluster_removal_private.parquet'); y = r.label.to_numpy(); base_auc = roc_auc_score(y, r.base_logit)
    rows = []
    for k in range(23):
        d = r[f'rm_{k}'] - r.base_logit; dr = r[f'rnd_{k}'] - r.base_logit; has = r[f'n_{k}'] > 0
        frac = r[f'n_{k}'] / r.n_cells
        rows.append(dict(cluster_display=NUM[k], internal=k, abundance_auc=T.abundance_auc[k], emphasis_log2=T.emphasis_log2_cancer[k],
                         mean_cell_fraction_cancer=frac[y == 1].mean(), mean_cell_fraction_control=frac[y == 0].mean(),
                         change_cancer=d[y == 1].mean(), change_control=d[y == 0].mean(), random_change_cancer=dr[y == 1].mean(), random_change_control=dr[y == 0].mean(),
                         change_per_100_cells_cancer=(d / r[f'n_{k}'] * 100)[has & (y == 1)].median(), change_per_100_cells_control=(d / r[f'n_{k}'] * 100)[has & (y == 0)].median(),
                         auroc_without=roc_auc_score(y, r[f'rm_{k}'].fillna(r.base_logit)), auroc_without_random=roc_auc_score(y, r[f'rnd_{k}'].fillna(r.base_logit)),
                         per_cell_effect_auc=roc_auc_score(y[has], -(d / r[f'n_{k}'])[has])))
    s = pd.DataFrame(rows).sort_values('cluster_display'); s['auroc_drop'] = base_auc - s.auroc_without
    s.to_csv(OUT / 'cluster_removal_summary.csv', index=False)
    # pairwise interaction
    prow = []
    for a, b in itertools.combinations(PAIR_SET, 2):
        col = f'rm_{a}_{b}'
        if col not in r:
            continue
        ok = r[col].notna()
        da, db, dab = (r[f'rm_{a}'] - r.base_logit)[ok], (r[f'rm_{b}'] - r.base_logit)[ok], (r[col] - r.base_logit)[ok]
        I = dab - da - db; yy = y[ok.to_numpy()]
        prow.append(dict(a=NUM[a], b=NUM[b], patients=int(ok.sum()), additive_cancer=(da + db)[yy == 1].mean(), joint_cancer=dab[yy == 1].mean(), interaction_cancer=I[yy == 1].mean(),
                         additive_control=(da + db)[yy == 0].mean(), joint_control=dab[yy == 0].mean(), interaction_control=I[yy == 0].mean(),
                         interaction_rel=float(np.median(np.abs(I) / (np.abs(da) + np.abs(db) + 1e-6))),
                         interaction_auc=roc_auc_score(yy, I)))
    p = pd.DataFrame(prow); p.to_csv(OUT / 'cluster_pair_interaction.csv', index=False)
    print('base AUROC', round(base_auc, 4)); print(s.round(3).to_string()); print(p.round(3).sort_values('interaction_rel', ascending=False).to_string())


if __name__ == '__main__':
    main()
