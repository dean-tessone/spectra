"""Revision copy (2026-11-17): within-cluster embedding test of Supplementary Fig. 5 rerun with 2,000 permutations per cluster
(reviewers: 200 too few for Benjamini-Hochberg). Identical folds, classifier and data; original outputs untouched.
"""
"""Do the cell embeddings carry diagnostic information beyond cluster composition? (exploratory, held-out test cohort)

Model-independent: uses every patient's model-input cells (no attribution or SPECTRA scores), the PCA-50 of the
normalized BLUE v2 embeddings and the Leiden cluster of every cell. Same 757 patients, same classifier and the same
5x5 repeated stratified CV folds as the composition-only reference (logistic regression, C=0.1, balanced).

1. Added value: composition (23 centred log-ratio cluster shares) versus composition + the patient's mean embedding
   (50 PCs, standardized). Null: permute the embedding block across patients (keeps composition and labels), refit.
2. Within cluster: for each cluster, classify patients from the mean embedding of their cells in that cluster only
   (patients with >= MIN_CELLS cells there); abundance cannot enter. Null: permute labels, refit.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
R = REPO / 'runs/figure3_locked6_attribution_20261111'; W = R / 'mass90_leiden'
OUT = REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'  # inputs
DEST = REPO / 'analyses/revision_20261117/within_cluster_permutations'; DEST.mkdir(parents=True, exist_ok=True)
NPERM, MIN_CELLS, SEED = 2000, 3, 17  # revision: 2,000 label permutations per cluster (was 200)


def cv_auc(X, y, n_comp=None):
    """Mean AUROC over the composition reference's 5x5 folds; n_comp = number of leading passthrough (composition) columns."""
    aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y):
        if n_comp is None:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight='balanced', max_iter=4000))
        elif n_comp == X.shape[1]:
            m = LogisticRegression(C=.1, class_weight='balanced', max_iter=4000)
        else:
            ct = ColumnTransformer([('comp', 'passthrough', list(range(n_comp))), ('emb', StandardScaler(), list(range(n_comp, X.shape[1])))])
            m = make_pipeline(ct, LogisticRegression(C=.1, class_weight='balanced', max_iter=4000))
        m.fit(X[tr], y[tr]); aucs.append(roc_auc_score(y[te], m.decision_function(X[te])))
    return float(np.mean(aucs)), aucs


def main():
    cells = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id', 'label'])
    Z = np.load(R / 'all_cells_pca50.npy'); k = np.load(W / 'pool_cluster_all_cells.npy')
    assert len(Z) == len(k) == len(cells)
    donors = cells.drop_duplicates('donor_id').set_index('donor_id').label
    pool = pd.crosstab(cells.donor_id, k, normalize='index')
    ids = pool.index; y = donors.loc[ids].to_numpy()
    comp = np.log(pool.to_numpy() + 1e-3); comp = comp - comp.mean(1, keepdims=True)            # as in the composition reference
    codes = pd.Categorical(cells.donor_id, categories=ids).codes
    emb = np.zeros((len(ids), 50)); np.add.at(emb, codes, Z); emb /= np.bincount(codes)[:, None]
    rng = np.random.default_rng(SEED)

    added = None  # part 1 (composition + mean embedding) dropped from the paper; not rerun

    # 2. within-cluster embeddings
    tab = pd.read_csv(OUT / 'cluster_table.csv', index_col=0)
    num = pd.read_csv(OUT / 'cluster_display_numbers.csv').set_index('cluster').number
    rows = []
    for c in sorted(np.unique(k)):
        m = k == c; cc = codes[m]; n = np.bincount(cc, minlength=len(ids))
        keep = n >= MIN_CELLS
        e = np.zeros((len(ids), 50)); np.add.at(e, cc, Z[m]); e = e[keep] / n[keep, None]; yy = y[keep]
        if yy.sum() < 10 or (1 - yy).sum() < 10:
            rows.append(dict(cluster=int(c), number=int(num[c]), patients=int(keep.sum()), controls=int((1 - yy).sum()))); continue
        auc, _ = cv_auc(e, yy)
        nl = Parallel(n_jobs=16)(delayed(lambda s: cv_auc(e, np.random.default_rng(s).permutation(yy))[0])(s) for s in range(NPERM))
        rows.append(dict(cluster=int(c), number=int(num[c]), patients=int(keep.sum()), controls=int((1 - yy).sum()), within_cluster_embedding_auc=auc,
                         permutation_p=float((1 + (np.array(nl) >= auc).sum()) / (1 + NPERM)), abundance_auc=float(tab.loc[c, 'abundance_auc']),
                         emphasis_log2_cancer=float(tab.loc[c, 'emphasis_log2_cancer'])))
        print(rows[-1], flush=True)
    w = pd.DataFrame(rows).sort_values('number')
    ok = w.permutation_p.notna(); pv = w.loc[ok, 'permutation_p'].to_numpy(); o = np.argsort(pv); q = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; qq = np.empty_like(q); qq[o] = np.minimum(q, 1); w.loc[ok, 'permutation_q'] = qq   # Benjamini-Hochberg
    w.to_csv(DEST / 'within_cluster_embedding_auc_2000perm.csv', index=False)
    json.dump(dict(added_value=added, min_cells=MIN_CELLS, permutations=NPERM, folds='5x5 repeated stratified, seed 17', classifier='logistic regression C=0.1 balanced',
                   note='Exploratory; held-out test cohort; all model-input cells; no attribution or SPECTRA scores used'),
              open(DEST / 'within_cluster_2000perm.json', 'w'), indent=1)
    print(w.round(3).to_string())


if __name__ == '__main__':
    main()
