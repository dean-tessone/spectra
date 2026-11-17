"""Overall effect of cluster-resolved embeddings on patient classification (exploratory, held-out test cohort).

Model-independent (all model-input cells; no attribution or SPECTRA scores). Same 757 patients and the same 5x5 repeated stratified
outer folds as the composition-only reference (logistic regression, C=0.1, class-balanced).

  composition:            23 centred log-ratio cluster shares                                   (= composition reference)
  composition+embedding:  the 23 shares plus 23 cluster embedding scores. Each cluster score comes from a logistic regression on the
                          mean PCA-50 embedding of the patient's cells in that cluster (patients with >= 3 cells there), fitted on
                          the outer-training patients only; training patients get inner 5-fold out-of-fold scores, so the second
                          stage never sees a score fitted on the same patient. Patients without the cluster get the training median.
  embedding only:         the 23 cluster embedding scores alone.

Reported: mean outer-fold AUROC (as for the composition reference), folds improved, and the change in pooled AUROC of the
repeat-averaged out-of-fold scores with a stratified patient bootstrap (B=2000).
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
R = REPO / 'runs/figure3_locked6_attribution_20261111'; W = R / 'mass90_leiden'
OUT = REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
MIN_CELLS, SEED, B = 3, 17, 2000


def lr(scale=True):
    m = LogisticRegression(C=.1, class_weight='balanced', max_iter=4000)
    return make_pipeline(StandardScaler(), m) if scale else m


def cluster_scores(E, has, y, tr, te):
    """Cluster embedding scores: inner-OOF for training patients, fitted-on-all-training for test patients."""
    s_tr, s_te = np.zeros((len(tr), len(E))), np.zeros((len(te), len(E)))
    for c, (e, h) in enumerate(zip(E, has)):
        a, b = tr[h[tr]], te[h[te]]
        oof = np.full(len(tr), np.nan); tst = np.full(len(te), np.nan)
        if y[a].sum() >= 5 and (1 - y[a]).sum() >= 5:
            pos = {p: i for i, p in enumerate(tr)}; post = {p: i for i, p in enumerate(te)}
            for itr, ite in StratifiedKFold(5, shuffle=True, random_state=SEED).split(e[a], y[a]):
                m = lr().fit(e[a[itr]], y[a[itr]]); oof[[pos[p] for p in a[ite]]] = m.decision_function(e[a[ite]])
            m = lr().fit(e[a], y[a])
            if len(b):
                tst[[post[p] for p in b]] = m.decision_function(e[b])
        med = np.nanmedian(oof) if np.isfinite(oof).any() else 0.
        s_tr[:, c] = np.where(np.isnan(oof), med, oof); s_te[:, c] = np.where(np.isnan(tst), med, tst)
    return s_tr, s_te


def one_fold(k, tr, te, comp, E, has, y):
    s_tr, s_te = cluster_scores(E, has, y, tr, te)
    out = {}
    for name, Xtr, Xte, scale in [('composition', comp[tr], comp[te], False),
                                  ('composition_plus_embedding', np.hstack([comp[tr], s_tr]), np.hstack([comp[te], s_te]), True),
                                  ('embedding_only', s_tr, s_te, True)]:
        m = lr(scale).fit(Xtr, y[tr]); out[name] = m.decision_function(Xte)
    return k, te, out


def main():
    cells = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id', 'label'])
    Z = np.load(R / 'all_cells_pca50.npy'); kcl = np.load(W / 'pool_cluster_all_cells.npy')
    donors = cells.drop_duplicates('donor_id').set_index('donor_id').label
    pool = pd.crosstab(cells.donor_id, kcl, normalize='index'); ids = pool.index; y = donors.loc[ids].to_numpy()
    comp = np.log(pool.to_numpy() + 1e-3); comp = comp - comp.mean(1, keepdims=True)
    codes = pd.Categorical(cells.donor_id, categories=ids).codes
    E, has = [], []
    for c in sorted(np.unique(kcl)):
        m = kcl == c; n = np.bincount(codes[m], minlength=len(ids))
        e = np.zeros((len(ids), 50)); np.add.at(e, codes[m], Z[m]); e[n > 0] /= n[n > 0, None]
        E.append(e); has.append(n >= MIN_CELLS)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(comp, y))
    res = Parallel(n_jobs=25)(delayed(one_fold)(k, tr, te, comp, E, has, y) for k, (tr, te) in enumerate(splits))
    names = ['composition', 'composition_plus_embedding', 'embedding_only']
    fold = {n: [] for n in names}; oof = {n: np.zeros((5, len(ids))) for n in names}
    for k, te, out in sorted(res, key=lambda r: r[0]):
        for n in names:
            fold[n].append(roc_auc_score(y[te], out[n])); oof[n][k // 5, te] = out[n]
    # rank-normalize each repeat's pooled OOF scores before averaging (fold models are on different scales)
    from scipy.stats import rankdata
    avg = {n: np.mean([rankdata(oof[n][r]) for r in range(5)], 0) for n in names}
    auc = {n: roc_auc_score(y, avg[n]) for n in names}
    rng = np.random.default_rng(SEED); pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0); d = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[i], avg['composition_plus_embedding'][i]) - roc_auc_score(y[i], avg['composition'][i]))
    res = dict(mean_fold_auc={n: float(np.mean(fold[n])) for n in names}, sd_fold_auc={n: float(np.std(fold[n])) for n in names},
               folds_improved=int(sum(a > b for a, b in zip(fold['composition_plus_embedding'], fold['composition']))), folds=len(splits),
               pooled_oof_auc=auc, pooled_gain=float(auc['composition_plus_embedding'] - auc['composition']),
               pooled_gain_ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))], bootstrap=B, min_cells=MIN_CELLS,
               note='Exploratory; within-test nested CV; all model-input cells; no attribution or SPECTRA scores')
    json.dump(res, open(OUT / 'embedding_stacked.json', 'w'), indent=1); print(json.dumps(res, indent=1))
    pd.DataFrame({n: fold[n] for n in names}).to_csv(OUT / 'embedding_stacked_folds.csv', index=False)


if __name__ == '__main__':
    main()
