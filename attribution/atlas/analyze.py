"""Patient-level atlas analysis + deterministic exemplar selection (CPU, descriptive).

Meta-states: Ward linkage of the 35 HDBSCAN cluster centroids (PCA50), cut at 8.
Pool fractions: every retained model-input cell assigned (kNN) to an atlas cluster.
Composition-only reference: logistic regression on centred-log-ratio pool fractions,
5x repeated 5-fold CV *within the test cohort* (exploratory; not a SPECTRA comparison
under identical conditions, since SPECTRA was trained on the training cohort).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster

REPO = Path(__file__).resolve().parents[2]
R = REPO / 'runs/figure3_locked6_attribution_20261111'
OUT = REPO / 'analyses/figure3_locked6_attribution_20261111'; OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ['Large Nuclei', 'imCTCs', 'CTCs', 'pcCTCs', 'CECs', 'Megakaryocytes', 'Fibroblasts', 'Lymph', 'Gran', 'Mono']


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


def main():
    atlas = pd.read_parquet(R / 'atlas_top50.parquet')
    Z = np.load(R / 'all_cells_pca50.npy', mmap_mode='r')
    cells = pd.read_parquet(R / 'all_cells.parquet')
    cells['cluster'] = pd.read_parquet(R / 'pool_cluster_assignment.parquet').pool_cluster.to_numpy()
    k = int(atlas.cluster.max() + 1)
    cent = np.stack([np.asarray(Z[atlas.index[atlas.cluster == c]]).mean(0) for c in range(k)])
    meta = fcluster(linkage(cent, 'ward'), 8, 'maxclust') - 1
    # order meta-states by attributed cell count
    atlas['meta_raw'] = np.where(atlas.cluster >= 0, meta[atlas.cluster.clip(0)], -1)
    order = atlas[atlas.meta_raw >= 0].meta_raw.value_counts().index.tolist(); remap = {m: i for i, m in enumerate(order)}
    meta = np.array([remap[m] for m in meta])
    atlas['meta'] = np.where(atlas.cluster >= 0, meta[atlas.cluster.clip(0)], -1)
    cells['meta'] = meta[cells.cluster]
    atlas.to_parquet(OUT / 'atlas_cells_private.parquet')
    pd.DataFrame(dict(cluster=range(k), meta=meta)).to_csv(OUT / 'cluster_to_meta.csv', index=False)

    def profile(df, key):
        g = df.groupby(key); cls = pd.crosstab(df[key], df.bluev2_class.map(dict(enumerate(CLASSES))), normalize='index')
        prof = pd.DataFrame(dict(cells=g.size(), donors=g.donor_id.nunique(), case_fraction=g.label.mean(),
                                 CK=g.TRITC_mean.median(), VIM=g.FITC_mean.median(), CD45_CD31=g.CY5_mean.median(), DAPI=g.DAPI_mean.median(),
                                 area=g.area.median(), eccentricity=g.eccentricity.median(), CK_gate_fraction=g.CK_gate.mean()))
        return prof.join(cls)
    profile(atlas[atlas.cluster >= 0], 'meta').to_csv(OUT / 'meta_state_profile_attributed.csv')
    profile(atlas[atlas.cluster >= 0], 'cluster').to_csv(OUT / 'cluster_profile_attributed.csv')
    profile(cells, 'meta').to_csv(OUT / 'meta_state_profile_pool.csv')

    donors = cells[['donor_id', 'label', 'probability', 'cancer_types']].drop_duplicates('donor_id').set_index('donor_id')
    rows = []
    for level in ['cluster', 'meta']:
        frac = pd.crosstab(cells.donor_id, cells[level], normalize='index')
        top = cells[cells.top50]; tf = pd.crosstab(top.donor_id, top[level], normalize='index').reindex(index=frac.index, columns=frac.columns).fillna(0)
        frac.to_csv(OUT / f'donor_pool_fraction_{level}_private.csv'); tf.to_csv(OUT / f'donor_top50_fraction_{level}_private.csv')
        lab = donors.loc[frac.index, 'label']; sc = donors.loc[frac.index, 'probability']
        res = []
        for c in frac.columns:
            x1, x0 = frac.loc[lab == 1, c], frac.loc[lab == 0, c]; u = mannwhitneyu(x1, x0)
            t1, t0 = tf.loc[lab == 1, c], tf.loc[lab == 0, c]
            res.append(dict(level=level, state=int(c), pool_fraction_cases=x1.mean(), pool_fraction_controls=x0.mean(), pool_auc_case_vs_control=u.statistic / len(x1) / len(x0),
                            pool_p=u.pvalue, spearman_pool_vs_probability=spearmanr(frac[c], sc)[0],
                            top50_fraction_cases=t1.mean(), top50_fraction_controls=t0.mean(),
                            attribution_enrichment_cases=np.log2((t1.mean() + 1e-6) / (x1.mean() + 1e-6)),
                            donors_with_top50_cell_cases=float((t1 > 0).mean()), donors_with_top50_cell_controls=float((t0 > 0).mean())))
        res = pd.DataFrame(res); res['pool_q_BH'] = bh(res.pool_p); rows.append(res)
    stats = pd.concat(rows); stats.to_csv(OUT / 'state_patient_level_statistics.csv', index=False)

    # composition-only reference model (exploratory, within-test CV)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_auc_score
    frac = pd.crosstab(cells.donor_id, cells.cluster, normalize='index')
    X = np.log(frac.to_numpy() + 1e-3); X = X - X.mean(1, keepdims=True); y = donors.loc[frac.index, 'label'].to_numpy()
    aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=17).split(X, y):
        m = LogisticRegression(C=.1, class_weight='balanced', max_iter=2000).fit(X[tr], y[tr]); aucs.append(roc_auc_score(y[te], m.decision_function(X[te])))
    spectra_auc = roc_auc_score(y, donors.loc[frac.index, 'probability'])
    json.dump(dict(composition_clr_logistic_within_test_cv_auc_mean=float(np.mean(aucs)), sd=float(np.std(aucs)), folds=len(aucs),
                   spectra_test_auc=float(spectra_auc), note='Exploratory within-test CV; C fixed at 0.1, not tuned'), open(OUT / 'composition_reference.json', 'w'), indent=1)

    # deterministic exemplars: top-50 cells from correctly called cases, nearest state PCA median, one per donor
    ex = []
    called = set(donors.index[(donors.label == 1)]) & set(cells.donor_id[cells.called_at_90spec])
    ck = atlas[(atlas.cluster >= 0)].groupby('cluster').CK_gate.mean().idxmax()
    groups = [(f'meta{m}', atlas[(atlas.meta == m)]) for m in range(8)] + [(f'cluster{ck}_CKgate', atlas[atlas.cluster == ck])]
    for name, g in groups:
        g = g[g.donor_id.isin(called)]
        x = np.asarray(Z[g.index]); med = np.median(x, 0); d = ((x - med) ** 2).sum(1)
        g = g.assign(dist=d).sort_values(['dist', 'slide_id', 'source_row'])
        g = g.drop_duplicates('donor_id').head(8)
        ex.append(g.assign(state=name, exemplar_rank=np.arange(1, len(g) + 1)))
    ex = pd.concat(ex); ex['cell_key'] = [f'{s}_r{int(r):05d}' for s, r in zip(ex.slide_id, ex.source_row)]
    ex.to_csv(OUT / 'exemplar_cells_private.csv', index=False)
    print(stats[stats.level == 'meta'].round(3).to_string()); print(open(OUT / 'composition_reference.json').read())


if __name__ == '__main__':
    main()
