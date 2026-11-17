"""Patient-level statistics for the Leiden atlas of mass-90 attributed cells (descriptive, post hoc)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[2]
R = REPO / 'runs/figure3_locked6_attribution_20261111'; W = R / 'mass90_leiden'
OUT = REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'; OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ['Large Nuclei', 'imCTCs', 'CTCs', 'pcCTCs', 'CECs', 'Megakaryocytes', 'Fibroblasts', 'Lymph', 'Gran', 'Mono']


def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.minimum(q, 1); return out


def main():
    cells = pd.read_parquet(R / 'all_cells.parquet')
    cells['cluster'] = np.load(W / 'pool_cluster_all_cells.npy')
    idx = np.load(W / 'mass90_index.npy'); prim = np.load(W / 'cluster_primary.npy'); xy = np.load(W / 'umap2d.npy')
    atlas = cells.iloc[idx].copy(); atlas['cluster'] = prim; atlas['umap_1'], atlas['umap_2'] = xy[:, 0], xy[:, 1]
    atlas['atlas_row'] = np.arange(len(atlas))
    atlas['cell_key'] = [f'{s}_r{int(r):05d}' for s, r in zip(atlas.slide_id, atlas.source_row)]
    atlas.to_parquet(OUT / 'atlas_mass90_private.parquet')
    g = atlas.groupby('cluster'); cls = pd.crosstab(atlas.cluster, atlas.bluev2_class.map(dict(enumerate(CLASSES))), normalize='index')
    prof = pd.DataFrame(dict(cells=g.size(), donors=g.donor_id.nunique(), cancer_cell_fraction=g.label.mean(),
                             CK=g.TRITC_mean.median(), CD45_CD31=g.CY5_mean.median(), VIM=g.FITC_mean.median(), DAPI=g.DAPI_mean.median(),
                             area=g.area.median(), eccentricity=g.eccentricity.median(), CK_gate_fraction=g.CK_gate.mean(),
                             relevance_share=g.relevance.sum() / atlas.relevance.sum())).join(cls)
    donors = cells[['donor_id', 'label', 'probability']].drop_duplicates('donor_id').set_index('donor_id')
    pool = pd.crosstab(cells.donor_id, cells.cluster, normalize='index')
    att = pd.crosstab(atlas.donor_id, atlas.cluster, normalize='index').reindex(index=pool.index, columns=pool.columns).fillna(0)
    lab = donors.loc[pool.index, 'label']; rows = []
    for c in pool.columns:
        x1, x0 = pool.loc[lab == 1, c], pool.loc[lab == 0, c]; u = mannwhitneyu(x1, x0)
        a1, a0 = att.loc[lab == 1, c], att.loc[lab == 0, c]
        rows.append(dict(cluster=int(c), pool_share_cancer=x1.mean(), pool_share_control=x0.mean(), abundance_auc=u.statistic / len(x1) / len(x0), abundance_p=u.pvalue,
                         attributed_share_cancer=a1.mean(), attributed_share_control=a0.mean(),
                         emphasis_log2_cancer=np.log2((a1.mean() + 1e-6) / (x1.mean() + 1e-6)), emphasis_log2_control=np.log2((a0.mean() + 1e-6) / (x0.mean() + 1e-6)),
                         patients_with_attributed_cancer=float((a1 > 0).mean()), patients_with_attributed_control=float((a0 > 0).mean())))
    st = pd.DataFrame(rows).set_index('cluster'); st['abundance_q'] = bh(st.abundance_p)
    prof.join(st).to_csv(OUT / 'cluster_table.csv')
    pool.to_csv(OUT / 'donor_pool_fraction_private.csv'); att.to_csv(OUT / 'donor_attributed_fraction_private.csv')
    # composition-only reference: within-test 5x5 CV, fixed C
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_auc_score
    X = np.log(pool.to_numpy() + 1e-3); X = X - X.mean(1, keepdims=True); y = lab.to_numpy(); aucs = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=17).split(X, y):
        mdl = LogisticRegression(C=.1, class_weight='balanced', max_iter=4000).fit(X[tr], y[tr]); aucs.append(roc_auc_score(y[te], mdl.decision_function(X[te])))
    json.dump(dict(composition_auc_mean=float(np.mean(aucs)), sd=float(np.std(aucs)), folds=len(aucs), clusters=int(pool.shape[1]),
                   spectra_auc=float(roc_auc_score(y, donors.loc[pool.index, 'probability']))), open(OUT / 'composition_reference.json', 'w'), indent=1)
    print(prof.join(st)[['cells', 'donors', 'CK', 'CD45_CD31', 'VIM', 'DAPI', 'area', 'CK_gate_fraction', 'abundance_auc', 'abundance_q', 'emphasis_log2_cancer',
                         'patients_with_attributed_cancer', 'patients_with_attributed_control']].round(3).to_string())
    print(open(OUT / 'composition_reference.json').read())


if __name__ == '__main__':
    main()
