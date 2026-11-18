"""Revision step 2.1b analysis: which image features of cluster 10 cells differ between groups (2026-11-17). Exploratory.

Patient-level means of every feature over the patient's extracted cluster-10 cells. For each feature: AUROC (group a vs b) with
stratified bootstrap CI, Mann-Whitney P, Benjamini-Hochberg q within each comparison. Multivariate: L2 logistic regression on all
standardised patient-mean features, 5x5 repeated stratified CV AUROC (cancer vs reference control), with a 200-permutation null,
for comparison with the within-cluster embedding AUROC (0.61).
"""
from pathlib import Path
import glob, json
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / 'runs/revision_20261117/cell_features'
OUT = REPO / 'analyses/revision_20261117/cell_features'; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261117); B = 2000
sel = pd.read_parquet(RUN / 'selected_cells_private.parquet')
d = pd.concat([pd.read_parquet(f) for f in sorted(RUN.glob('*.parquet')) if f.name != 'selected_cells_private.parquet'], ignore_index=True)
feats = [c for c in d.columns if c not in ('cell_key', 'donor_id', 'group')]
pp = d.groupby(['donor_id', 'group'])[feats].mean().reset_index()
cov = dict(selected=sel.groupby('group').size().to_dict(), extracted=d.groupby('group').size().to_dict(),
           patients_selected=sel.groupby('group').donor_id.nunique().to_dict(), patients_extracted=pp.groupby('group').size().to_dict())


def auc(a, b):
    allv = np.concatenate([a, b]); rk = pd.Series(allv).rank().to_numpy()
    return (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


rows = []
for ga, gb in [('Cancer', 'Reference control'), ('BRCA-carrier control', 'Reference control'), ('Lung work-up control', 'Reference control'),
               ('Cancer', 'BRCA-carrier control'), ('Cancer', 'Lung work-up control')]:
    A, Bb = pp[pp.group == ga], pp[pp.group == gb]
    for f in feats:
        a, b = A[f].dropna().to_numpy(), Bb[f].dropna().to_numpy()
        if len(a) < 3 or len(b) < 3: continue
        I, J = rng.integers(0, len(a), (B, len(a))), rng.integers(0, len(b), (B, len(b)))
        r = np.array([auc(a[I[k]], b[J[k]]) for k in range(B)])
        rows.append(dict(group_a=ga, group_b=gb, feature=f, n_a=len(a), n_b=len(b), median_a=np.median(a), median_b=np.median(b),
                         AUROC=auc(a, b), ci_low=np.percentile(r, 2.5), ci_high=np.percentile(r, 97.5), p=mannwhitneyu(a, b).pvalue))
res = pd.DataFrame(rows)
for (ga, gb), g in res.groupby(['group_a', 'group_b']):
    pv = g.p.to_numpy(); o = np.argsort(pv); q = pv[o] * len(pv) / np.arange(1, len(pv) + 1); q = np.minimum.accumulate(q[::-1])[::-1]
    qq = np.empty_like(q); qq[o] = np.minimum(q, 1); res.loc[g.index, 'q'] = qq
res['effect'] = (res.AUROC - .5).abs()
res.sort_values(['group_a', 'group_b', 'p']).to_csv(OUT / 'feature_comparisons.csv', index=False)

# multivariate CV: cancer vs reference control
m = pp[pp.group.isin(['Cancer', 'Reference control'])].dropna(axis=1, thresh=int(.9 * len(pp)))
fs = [f for f in feats if f in m.columns]; Xm = m[fs].fillna(m[fs].median()).to_numpy(); y = (m.group == 'Cancer').astype(int).to_numpy()
def cv_auc(X, y):
    out = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=17).split(X, y):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(C=.1, class_weight='balanced', max_iter=5000)).fit(X[tr], y[tr])
        out.append(roc_auc_score(y[te], mdl.decision_function(X[te])))
    return float(np.mean(out))
obs = cv_auc(Xm, y); null = np.array([cv_auc(Xm, rng.permutation(y)) for _ in range(200)])
multi = dict(features=len(fs), patients=len(y), cancers=int(y.sum()), controls=int((1 - y).sum()), cv_AUROC=obs, perm_mean=float(null.mean()),
             perm_p=float((1 + (null >= obs).sum()) / 201))
json.dump(dict(coverage=cov, multivariate_cancer_vs_reference=multi), open(OUT / 'summary.json', 'w'), indent=1, default=str)
pd.set_option('display.width', 250)
print(json.dumps(dict(coverage=cov, multivariate=multi), indent=1, default=str))
for (ga, gb), g in res.groupby(['group_a', 'group_b']):
    print(f'\n== {ga} vs {gb}: features q<0.05: {(g.q < .05).sum()} of {len(g)}')
    print(g.sort_values('p').head(8)[['feature', 'median_a', 'median_b', 'AUROC', 'ci_low', 'ci_high', 'p', 'q']].round(3).to_string(index=False))
