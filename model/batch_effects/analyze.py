"""Batch-effect checks on the locked primary test predictions (713 cancers / 44 reference controls), inference-free.

Acquisition covariates per patient (primary-group slides): scanner instrument, scan date, experiment (processing)
ID and channel exposures, read from slideinfo.txt (collect.py).
1. Score by scanner within cancers and within controls (Kruskal-Wallis), AUROC within each scanner, and a
   scanner-stratified AUROC that only compares cancer-control pairs scanned on the same instrument.
2. Score vs scan date within cancers and within controls (Spearman); AUROC stratified by scan year and by
   pairs scanned within 180 days of each other.
3. Experiment ID: number of processing batches holding both cancers and controls; within-batch pair AUROC.
4. Exposure: score vs channel exposure within label (Spearman).
5. Covariate-adjusted logistic regression (unpenalized): label ~ score + scanner + scan year; likelihood-ratio test for score.
Patients whose two primary slides span two instruments are assigned 'mixed'. No model refits or retuning."""
import os
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'figures/code')); import sx  # noqa: E402
OUT = REPO / 'analyses/batch_effects_20261111'

p = sx.primary_predictions()
s = pd.read_csv(os.environ.get('SPECTRA_SLIDE_TABLE', 'data/slides.csv'), dtype=str)
m = pd.read_csv(OUT / 'slide_acquisition_metadata_private.csv', dtype={'slide_id': str, 'instrument': str}, parse_dates=['scan_date'])
t = s.merge(m, on='slide_id')
t = t[t.canonical_patient_id.isin(p.donor_id) & t.primary_evaluation.eq('True')]
for ch in ['DAPI', 'FITC', 'TRITC', 'CY5']:
    t[f'{ch}_exposure'] = pd.to_numeric(t[f'{ch}_exposure'], errors='coerce')
d = t.groupby('canonical_patient_id').agg(
    scanner=('instrument', lambda x: (lambda u: u[0] if len(u) == 1 else ('mixed' if len(u) else 'unknown'))(sorted(set(x.dropna())))),
    scan_date=('scan_date', 'min'), expt_id=('expt_id', 'first'), n_expt=('expt_id', 'nunique'),
    **{f'{ch}_exposure': (f'{ch}_exposure', 'mean') for ch in ['DAPI', 'FITC', 'TRITC', 'CY5']})
d = p.set_index('donor_id')[['label', 'cancer_types', 'study', 'score', 'probability']].join(d)
d.index.name = 'donor_id'
R = {}

def strat_auc(df, key):
    """AUROC counting only cancer-control pairs that share `key`."""
    num = den = 0.0
    for _, g in df.groupby(key):
        a, b = g.score[g.label == 1].to_numpy(), g.score[g.label == 0].to_numpy()
        if len(a) and len(b):
            num += ((a[:, None] > b[None]).sum() + .5 * (a[:, None] == b[None]).sum()); den += len(a) * len(b)
    return (num / den if den else np.nan), int(den)

# 1. scanner
R['overall_auroc'] = roc_auc_score(d.label, d.score)
R['scanner_counts'] = pd.crosstab(d.scanner, d.label).rename(columns={0: 'controls', 1: 'cancers'}).to_dict('index')
rows = []
for k, g in d.groupby('scanner'):
    rows.append(dict(scanner=k, cancers=int(g.label.sum()), controls=int((g.label == 0).sum()),
                     auroc=roc_auc_score(g.label, g.score) if g.label.nunique() == 2 else np.nan,
                     median_score_cancers=g.score[g.label == 1].median(), median_score_controls=g.score[g.label == 0].median()))
pd.DataFrame(rows).to_csv(OUT / 'by_scanner.csv', index=False)
main = d[d.scanner.isin(['3', '4', '5'])]
for lab, nm in [(1, 'cancers'), (0, 'controls')]:
    g = main[main.label == lab]
    R[f'kruskal_score_by_scanner_{nm}'] = dict(zip(['H', 'p'], map(float, stats.kruskal(*[x.score for _, x in g.groupby('scanner')]))))
R['scanner_stratified_auroc'], R['scanner_stratified_pairs'] = strat_auc(d[d.scanner != 'unknown'], 'scanner')

# 2. scan date
d['scan_year'] = d.scan_date.dt.year
d['days'] = (d.scan_date - pd.Timestamp('2019-01-01')).dt.days
for lab, nm in [(1, 'cancers'), (0, 'controls')]:
    g = d[(d.label == lab) & d.days.notna()]
    r = stats.spearmanr(g.days, g.score); R[f'spearman_score_vs_date_{nm}'] = dict(rho=float(r.statistic), p=float(r.pvalue), n=len(g))
R['scan_year_counts'] = pd.crosstab(d.scan_year, d.label).rename(columns={0: 'controls', 1: 'cancers'}).to_dict('index')
R['year_stratified_auroc'], R['year_stratified_pairs'] = strat_auc(d[d.scan_year.notna()], 'scan_year')
a = d[(d.label == 1) & d.days.notna()]; b = d[(d.label == 0) & d.days.notna()]
close = np.abs(a.days.to_numpy()[:, None] - b.days.to_numpy()[None]) <= 180
gt = (a.score.to_numpy()[:, None] > b.score.to_numpy()[None]) + .5 * (a.score.to_numpy()[:, None] == b.score.to_numpy()[None])
R['within180d_auroc'] = float(gt[close].mean()); R['within180d_pairs'] = int(close.sum())
yr = []
for y, g in d.groupby('scan_year'):
    yr.append(dict(scan_year=int(y), cancers=int(g.label.sum()), controls=int((g.label == 0).sum()),
                   auroc=roc_auc_score(g.label, g.score) if g.label.nunique() == 2 else np.nan))
pd.DataFrame(yr).to_csv(OUT / 'by_scan_year.csv', index=False)

# 3. processing batch (experiment ID)
mixed = d.groupby('expt_id').label.nunique()
R['experiments_total'] = int(d.expt_id.nunique()); R['experiments_with_both_classes'] = int((mixed == 2).sum())
R['experiment_stratified_auroc'], R['experiment_stratified_pairs'] = strat_auc(d[d.expt_id.notna()], 'expt_id')

# 4. exposures
ex = {}
for ch in ['DAPI', 'FITC', 'TRITC', 'CY5']:
    for lab, nm in [(1, 'cancers'), (0, 'controls')]:
        g = d[(d.label == lab) & d[f'{ch}_exposure'].notna()]
        r = stats.spearmanr(g[f'{ch}_exposure'], g.score); ex[f'{ch}_{nm}'] = dict(rho=float(r.statistic), p=float(r.pvalue), n=len(g))
R['spearman_score_vs_exposure'] = ex

# 5. covariate-adjusted logistic regression
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
g = d[d.scanner.isin(['3', '4', '5']) & d.scan_year.notna()].copy(); g['z'] = (g.score - g.score.mean()) / g.score.std()
cov = pd.get_dummies(g.scanner, prefix='scanner', drop_first=True).astype(float).assign(year=g.scan_year - 2022)
def ll(X):
    f = LogisticRegression(penalty=None, max_iter=5000).fit(X, g.label); return -log_loss(g.label, f.predict_proba(X)[:, 1], normalize=False), f
l_cov, _ = ll(cov); l_full, f_full = ll(cov.assign(z=g.z)); l_z, f_z = ll(g[['z']])
R['logit_n'] = len(g)
R['logit_score_coef_unadjusted'] = float(f_z.coef_[0][0])
R['logit_score_coef_adjusted_scanner_year'] = float(f_full.coef_[0][-1])
R['LRT_score_given_scanner_year'] = dict(chi2=float(2 * (l_full - l_cov)), p=float(stats.chi2.sf(2 * (l_full - l_cov), 1)))
R['adjusted_coefs'] = dict(zip(list(cov.columns) + ['z'], map(float, f_full.coef_[0])))
d.reset_index().to_csv(OUT / 'patient_acquisition_private.csv', index=False)
json.dump(R, open(OUT / 'summary.json', 'w'), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))

# 6. (added) scan-week strata and DAPI exposure strata
d['scan_week'] = d.scan_date.dt.to_period('W').astype(str)
R2 = {}
R2['week_stratified_auroc'], R2['week_stratified_pairs'] = strat_auc(d[d.scan_date.notna()], 'scan_week')
R2['weeks_with_both_classes'] = int((d[d.scan_date.notna()].groupby('scan_week').label.nunique() == 2).sum())
R2['DAPI_exposure_by_label'] = d.groupby('label').DAPI_exposure.describe().round(2).to_dict('index')
d['dapi_bin'] = pd.qcut(d.DAPI_exposure, 4, duplicates='drop').astype(str)
R2['DAPI_quartile_stratified_auroc'], R2['DAPI_quartile_pairs'] = strat_auc(d[d.DAPI_exposure.notna()], 'dapi_bin')
rows = []
for k, g in d[d.DAPI_exposure.notna()].groupby('dapi_bin'):
    rows.append(dict(DAPI_exposure_quartile=k, cancers=int(g.label.sum()), controls=int((g.label == 0).sum()),
                     auroc=roc_auc_score(g.label, g.score) if g.label.nunique() == 2 else np.nan))
pd.DataFrame(rows).to_csv(OUT / 'by_DAPI_exposure_quartile.csv', index=False)
c = d[(d.label == 0) & d.DAPI_exposure.notna()]
R2['controls_DAPI_exposure_vs_scanner'] = c.groupby('scanner').DAPI_exposure.median().to_dict()
R2['controls_DAPI_exposure_vs_date_spearman'] = float(stats.spearmanr(c.days, c.DAPI_exposure).statistic)
R.update(R2)
json.dump(R, open(OUT / 'summary.json', 'w'), indent=1, default=float)
print(json.dumps(R2, indent=1, default=float))
