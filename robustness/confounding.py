"""Revision step 1: study, site and acquisition confounding (2026-11-17).

Inference-free: uses saved locked-model test scores (primary 713 cancers / 44 reference controls), the saved
lung/breast reserved-control predictions of the pan-cancer source model (test and CV folds), and the saved training-CV
out-of-fold scores of the reference-policy CV. No model, threshold or endpoint changes. All analyses are descriptive
or exploratory and follow test inspection.

Outputs (analyses/revision_20261117/confounding): aggregate CSVs/JSON without patient identifiers, plus *_private.csv.
"""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
from scipy.stats import norm, kruskal, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'figures/code')); import sx  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent)); from sites import patient_sites, LABMETA  # noqa: E402

OUT = REPO / 'analyses/revision_20261117/confounding'; OUT.mkdir(parents=True, exist_ok=True)
B = 4000; rng = np.random.default_rng(20261117)
OPS = sx.operating_points()
T90, T95 = (np.log(OPS[t] / (1 - OPS[t])) for t in (.9, .95))  # locked cut-offs on the log-odds scale


def wilson(k, n, z=norm.ppf(.975)):
    if n == 0: return np.nan, np.nan
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def auc(pos, neg):
    allv = np.concatenate([pos, neg]); r = pd.Series(allv).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_auc(pos, neg, b=B):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2: return auc(pos, neg) if len(pos) and len(neg) else np.nan, np.nan, np.nan
    I = rng.integers(0, len(pos), (b, len(pos))); J = rng.integers(0, len(neg), (b, len(neg)))
    reps = np.array([auc(pos[I[k]], neg[J[k]]) for k in range(b)])
    lo, hi = np.percentile(reps, [2.5, 97.5]); return auc(pos, neg), lo, hi


# ---------------------------------------------------------------- inputs
led = patient_sites()
test = sx.primary_predictions().drop(columns=['study'], errors='ignore').merge(led[['donor_id', 'study', 'site', 'patient_breast_early_late']], on='donor_id', how='left', validate='one_to_one')
acq = pd.read_csv(REPO / 'analyses/batch_effects_20261111/patient_acquisition_private.csv', dtype={'donor_id': str})
test = test.merge(acq[['donor_id', 'scanner', 'scan_year', 'DAPI_exposure', 'FITC_exposure', 'TRITC_exposure', 'CY5_exposure']], on='donor_id', how='left')
assert len(test) == 757 and test.label.sum() == 713
ca, co = test[test.label == 1], test[test.label == 0]

oof = pd.read_csv(REPO / 'analyses/reference_control_refit_20261110/private_OOF_predictions.csv', dtype={'donor_id': str})
oof['score'] = sx.logit(oof.probability)
oof = oof.merge(led[['donor_id', 'site', 'patient_breast_early_late']], on='donor_id', how='left')
spec = pd.read_csv(REPO / 'analyses/specific_adaptation_20261111/predictions_private.csv', dtype={'donor_id': str})
spec = spec[spec.arm == 'source'].copy(); spec['score'] = sx.logit(spec.probability)
spec = spec.merge(led[['donor_id', 'study', 'site']], on='donor_id', how='left')
results = {}

# ---------------------------------------------------------------- 1.1 study x label x acquisition table
slm = pd.read_csv(REPO / 'analyses/batch_effects_20261111/slide_acquisition_metadata_private.csv', dtype={'slide_id': str})
sl = pd.read_csv(REPO / 'analyses/cohort_revision_20261109/input/spectra_study_slides_annotated.csv', usecols=['patient_id', 'slide_id'])
labm = pd.read_csv(LABMETA, usecols=['slide_id', 'tube_type'])
sl = sl.merge(slm, on='slide_id', how='left').merge(labm, on='slide_id', how='left')
sl = sl.merge(led[['patient_id', 'donor_id', 'study', 'label', 'split', 'primary_cancer_vs_reference_eligible', 'cancer_types', 'site']], on='patient_id', how='inner')
sl['year'] = pd.to_datetime(sl.scan_date, errors='coerce').dt.year
rows = []
for st, g in led.groupby('study'):
    s = sl[sl.study == st]
    role = 'cancer' if g.label.eq(1).all() else ('reference control' if g.primary_reference_control.any() else 'reserved control' if g.label.eq(0).all() else 'mixed')
    if g.label.eq(0).any() and g.label.eq(1).any(): role = 'cancers + reserved controls'
    yrs = s.year.dropna()
    rows.append(dict(study=st, role=role, site=', '.join(sorted(g.site.unique())),
                     cancer_types=', '.join(sorted(g.cancer_types.dropna().unique())),
                     cancers_train=int(((g.label == 1) & (g.split == 'train')).sum()), cancers_test=int(((g.label == 1) & (g.split == 'test')).sum()),
                     controls_train=int(((g.label == 0) & (g.split == 'train')).sum()), controls_test=int(((g.label == 0) & (g.split == 'test')).sum()),
                     scan_years=f'{int(yrs.min())}-{int(yrs.max())}' if len(yrs) else 'NA',
                     scanners=', '.join(str(x) for x in sorted(s.instrument.dropna().astype(str).unique())) or 'NA',
                     staining_experiments=int(s.expt_id.nunique()),
                     tube_type=', '.join(sorted(s.tube_type.dropna().astype(str).unique())) or 'NA',
                     median_DAPI_exposure=round(float(s.DAPI_exposure.median()), 1) if s.DAPI_exposure.notna().any() else np.nan))
tab = pd.DataFrame(rows).sort_values(['role', 'study'])
tab.to_csv(OUT / 'study_acquisition_table.csv', index=False)

# ---------------------------------------------------------------- 1.2 per control source
src = []
for s, g in co.groupby('study'):
    a, lo, hi = boot_auc(ca.score, g.score)
    k90, k95 = int((g.score < T90 - 1e-9).sum()), int((g.score < T95 - 1e-9).sum())
    src.append(dict(control_source=s, site=g.site.iloc[0], controls=len(g), cancers=len(ca), AUROC=a, ci_low=lo, ci_high=hi,
                    median_score=g.score.median(), spec90_cutoff=f'{k90}/{len(g)}', spec90=k90 / len(g), spec90_ci=wilson(k90, len(g)),
                    spec95_cutoff=f'{k95}/{len(g)}', spec95=k95 / len(g), spec95_ci=wilson(k95, len(g))))
    # leave this source out of the pooled control set
    rest = co[co.study != s]; a2, lo2, hi2 = boot_auc(ca.score, rest.score)
    src[-1].update(AUROC_without_source=a2, without_ci_low=lo2, without_ci_high=hi2, controls_without=len(rest))
srcdf = pd.DataFrame(src); srcdf.to_csv(OUT / 'per_control_source.csv', index=False)
kw = kruskal(*[g.score for _, g in co.groupby('study')])
results['control_source_kruskal'] = dict(H=kw.statistic, p=kw.pvalue)
# OOF (training CV) per source, using the per-fold-pooled OOF scores
srco = []
oca, oco = oof[oof.label == 1], oof[oof.label == 0]
for s, g in oco.groupby('study'):
    a, lo, hi = boot_auc(oca.score, g.score)
    srco.append(dict(control_source=s, controls=len(g), cancers=len(oca), AUROC=a, ci_low=lo, ci_high=hi, median_score=g.score.median()))
pd.DataFrame(srco).to_csv(OUT / 'per_control_source_trainingOOF.csv', index=False)

# ---------------------------------------------------------------- 1.3 same-site / same-study comparisons
same = []
def comp(name, pos, neg, setting, note=''):
    a, lo, hi = boot_auc(pos, neg)
    same.append(dict(comparison=name, setting=setting, cancers=len(pos), controls=len(neg), AUROC=a, ci_low=lo, ci_high=hi, note=note))

T = test; O = oof
for setting, D in [('test', T), ('training CV out-of-fold', O)]:
    c, n = D[D.label == 1], D[D.label == 0]
    comp('Cedars-Sinai pancreatic: PANCEUS vs PANC-Ctrl (one study)', c[c.study == 'PANCEUS'].score, n[n.study == 'PANC-Ctrl'].score, setting)
    comp('Cedars-Sinai pancreatic: PANCEUS + PANC vs PANC-Ctrl', c[c.study.isin(['PANCEUS', 'PANC'])].score, n[n.study == 'PANC-Ctrl'].score, setting)
    comp('Cedars-Sinai breast: INTC17 vs INTC17-C', c[c.study == 'INTC17'].score, n[n.study == 'INTC17-C'].score, setting)
    comp('Cedars-Sinai: all cancers vs all reference controls', c[c.site == 'Cedars-Sinai'].score, n[n.site == 'Cedars-Sinai'].score, setting)
    comp('Pooled primary comparison (all sites)', c.score, n.score, setting)
# lung workup controls within study (pan-cancer source model; reserved controls)
for setting, parts in [('test', ['test']), ('training CV out-of-fold', ['cv0', 'cv1', 'cv2'])]:
    D = spec[(spec.task == 'lung') & spec.part.isin(parts)]
    c, n = D[D.label == 1], D[D.label == 0]
    comp('Stanford lung: SU cancers vs SU workup controls (one study)', c[c.study == 'SU'].score, n[n.study == 'SU'].score, setting, 'pan-cancer model; workup controls')
    comp('USC lung: JN/JN2/JN2-R cancers vs workup controls (one study family)', c[c.site == 'USC (JN)'].score, n[n.site == 'USC (JN)'].score, setting, 'pan-cancer model; workup controls')
    comp('All lung cancers vs all lung workup controls', c.score, n.score, setting, 'as reported (Fig. 6 source arm)')
pd.DataFrame(same).to_csv(OUT / 'same_site_comparisons.csv', index=False)

# ---------------------------------------------------------------- 1.4 technical covariates
cov = {}
for s, g in co.groupby('study'):
    if len(g) >= 5:
        r = spearmanr(g.score, g.DAPI_exposure, nan_policy='omit'); cov[f'DAPI_vs_score_within_{s}'] = dict(rho=r.statistic, p=r.pvalue, n=int(g.DAPI_exposure.notna().sum()))
r = spearmanr(co.score, co.DAPI_exposure, nan_policy='omit'); cov['DAPI_vs_score_all_controls'] = dict(rho=r.statistic, p=r.pvalue, n=int(co.DAPI_exposure.notna().sum()))
r = spearmanr(ca.score, ca.DAPI_exposure, nan_policy='omit'); cov['DAPI_vs_score_all_cancers'] = dict(rho=r.statistic, p=r.pvalue, n=int(ca.DAPI_exposure.notna().sum()))
# residualise DAPI on control source, then correlate within controls
cc = co.dropna(subset=['DAPI_exposure']).copy()
cc['dapi_res'] = cc.DAPI_exposure - cc.groupby('study').DAPI_exposure.transform('median')
cc['score_res'] = cc.score - cc.groupby('study').score.transform('median')
r = spearmanr(cc.score_res, cc.dapi_res); cov['DAPI_vs_score_controls_source_centred'] = dict(rho=r.statistic, p=r.pvalue, n=len(cc))
cov['DAPI_exposure_median_by_control_source'] = co.groupby('study').DAPI_exposure.median().round(1).to_dict()
cov['DAPI_exposure_median_cancers'] = float(ca.DAPI_exposure.median())
# logistic: label ~ score + covariates (standardised); report score coefficient with and without adjustment, and AUROC of covariates alone
dd = test.dropna(subset=['DAPI_exposure', 'scan_year']).copy()
dd['scanner5'] = (dd.scanner.astype(str) == '5').astype(int); dd['scanner3'] = (dd.scanner.astype(str) == '3').astype(int)
X = dd[['DAPI_exposure', 'scan_year', 'scanner3', 'scanner5']].astype(float); X = (X - X.mean()) / X.std()
lr = LogisticRegression(C=1e6, class_weight='balanced', max_iter=5000)
cv_cov = lr.fit(X, dd.label).predict_proba(X)[:, 1]
cov['covariates_only_in_sample_AUROC'] = roc_auc_score(dd.label, cv_cov)
Xs = X.assign(score=(dd.score - dd.score.mean()) / dd.score.std())
m1 = LogisticRegression(C=1e6, class_weight='balanced', max_iter=5000).fit(Xs[['score']], dd.label)
m2 = LogisticRegression(C=1e6, class_weight='balanced', max_iter=5000).fit(Xs, dd.label)
cov['score_coef_unadjusted'] = float(m1.coef_[0][0]); cov['score_coef_adjusted'] = float(m2.coef_[0][-1])
cov['adjusted_model_coefs'] = dict(zip(Xs.columns, m2.coef_[0].round(3).tolist())); cov['n_complete'] = len(dd)
# bootstrap the adjusted score coefficient
bc = []
for _ in range(1000):
    i = np.r_[rng.choice(np.where(dd.label == 1)[0], (dd.label == 1).sum()), rng.choice(np.where(dd.label == 0)[0], (dd.label == 0).sum())]
    bc.append(LogisticRegression(C=1e6, class_weight='balanced', max_iter=5000).fit(Xs.iloc[i], dd.label.iloc[i]).coef_[0][-1])
cov['score_coef_adjusted_ci'] = np.percentile(bc, [2.5, 97.5]).tolist()
# DAPI-exposure-matched AUROC: each control vs cancers within +/-10% exposure
pairs = [];
for _, c in co.dropna(subset=['DAPI_exposure']).iterrows():
    m = ca[(ca.DAPI_exposure - c.DAPI_exposure).abs() <= .1 * c.DAPI_exposure]
    pairs += [(x > c.score) + .5 * (x == c.score) for x in m.score]
cov['DAPI_matched_10pct_AUROC'] = float(np.mean(pairs)); cov['DAPI_matched_10pct_pairs'] = len(pairs)
results['covariates'] = cov

# ---------------------------------------------------------------- 1.6 same cancer type, different sites
site_rows, pair_rows, kw_rows = [], [], []
GROUPS = {
    'Breast': ['Billings Clinic', 'Duke', 'Cedars-Sinai', 'Walter Reed', 'Anne Arundel', 'University of Colorado',
               'Unknown (S1222)', 'NRG-BR002 sites', 'City of Hope', 'USC'],  # S1416 (triple-negative) excluded; the other breast cohorts are HR+
    'Prostate': ['MD Anderson', 'Morehouse/Grady', 'UAB', 'City of Hope', 'UPenn', 'Johns Hopkins', 'UCLA'],
    'Lung': ['Stanford', 'USC', 'USC (JN)', 'Manchester', 'MSKCC'],
}
LEAD = {'Breast': [('Billings Clinic', 'Duke'), ('Walter Reed', 'Anne Arundel'), ('Billings Clinic', 'Cedars-Sinai'), ('Duke', 'Cedars-Sinai')],
        'Prostate': [('MD Anderson', 'Morehouse/Grady'), ('MD Anderson', 'City of Hope'), ('MD Anderson', 'Johns Hopkins'), ('MD Anderson', 'UCLA')],
        'Lung': [('Stanford', 'USC'), ('Stanford', 'Manchester'), ('Stanford', 'USC (JN)')]}
for setting, D in [('test', ca), ('training CV out-of-fold', oca)]:
    for ct, sites in GROUPS.items():
        d = D[D.cancer_types == ct]
        for s in sites:
            g = d[d.site == s]
            if not len(g): continue
            k = int((g.score >= T90 - 1e-9).sum()) if setting == 'test' else np.nan
            site_rows.append(dict(setting=setting, cancer_type=ct, site=s, patients=len(g), median_score=g.score.median(),
                                  q25=g.score.quantile(.25), q75=g.score.quantile(.75),
                                  sens90_locked=(k / len(g)) if setting == 'test' else np.nan,
                                  sens90_ci=wilson(k, len(g)) if setting == 'test' else None,
                                  AUROC_vs_44_reference_controls=auc(g.score.to_numpy(), co.score.to_numpy()) if setting == 'test' else np.nan))
        gs = [d[d.site == s].score for s in sites if (d.site == s).sum() >= 5]
        if len(gs) >= 2:
            k_ = kruskal(*gs); kw_rows.append(dict(setting=setting, cancer_type=ct, sites_with_5plus=len(gs), H=k_.statistic, p=k_.pvalue))
        for a_, b_ in LEAD[ct]:
            x, y = d[d.site == a_].score, d[d.site == b_].score
            if len(x) >= 3 and len(y) >= 3:
                a, lo, hi = boot_auc(x, y)
                pair_rows.append(dict(setting=setting, cancer_type=ct, site_a=a_, site_b=b_, n_a=len(x), n_b=len(y),
                                      AUROC_site_a_vs_b=a, ci_low=lo, ci_high=hi))
pd.DataFrame(site_rows).to_csv(OUT / 'cancer_scores_by_site.csv', index=False)
pd.DataFrame(pair_rows).to_csv(OUT / 'site_pairwise_AUROC.csv', index=False)
pd.DataFrame(kw_rows).to_csv(OUT / 'site_kruskal.csv', index=False)

# breast: score ~ site + stage (OLS on log-odds), test + OOF pooled per setting
from scipy.stats import f as fdist


def ols_r2(y, cats):
    X = np.column_stack([np.ones(len(y))] + [pd.get_dummies(c, drop_first=True).to_numpy(float) for c in cats])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); rss = ((y - X @ beta) ** 2).sum()
    return 1 - rss / ((y - y.mean()) ** 2).sum(), rss, X.shape[1]


reg = {}
for setting, D in [('test', ca), ('training CV out-of-fold', oca)]:
    d = D[(D.cancer_types == 'Breast') & (D.site != 'SWOG S1416 sites')].copy()  # S1416 (triple-negative) excluded; the other breast cohorts are HR+
    d['stage'] = d.patient_breast_early_late.replace({'Conflicting': 'Unknown'}).fillna('Unknown')
    d = d[d.site.map(d.site.value_counts()) >= 5]
    y = d.score.to_numpy(float)
    r0, rss0, k0 = ols_r2(y, [d.stage]); r1, rss1, k1 = ols_r2(y, [d.stage, d.site])
    F = ((rss0 - rss1) / (k1 - k0)) / (rss1 / (len(y) - k1))
    reg[setting] = dict(n=len(d), sites=sorted(d.site.unique()), R2_stage=r0, R2_stage_site=r1,
                        F_site_given_stage=F, F_site_given_stage_p=float(fdist.sf(F, k1 - k0, len(y) - k1)))
    # early-stage only: Billings vs Duke vs Cedars-Sinai
    e = d[(d.stage == 'Early') & d.site.isin(['Billings Clinic', 'Duke', 'Cedars-Sinai'])]
    if e.site.nunique() >= 2:
        reg[setting]['early_stage_counts'] = e.site.value_counts().to_dict()
        reg[setting]['early_stage_kruskal_p'] = kruskal(*[g.score for _, g in e.groupby('site')]).pvalue
        reg[setting]['early_stage_median'] = e.groupby('site').score.median().round(2).to_dict()
results['breast_site_stage_regression'] = reg
results['locked_cutoffs_logodds'] = dict(spec90=T90, spec95=T95)
json.dump(results, open(OUT / 'summary.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

# private patient-level file for figures
test[['donor_id', 'label', 'study', 'site', 'cancer_types', 'score', 'DAPI_exposure', 'scan_year', 'scanner']].to_csv(OUT / 'test_scores_with_site_private.csv', index=False)
oof[['donor_id', 'label', 'study', 'site', 'cancer_types', 'score', 'fold']].to_csv(OUT / 'oof_scores_with_site_private.csv', index=False)
print('done')
