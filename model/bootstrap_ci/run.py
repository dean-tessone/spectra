"""95% confidence intervals for manuscript test metrics. Inference-free: uses saved locked-model test predictions only.

- AUROC: stratified patient bootstrap (cancers and controls resampled separately, B=4000, seed 20261111),
  percentile intervals. Per-cancer AUROCs resample that cancer type together with the common 44 controls.
  Macro AUROC: mean over types with >=10 cases, recomputed in each replicate.
- Sensitivity/specificity at the fixed reported operating thresholds (common overall 90%/95% specificity cut-offs):
  Wilson score intervals for the observed proportions (thresholds are not re-derived).
- Paired differences (SPECTRA minus marker burden, adapted minus pan-cancer): same patients in each replicate.
"""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'figures/code')); import sx  # noqa: E402
OUT = REPO / 'analyses/bootstrap_ci_20261111'; OUT.mkdir(parents=True, exist_ok=True)
B = 4000; rng = np.random.default_rng(20261111)


def wilson(k, n, z=norm.ppf(.975)):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def auc_fast(pos, neg):
    """Mann-Whitney AUROC with ties counted half."""
    allv = np.concatenate([pos, neg]); r = pd.Series(allv).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_idx(n):
    return rng.integers(0, n, (B, n))


p = sx.primary_predictions()
neg = p[p.label == 0]; posall = p[p.label == 1]
NI = boot_idx(len(neg))
rows = []

def add(name, est, reps, kind='AUROC', **kw):
    lo, hi = np.nanpercentile(reps, [2.5, 97.5]) if reps is not None else (np.nan, np.nan)
    rows.append(dict(metric=name, kind=kind, estimate=est, ci_low=lo, ci_high=hi, **kw))

# --- primary pooled AUROC
PI = boot_idx(len(posall)); ps, ns = posall.score.to_numpy(), neg.score.to_numpy()
rep = np.array([auc_fast(ps[PI[b]], ns[NI[b]]) for b in range(B)])
add('Primary pooled AUROC (713 cancers vs 44 controls)', auc_fast(ps, ns), rep, cases=len(ps), controls=len(ns))

# --- per cancer type and macro
types = posall.cancer_types.value_counts(); per = {}
for ct, n in types.items():
    v = posall.score[posall.cancer_types == ct].to_numpy(); I = rng.integers(0, len(v), (B, len(v)))
    per[ct] = np.array([auc_fast(v[I[b]], ns[NI[b]]) for b in range(B)])
    add(f'AUROC {ct}', auc_fast(v, ns), per[ct] if n >= 10 else None, cases=int(n), controls=len(ns), note='' if n >= 10 else 'fewer than 10 cases; descriptive, no interval')
mac = [ct for ct, n in types.items() if n >= 10]
add(f'Macro AUROC ({len(mac)} types with >=10 cases)', np.mean([auc_fast(posall.score[posall.cancer_types == c].to_numpy(), ns) for c in mac]),
    np.mean([per[c] for c in mac], axis=0), cases=int(types[mac].sum()), controls=len(ns))

# --- operating points (fixed thresholds)
for tgt, col in [(.9, 'called90'), (.95, 'called95')]:
    k = int(posall[col].sum()); n = len(posall); lo, hi = wilson(k, n)
    rows.append(dict(metric=f'Sensitivity at {int(tgt * 100)}% specificity target', kind='proportion', estimate=k / n, ci_low=lo, ci_high=hi, cases=n, k=k))
    k0 = int((~neg[col]).sum()); lo, hi = wilson(k0, len(neg))
    rows.append(dict(metric=f'Specificity achieved at {int(tgt * 100)}% target', kind='proportion', estimate=k0 / len(neg), ci_low=lo, ci_high=hi, controls=len(neg), k=k0))
    for ct, n_ in types.items():
        kk = int(posall[col][posall.cancer_types == ct].sum()); lo, hi = wilson(kk, n_)
        rows.append(dict(metric=f'Sensitivity at {int(tgt * 100)}% specificity target, {ct}', kind='proportion', estimate=kk / n_, ci_low=lo, ci_high=hi, cases=int(n_), k=kk))

# --- markers (paired)
m = sx.markers()
d = p.merge(m[['donor_id', 'CK_gate', 'CTC_like', 'CTCs']], on='donor_id', validate='one_to_one')   # counts per patient (summed over the patient's analysed slides)
dp, dn = d[d.label == 1], d[d.label == 0]
I1 = boot_idx(len(dp)); I0 = boot_idx(len(dn))
base = np.array([auc_fast(dp.score.to_numpy()[I1[b]], dn.score.to_numpy()[I0[b]]) for b in range(B)])
for col, nm in [('CK_gate', 'CK-gated burden'), ('CTC_like', 'CTC-like burden'), ('CTCs', 'CTCs-only burden')]:
    a, b_ = dp[col].to_numpy(), dn[col].to_numpy()
    r = np.array([auc_fast(a[I1[b]], b_[I0[b]]) for b in range(B)])
    add(f'AUROC {nm}', auc_fast(a, b_), r, cases=len(a), controls=len(b_))
    add(f'AUROC difference SPECTRA minus {nm} (paired)', auc_fast(dp.score.to_numpy(), dn.score.to_numpy()) - auc_fast(a, b_), base - r, kind='AUROC difference')

# --- breast stage
led = pd.read_csv(REPO / 'analyses/clinical_cohort_annotations_20261110/patient_clinical_ledger.csv', dtype={'donor_id': str})
br = posall.merge(led[['donor_id', 'breast_early_late']], on='donor_id')
br = br[br.cancer_types == 'Breast']
for st in ['Early', 'Late', 'Unknown']:
    v = br.score[br.breast_early_late == st].to_numpy(); I = rng.integers(0, len(v), (B, len(v)))
    add(f'AUROC breast {st}', auc_fast(v, ns), np.array([auc_fast(v[I[b]], ns[NI[b]]) for b in range(B)]), cases=len(v), controls=len(ns))

# --- adaptation (lung, breast)
pr = pd.read_csv(REPO / 'analyses/specific_adaptation_20261111/predictions_private.csv', dtype={'donor_id': str}); pr = pr[pr.part == 'test']
for task in ['lung', 'breast']:
    w = pr[pr.task == task].pivot_table(index=['donor_id', 'label'], columns='arm', values='probability').reset_index()
    wp, wn = w[w.label == 1], w[w.label == 0]
    J1 = rng.integers(0, len(wp), (B, len(wp))); J0 = rng.integers(0, len(wn), (B, len(wn)))
    reps = {}
    for arm in ['source', 'probe', 'adapt', 'lora']:
        a, b_ = wp[arm].to_numpy(), wn[arm].to_numpy()
        reps[arm] = np.array([auc_fast(a[J1[b]], b_[J0[b]]) for b in range(B)])
        add(f'Adaptation {task} AUROC {arm}', auc_fast(a, b_), reps[arm], cases=len(a), controls=len(b_))
    for arm in ['probe', 'adapt', 'lora']:
        add(f'Adaptation {task} AUROC gain {arm} minus source (paired)', auc_fast(wp[arm].to_numpy(), wn[arm].to_numpy()) - auc_fast(wp.source.to_numpy(), wn.source.to_numpy()),
            reps[arm] - reps['source'], kind='AUROC difference')

r = pd.DataFrame(rows)
r.to_csv(OUT / 'confidence_intervals.csv', index=False)
json.dump(dict(B=B, seed=20261111, method=__doc__), open(OUT / 'method.json', 'w'), indent=1)
pd.set_option('display.width', 250)
print(r[['metric', 'estimate', 'ci_low', 'ci_high']].round(3).to_string())
