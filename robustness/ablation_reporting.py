"""Revision step 3: complete reporting of the locked-model cell-removal experiment (inference-free, 2026-11-17).

Reads the saved per-patient removal scores (all 757 primary test patients; cancers and controls ablated identically) and reports for
every condition: AUROC with stratified-bootstrap CI, paired AUROC difference versus matched random removal and versus no removal,
sensitivity and achieved specificity at the original locked cut-offs, and sensitivity at a cut-off re-estimated on the altered
control scores to give the same 40/44 (90%) and 42/44 (95%) controls below it.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / 'analyses/figure3_locked6_attribution_20261111/removal_per_patient_private.csv'
OUT = REPO / 'analyses/revision_20261117/ablation'; OUT.mkdir(parents=True, exist_ok=True)
ops = pd.read_csv(REPO / 'analyses/reference_control_test_20261110_v1a/operating_points.csv')
ops = ops[(ops.endpoint == 'primary') & (ops.cancer_type == 'Overall') & (ops.comparison == 'Cancer vs primary reference')].set_index('target_specificity').threshold
B = 4000; rng = np.random.default_rng(20261117)
r = pd.read_csv(SRC); y = r.label.to_numpy(); P, N = np.where(y == 1)[0], np.where(y == 0)[0]
assert len(P) == 713 and len(N) == 44
PI, NI = rng.integers(0, len(P), (B, len(P))), rng.integers(0, len(N), (B, len(N)))


def auc(pos, neg):
    allv = np.concatenate([pos, neg]); rk = pd.Series(allv).rank().to_numpy()
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot(col):
    v = r[col].to_numpy(); return np.array([auc(v[P][PI[b]], v[N][NI[b]]) for b in range(B)])


def wilson(k, n, z=norm.ppf(.975)):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def refit_sens(v, keep_controls):
    """Cut-off placed so that exactly `keep_controls` of 44 controls fall below it (same rule as the locked report), then sensitivity."""
    neg = np.sort(v[N]); t = neg[keep_controls - 1]  # highest control kept below; cases must exceed it
    return float((v[P] > t).mean()), int((v[P] > t).sum())


base_rep = boot('base'); rows = []
LABEL = {'none': 'No removal', 'top10': 'Top-10 attributed cells', 'top50': 'Top-50 attributed cells', 'mass90': 'Cells carrying 90% of positive attribution',
         'CK_gate': 'CK-gated cells', 'CTC_like': 'All CTC-like cells', 'state8': 'CK-bright state (not in paper)'}
for cond in ['none', 'top10', 'top50', 'mass90', 'CK_gate', 'CTC_like', 'state8']:
    cols = [('removed', 'base' if cond == 'none' else f'{cond}_removed')] + ([] if cond == 'none' else [('matched random', f'{cond}_random')])
    reps = {}
    for arm, col in cols:
        v = r[col].to_numpy(); reps[arm] = boot(col) if col != 'base' else base_rep
        k90, k95 = int((v[P] >= ops[.9]).sum()), int((v[P] >= ops[.95]).sum())
        c90, c95 = int((v[N] < ops[.9]).sum()), int((v[N] < ops[.95]).sum())
        s90r, k90r = refit_sens(v, 40); s95r, k95r = refit_sens(v, 42)
        rows.append(dict(condition=LABEL[cond], arm=arm, AUROC=auc(v[P], v[N]), AUROC_ci_low=np.percentile(reps[arm], 2.5), AUROC_ci_high=np.percentile(reps[arm], 97.5),
                         sens90_locked=k90 / 713, sens90_locked_ci=wilson(k90, 713), spec90_locked=f'{c90}/44', spec90_locked_value=c90 / 44,
                         sens95_locked=k95 / 713, spec95_locked=f'{c95}/44',
                         sens90_recomputed=s90r, sens90_recomputed_n=k90r, sens95_recomputed=s95r, sens95_recomputed_n=k95r,
                         median_targets_cancers=float(r.loc[y == 1, f'{cond}_n'].median()) if cond != 'none' else 0,
                         controls_with_any_target=float((r.loc[y == 0, f'{cond}_n'] > 0).mean()) if cond != 'none' else np.nan))
    if cond != 'none':
        d_rand = reps['removed'] - reps['matched random']; d_base = reps['removed'] - base_rep
        rows[-2].update(diff_vs_random=rows[-2]['AUROC'] - rows[-1]['AUROC'], diff_vs_random_ci=tuple(np.percentile(d_rand, [2.5, 97.5])),
                        diff_vs_random_p=float(2 * min((d_rand <= 0).mean(), (d_rand >= 0).mean())),
                        diff_vs_none=rows[-2]['AUROC'] - auc(r.base.to_numpy()[P], r.base.to_numpy()[N]), diff_vs_none_ci=tuple(np.percentile(d_base, [2.5, 97.5])))
out = pd.DataFrame(rows); out.to_csv(OUT / 'removal_full_reporting.csv', index=False)
pd.set_option('display.width', 250)
print(out[['condition', 'arm', 'AUROC', 'AUROC_ci_low', 'AUROC_ci_high', 'sens90_locked', 'spec90_locked', 'sens90_recomputed', 'sens95_recomputed', 'diff_vs_random', 'diff_vs_random_ci', 'diff_vs_random_p']].round(3).to_string())
