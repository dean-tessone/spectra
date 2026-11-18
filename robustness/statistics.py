"""Revision step 5: statistics requested by the reviewers (inference-free, 2026-11-17).

  5.1 Fixed-specificity sensitivity with the cut-off re-estimated inside each bootstrap replicate (controls and cancers resampled),
      so the interval includes threshold uncertainty. Cut-off rule as in the sealed report: highest score among the controls that
      must stay below it (40 of 44 at 90%, 42 of 44 at 95%, scaled to the resampled control count).
  5.2 Single-seed test AUROCs of the three sealed models, and seed agreement.
  5.3 Wilson 95% CIs for every molecular proportion reported in the paper.
"""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'figures/code')); import sx  # noqa: E402
OUT = REPO / 'analyses/revision_20261117/statistics'; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261117); B = 4000


def wilson(k, n, z=norm.ppf(.975)):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def auc(pos, neg):
    allv = np.concatenate([pos, neg]); rk = pd.Series(allv).rank().to_numpy()
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


res = {}
p = sx.primary_predictions(); pos, neg = p.loc[p.label == 1, 'probability'].to_numpy(), p.loc[p.label == 0, 'probability'].to_numpy()


def sens_at(pos, neg, spec):
    k = int(np.ceil(spec * len(neg)))  # controls that must fall below the cut-off
    t = np.sort(neg)[k - 1]
    return float((pos > t).mean())


for spec in (.9, .95):
    obs = sens_at(pos, neg, spec)
    reps = np.array([sens_at(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))], spec) for _ in range(B)])
    k = int(round(obs * len(pos)))
    res[f'sens_at_{int(spec*100)}spec'] = dict(estimate=obs, cases_called=k, wilson_fixed_threshold=wilson(k, len(pos)),
                                                 bootstrap_threshold_reestimated=np.percentile(reps, [2.5, 97.5]).tolist())

# 5.2 single seeds
seeds = {}
for s in (17, 29, 43):
    g = pd.read_csv(REPO / f'runs/reference_control_test_20261110_v1a/private_seed_{s}_group_predictions.csv', dtype={'donor_id': str})
    g = g[g.donor_id.isin(p.donor_id)]
    if 'primary_evaluation' in g:
        g = g[g.primary_evaluation.astype(bool)]
    g = g.drop_duplicates('donor_id').merge(p[['donor_id', 'label']], on='donor_id', suffixes=('_g', ''))
    seeds[s] = g.set_index('donor_id').probability
    a = auc(g.probability[g.label == 1].to_numpy(), g.probability[g.label == 0].to_numpy())
    res[f'seed_{s}_AUROC'] = a; res[f'seed_{s}_patients'] = len(g)
S = pd.DataFrame(seeds).dropna()
res['seed_pairwise_spearman'] = S.corr('spearman').round(3).to_dict()
lab = p.set_index('donor_id').label.reindex(S.index)
for grp, v in [('cancers', 1), ('controls', 0)]:
    sub = S[lab == v]
    res[f'seed_agreement_{grp}'] = dict(all_three_above_half=float((sub > .5).all(1).mean()), all_three_below_half=float((sub < .5).all(1).mean()))
res['ensemble_AUROC'] = auc(pos, neg)

# 5.3 molecular proportions (counts as reported in Results)
mol = {'IMC cluster 21 EpCAM+': (6, 13), 'IMC cluster 10 EpCAM+': (2, 16), 'IMC cluster 10 CD31+': (7, 18), 'IMC cluster 10 myeloid-marker+': (7, 18),
       'WGS cluster 20 altered': (28, 31), 'WGS cluster 21 altered': (19, 26), 'WGS cluster 17 altered': (5, 6), 'WGS cluster 10 altered': (7, 27),
       'WGS cluster 16 altered': (2, 18)}
res['molecular_wilson'] = {k: dict(k=a, n=b, prop=a / b, ci=wilson(a, b)) for k, (a, b) in mol.items()}
json.dump(res, open(OUT / 'summary.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
print(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o)))
