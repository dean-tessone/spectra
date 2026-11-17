"""Cluster-level stability of Figure 3 attribution across the three independently trained seeds (all 757 test
patients, saved per-seed AttnLRP values; no inference). Per patient and seed: share of positive logit relevance per
Leiden cluster (every model-input cell assigned). Reports (1) per-patient correlation of 23-cluster share vectors
between seeds; (2) cohort-level cluster emphasis in cancers (log2 attributed share over pool share, as Fig 3b)
per seed, and its correlation between seeds and with the ensemble."""
from pathlib import Path
import itertools, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
REPO = Path(__file__).resolve().parents[2]
R = REPO / 'runs/figure3_locked6_attribution_20261111'; OUT = REPO / 'analyses/attribution_sanity_20261111'
c = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id', 'label'] + [f'logit_relevance_{s}' for s in (17, 29, 43)] + ['logit_relevance'])
c['cluster'] = np.load(R / 'mass90_leiden/pool_cluster_all_cells.npy'); assert len(c) == len(np.load(R / 'mass90_leiden/pool_cluster_all_cells.npy'))
heads = {'17': 'logit_relevance_17', '29': 'logit_relevance_29', '43': 'logit_relevance_43', 'ensemble': 'logit_relevance'}
K = int(c.cluster.max()) + 1
share = {}
for h, col in heads.items():
    pos = c[col].clip(lower=0)
    s = pos.groupby([c.donor_id, c.cluster]).sum().unstack(fill_value=0).reindex(columns=range(K), fill_value=0)
    share[h] = s.div(s.sum(axis=1), axis=0).fillna(0)
lab = c.groupby('donor_id').label.first()
rows = []
for a, b in itertools.combinations(['17', '29', '43'], 2):
    r = [pearsonr(share[a].loc[d], share[b].loc[d]).statistic for d in share[a].index]
    rows.append(dict(pair=f'{a}-{b}', median_patient_pearson=float(np.nanmedian(r))))
pool = c.groupby([c.donor_id, c.cluster]).size().unstack(fill_value=0).reindex(columns=range(K), fill_value=0)
pool = pool.div(pool.sum(axis=1), axis=0)
ca = lab[lab == 1].index
emph = {h: np.log2((share[h].loc[ca].mean() + 1e-4) / (pool.loc[ca].mean() + 1e-4)) for h in heads}
E = pd.DataFrame(emph)
cohort = {f'{a}-{b}': float(spearmanr(E[a], E[b]).statistic) for a, b in itertools.combinations(['17', '29', '43'], 2)}
cohort.update({f'{s}-ensemble': float(spearmanr(E[s], E['ensemble']).statistic) for s in ['17', '29', '43']})
sign = float((np.sign(E[['17', '29', '43']]).nunique(axis=1) == 1).mean())
E.to_csv(OUT / 'cluster_emphasis_by_seed.csv')
res = dict(per_patient_cluster_share=rows, cohort_cluster_emphasis_spearman=cohort, fraction_clusters_same_emphasis_sign_all_seeds=sign)
json.dump(res, open(OUT / 'cluster_seed_stability.json', 'w'), indent=1); print(json.dumps(res, indent=1)); print(E.round(2))

# --- (added) Figure 3b emphasis recomputed exactly per seed: mass-90 cells per patient selected with that seed's
# probability relevance (policy.select_cells), per-patient cluster fractions, log2(attributed / pool) in cancers.
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location('pol', REPO / 'attribution/methods/policy.py'); pol = importlib.util.module_from_spec(spec); spec.loader.exec_module(pol)
c2 = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id', 'slide_id', 'source_row', 'relevance', 'relevance_17', 'relevance_29', 'relevance_43', 'mass90'])
c2['cluster'] = np.load(R / 'mass90_leiden/pool_cluster_all_cells.npy')
F3 = {}
for h in ['17', '29', '43', 'ensemble']:
    col = 'relevance' if h == 'ensemble' else f'relevance_{h}'
    sel = []
    for d, g in c2.groupby('donor_id', sort=False):
        t, _ = pol.select_cells(g[['slide_id', 'source_row', 'cluster']].assign(relevance=g[col].to_numpy()))
        sel.append(t.loc[t.mass90, ['cluster']].assign(donor_id=d))
    s = pd.concat(sel).groupby(['donor_id', 'cluster']).size().unstack(fill_value=0).reindex(index=pool.index, columns=range(K), fill_value=0)
    att = s.div(s.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    F3[h] = np.log2((att.loc[ca].mean() + 1e-6) / (pool.loc[ca].mean() + 1e-6))
F3 = pd.DataFrame(F3)
tab = pd.read_csv(REPO / 'analyses/figure3_locked6_attribution_20261111/leiden/cluster_table.csv', index_col=0)
num = pd.read_csv(REPO / 'analyses/figure3_locked6_attribution_20261111/leiden/cluster_display_numbers.csv').set_index('cluster').number
F3['figure3_reported'] = tab.emphasis_log2_cancer; F3['display_number'] = num
F3['same_sign_all_seeds'] = np.sign(F3[['17', '29', '43']]).nunique(axis=1) == 1
F3.to_csv(OUT / 'figure3_emphasis_by_seed.csv')
res2 = dict(ensemble_matches_reported_max_abs_diff=float((F3.ensemble - F3.figure3_reported).abs().max()),
            seed_pair_spearman={f'{a}-{b}': float(spearmanr(F3[a], F3[b]).statistic) for a, b in itertools.combinations(['17', '29', '43'], 2)},
            same_sign_fraction=float(F3.same_sign_all_seeds.mean()))
json.dump(dict(res, figure3_emphasis=res2), open(OUT / 'cluster_seed_stability.json', 'w'), indent=1)
print(json.dumps(res2, indent=1)); print(F3.sort_values('display_number').round(2).to_string())
