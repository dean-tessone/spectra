"""Revision step 2: what distinguishes cluster 10 (and contrast clusters 20, 21) between groups (2026-11-17). Inference-free.

Cluster numbers are the paper's display numbers (cluster_display_numbers.csv: display 10 = internal 2, 20 = internal 8,
21 = internal 5). Cells: mass-90 attributed cells of the locked model.
  * Primary test (757): the Figure 3 atlas (mass-90 cells, Leiden clusters).
  * Lung-workup and BRCA-carrier controls: saved per-cell source-model relevance from the adaptation attribution run, cells
    assigned to the atlas clusters by 15-NN; mass-90 recomputed per patient from the source head; features from source parquet.
Per patient, the mean of each feature over that patient's mass-90 cells in the cluster; groups compared at patient level
(medians, AUROC with stratified bootstrap CI). Also: cluster-10 share of positive relevance per group with bootstrap CI, and
cluster-10 recovery across Leiden seeds.
"""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
L = REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
RUNF = REPO / 'runs/figure3_locked6_attribution_20261111'
ADA = REPO / 'runs/specific_adaptation_20261111/attribution'
OUT = REPO / 'analyses/revision_20261117/cluster10'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import local  # noqa: E402
rng = np.random.default_rng(20261117); B = 4000
disp = pd.read_csv(L / 'cluster_display_numbers.csv').set_index('cluster').number
INT = {v: k for k, v in disp.items()}
FEATS = {'area_um2': 'Cell area (µm²)', 'eccentricity': 'Eccentricity', 'DAPI_mean': 'DAPI', 'TRITC_mean': 'Pan-cytokeratin',
         'FITC_mean': 'Vimentin', 'CY5_mean': 'CD45/CD31'}
PX = 0.59  # µm per pixel at 10x


def auc(pos, neg):
    allv = np.concatenate([pos, neg]); rk = pd.Series(allv).rank().to_numpy()
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_auc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3: return auc(a, b) if len(a) and len(b) else np.nan, np.nan, np.nan
    I, J = rng.integers(0, len(a), (B, len(a))), rng.integers(0, len(b), (B, len(b)))
    r = np.array([auc(a[I[k]], b[J[k]]) for k in range(B)]); return auc(a, b), *np.percentile(r, [2.5, 97.5])


def boot_mean(v):
    v = np.asarray(v, float); I = rng.integers(0, len(v), (B, len(v))); return v.mean(), *np.percentile(v[I].mean(1), [2.5, 97.5])


# ---------------- primary test atlas
at = pd.read_parquet(L / 'atlas_mass90_private.parquet', columns=['donor_id', 'label', 'cancer_types', 'study', 'cluster', 'area', 'eccentricity',
                                                                     'DAPI_mean', 'TRITC_mean', 'FITC_mean', 'CY5_mean', 'relevance'])
at['area_um2'] = at.area * PX * PX; at['group'] = np.where(at.label == 1, 'Cancer', 'Reference control')

# ---------------- diagnostic controls: mass-90 by source-head relevance, cluster-assigned, features from source parquet
inv = pd.read_csv(REPO / 'runs/paper_regeneration_20261109_v1a/cohort/local_cache_inventory.csv', dtype=str, usecols=['slide_id', 'source'])
src = dict(zip(inv.slide_id, inv.source))
dx = []
for task, grp in [('lung', 'Lung work-up control'), ('breast', 'BRCA-carrier control')]:
    c = pd.read_parquet(ADA / f'{task}_cells_clustered_private.parquet', columns=['donor_id', 'slide_id', 'source_row', 'source', 'label', 'cluster'])
    c = c[c.label == 0].copy(); c['pos'] = c.source.clip(lower=0)
    c = c.sort_values(['donor_id', 'pos'], ascending=[True, False])
    c['cum'] = c.groupby('donor_id').pos.cumsum() / c.groupby('donor_id').pos.transform('sum')
    c = c[(c.cum - c.pos / c.groupby('donor_id').pos.transform('sum')) < .9]  # cells needed to reach 90% of positive relevance
    parts = []
    for s, g in c.groupby('slide_id'):
        f = pd.read_parquet(local(src[s]), columns=['area', 'eccentricity', 'DAPI_mean', 'TRITC_mean', 'FITC_mean', 'CY5_mean'])
        parts.append(pd.concat([g.reset_index(drop=True), f.iloc[g.source_row.to_numpy()].reset_index(drop=True)], axis=1))
    c = pd.concat(parts); c['area_um2'] = c.area * PX * PX; c['group'] = grp; c['relevance'] = c.source
    dx.append(c)
dx = pd.concat(dx)
cells = pd.concat([at[['donor_id', 'group', 'cluster', 'relevance'] + list(FEATS)], dx[['donor_id', 'group', 'cluster', 'relevance'] + list(FEATS)]])

# ---------------- per-patient feature means within clusters 10/20/21
rows, comp = [], []
for num in (10, 20, 21):
    k = INT[num]; g = cells[cells.cluster == k]
    pp = g.groupby(['donor_id', 'group'])[list(FEATS)].mean().reset_index()
    pp['cells'] = g.groupby(['donor_id', 'group']).size().to_numpy()
    pp['cluster'] = num; rows.append(pp)
    for f, nm in FEATS.items():
        for a_, b_ in [('Cancer', 'Reference control'), ('BRCA-carrier control', 'Reference control'), ('Lung work-up control', 'Reference control'),
                       ('Cancer', 'BRCA-carrier control'), ('Cancer', 'Lung work-up control')]:
            x, y = pp.loc[pp.group == a_, f], pp.loc[pp.group == b_, f]
            a, lo, hi = boot_auc(x, y)
            comp.append(dict(cluster=num, feature=nm, group_a=a_, group_b=b_, n_a=len(x), n_b=len(y), median_a=x.median(), median_b=y.median(),
                             AUROC_a_vs_b=a, ci_low=lo, ci_high=hi))
pp = pd.concat(rows); pp.to_csv(OUT / 'per_patient_cluster_features_private.csv', index=False)
comp = pd.DataFrame(comp); comp.to_csv(OUT / 'cluster_feature_comparisons.csv', index=False)
med = pp.groupby(['cluster', 'group'])[list(FEATS) + ['cells']].median().round(2); med['patients'] = pp.groupby(['cluster', 'group']).size()
med.to_csv(OUT / 'cluster_feature_medians.csv')

# ---------------- cluster-10 share of positive relevance per group (patient mean, bootstrap CI)
share = []
pos = cells.assign(p=cells.relevance.clip(lower=0))
tot = pos.groupby('donor_id').p.sum()
s10 = pos[pos.cluster == INT[10]].groupby('donor_id').p.sum().reindex(tot.index, fill_value=0) / tot
grpof = pos.groupby('donor_id').group.first()
for grp in ['Cancer', 'Reference control', 'Lung work-up control', 'BRCA-carrier control']:
    v = s10[grpof == grp].dropna(); m, lo, hi = boot_mean(v)
    share.append(dict(group=grp, patients=len(v), mean_share=m, ci_low=lo, ci_high=hi, median_share=v.median()))
share = pd.DataFrame(share)
# difference BRCA minus reference, BRCA minus cancer (bootstrap)
def diff(a, b):
    a, b = s10[grpof == a].dropna().to_numpy(), s10[grpof == b].dropna().to_numpy()
    I, J = rng.integers(0, len(a), (B, len(a))), rng.integers(0, len(b), (B, len(b)))
    d = a[I].mean(1) - b[J].mean(1); return a.mean() - b.mean(), *np.percentile(d, [2.5, 97.5])
extra = {f'{a} minus {b}': diff(a, b) for a, b in [('BRCA-carrier control', 'Reference control'), ('BRCA-carrier control', 'Cancer'), ('Lung work-up control', 'Reference control')]}
share.to_csv(OUT / 'cluster10_relevance_share.csv', index=False)

# ---------------- cluster-10 recovery across Leiden seeds (best-match Jaccard)
lab = {s: np.load(RUNF / f'mass90_leiden/leiden_r1_seed{s}.npy') for s in (17, 29, 43)}
prim = np.load(RUNF / 'mass90_leiden/cluster_primary.npy')
rec = {}
for s, l in lab.items():
    a = prim == INT[10]; best = 0; bk = None
    for k in np.unique(l):
        b = l == k; j = (a & b).sum() / (a | b).sum()
        if j > best: best, bk = j, int(k)
    rec[f'seed_{s}'] = dict(best_jaccard=float(best), best_cluster=bk, share_of_cluster10_in_best=float((a & (l == bk)).sum() / a.sum()))
# per model seed: share of each seed's positive relevance in cluster 10 (primary test atlas)
seedshare = {}
at2 = pd.read_parquet(L / 'atlas_mass90_private.parquet', columns=['donor_id', 'cluster', 'relevance_17', 'relevance_29', 'relevance_43'])
for s in (17, 29, 43):
    p_ = at2[f'relevance_{s}'].clip(lower=0); seedshare[f'seed_{s}'] = float(p_[at2.cluster == INT[10]].sum() / p_.sum())
json.dump(dict(relevance_share_differences=extra, leiden_seed_recovery=rec, cluster10_share_by_model_seed=seedshare,
               display_to_internal={10: INT[10], 20: INT[20], 21: INT[21]}), open(OUT / 'summary.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 250)
print(med.to_string()); print(share.round(3).to_string()); print(json.dumps(extra, default=float, indent=1)); print(json.dumps(rec, indent=1)); print(seedshare)
print(comp[comp.cluster == 10].round(3).to_string())
