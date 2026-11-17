"""Checks behind the group-meeting reading of Figure 5 (post hoc, descriptive, no model calls).
1. Net change in the share of positive evidence on all CK+ clusters after adaptation, lung and breast, by group, with a
   patient bootstrap. 2. Abundance (share of model-input cells) of each Figure 3 cluster in the task populations (breast cancers,
   BRCA carriers, lung cancers, lung work-up controls) and the 44 primary reference controls."""
from pathlib import Path
import sys
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'adaptation/disease_specific'))
import analyze_attribution as AA  # noqa: E402
RUN = REPO / 'runs/specific_adaptation_20261111/attribution'
OUT = REPO / 'analyses/population_analyses'
F3 = REPO / 'runs/figure3_locked6_attribution_20261111'
NUM = pd.read_csv(REPO / 'analyses/figure3_locked6_attribution_20261111/leiden/cluster_display_numbers.csv', index_col=0).number
CKPOS = [5, 6, 8, 12, 13, 15]; CKNEG_TUMOUR = [5, 8]


def boot(v, B=4000, seed=17):
    v = np.asarray(v); rng = np.random.default_rng(seed); m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return np.percentile(m, [2.5, 97.5])


def main():
    shares, abund = [], []
    for task in ['lung', 'breast']:
        p = RUN / f'{task}_cells_clustered_private.parquet'
        if p.exists():
            c = pd.read_parquet(p)
        else:
            c = pd.read_parquet(RUN / f'{task}_cells_private.parquet').reset_index(drop=True); c['cluster'] = AA.assign(c); c.to_parquet(p)
        for h in ['source', 'probe', 'adapt']:
            c[f'{h}_pos'] = c[h].clip(lower=0)
        g = c.groupby('donor_id')
        tot = g[[f'{h}_pos' for h in ['source', 'probe', 'adapt']]].sum()
        for name, ks in [('all CK+ clusters', CKPOS), ('CK+ CD45/31- clusters 20, 21', CKNEG_TUMOUR), ('cluster 10', [2])]:
            sub = c[c.cluster.isin(ks)].groupby('donor_id')[[f'{h}_pos' for h in ['source', 'probe', 'adapt']]].sum().reindex(tot.index).fillna(0) / tot
            sub['label'] = g.label.first()
            for lab, s in sub.groupby('label'):
                for h in ['probe', 'adapt']:
                    dlt = (s[f'{h}_pos'] - s.source_pos) * 100; lo, hi = boot(dlt)
                    shares.append(dict(task=task, group='cancer' if lab else 'control', clusters=name, head=h, patients=len(s),
                                       source_share_pct=s.source_pos.mean() * 100, adapted_share_pct=s[f'{h}_pos'].mean() * 100,
                                       change_pp=dlt.mean(), ci_low=lo, ci_high=hi, patients_up=int((dlt > 0).sum())))
        a = pd.crosstab(c.donor_id, c.cluster, normalize='index'); a['label'] = g.label.first()
        for lab, s in a.groupby('label'):
            nm = {('lung', 1): 'Lung cancers', ('lung', 0): 'Lung work-up controls', ('breast', 1): 'Breast cancers', ('breast', 0): 'BRCA carriers'}[(task, lab)]
            abund.append(s.drop(columns='label').assign(population=nm))
    # reference controls from the Figure 3 run (same Leiden assignment)
    f3 = pd.read_parquet(F3 / 'all_cells.parquet', columns=['donor_id', 'label']); f3['cluster'] = np.load(F3 / 'mass90_leiden/pool_cluster_all_cells.npy')
    f3 = f3[f3.label == 0]; abund.append(pd.crosstab(f3.donor_id, f3.cluster, normalize='index').assign(population='Reference controls'))
    ab = pd.concat(abund).fillna(0); ab.to_csv(OUT / 'task_population_cluster_abundance_private.csv')
    rows = []
    for pop, s in ab.groupby('population'):
        for k in range(23):
            lo, hi = boot(s[k] * 100); rows.append(dict(population=pop, patients=len(s), cluster_display=NUM[k], mean_pct=s[k].mean() * 100, ci_low=lo, ci_high=hi, median_pct=s[k].median() * 100))
    ab_s = pd.DataFrame(rows); ab_s.to_csv(OUT / 'task_population_cluster_abundance.csv', index=False)
    sh = pd.DataFrame(shares); sh.to_csv(OUT / 'adaptation_CK_share_change.csv', index=False)
    print(sh.round(2).to_string())
    print(ab_s[ab_s.cluster_display.isin([10, 16, 11, 20, 21, 19, 18])].pivot(index='cluster_display', columns='population', values='mean_pct').round(2).to_string())


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


def cluster10_markers():
    """Marker means of cluster-10 cells by population and their share of pan-cancer positive evidence per cell."""
    sys.path.insert(0, str(REPO / 'model/pipeline'))
    from common import read
    receipt = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
    cols = ['area', 'eccentricity', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']; parts = []
    for task in ['lung', 'breast']:
        c = pd.read_parquet(RUN / f'{task}_cells_clustered_private.parquet'); c = c[c.cluster == 2]
        for sid, g in c.groupby('slide_id'):
            src = pd.read_parquet(receipt[sid]['source'], columns=cols)
            parts.append(pd.concat([g.reset_index(drop=True), src.iloc[g.source_row.to_numpy()].reset_index(drop=True)], axis=1).assign(task=task))
    t = pd.concat(parts)
    t['population'] = np.select([(t.task == 'lung') & (t.label == 1), (t.task == 'lung') & (t.label == 0), (t.task == 'breast') & (t.label == 1)],
                                ['Lung cancers', 'Lung work-up controls', 'Breast cancers'], 'BRCA carriers')
    f3 = pd.read_parquet(F3 / 'all_cells.parquet', columns=['donor_id', 'label'] + cols); f3['cluster'] = np.load(F3 / 'mass90_leiden/pool_cluster_all_cells.npy')
    f3 = f3[(f3.label == 0) & (f3.cluster == 2)].assign(population='Reference controls')
    allc = pd.concat([t[['donor_id', 'population'] + cols], f3[['donor_id', 'population'] + cols]])
    per = allc.groupby(['population', 'donor_id'])[cols].median().groupby('population').median()
    per['patients'] = allc.groupby('population').donor_id.nunique(); per['cells'] = allc.groupby('population').size()
    per.to_csv(OUT / 'cluster10_marker_medians_by_population.csv'); print(per.round(1).to_string())


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'markers':
    cluster10_markers()
