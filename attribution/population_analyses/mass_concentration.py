"""How concentrated is SPECTRA's positive attribution? Cells needed for 50/75/90% of positive logit relevance per patient,
and whether Figure 3 cluster emphasis depends on the 90% mass cut-off (post hoc, descriptive, no model calls)."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
R = REPO / 'runs/figure3_locked6_attribution_20261111'
OUT = REPO / 'analyses/population_analyses'; OUT.mkdir(parents=True, exist_ok=True)
NUM = pd.read_csv(REPO / 'analyses/figure3_locked6_attribution_20261111/leiden/cluster_display_numbers.csv', index_col=0).number


def main():
    c = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id', 'label', 'relevance', 'positive_rank', 'cumulative_positive_fraction', 'mass90', 'top50'])
    c['k'] = np.load(R / 'mass90_leiden/pool_cluster_all_cells.npy')
    rows = []
    for d, g in c.groupby('donor_id'):
        pos = np.sort(g.relevance.clip(lower=0).to_numpy())[::-1]; cum = np.cumsum(pos) / pos.sum()
        r = dict(donor_id=d, label=int(g.label.iloc[0]), cells=len(g), positive_cells=int((g.relevance > 0).sum()),
                 positive_share_of_abs=float(pos.sum() / g.relevance.abs().sum()), mass90_flag_cells=int(g.mass90.sum()))
        for q in (.5, .75, .9):
            r[f'cells_mass{int(q*100)}'] = int(np.searchsorted(cum, q) + 1)
        r['top10_mass'] = float(cum[min(9, len(cum) - 1)]); r['top50_mass'] = float(cum[min(49, len(cum) - 1)])
        rows.append(r)
    p = pd.DataFrame(rows)
    for q in (50, 75, 90):
        p[f'frac_mass{q}'] = p[f'cells_mass{q}'] / p.cells
    p.to_csv(OUT / 'mass_concentration_per_patient_private.csv', index=False)
    summ = p.groupby('label')[['cells', 'positive_cells', 'cells_mass50', 'cells_mass75', 'cells_mass90', 'frac_mass50', 'frac_mass75', 'frac_mass90',
                               'top10_mass', 'top50_mass', 'positive_share_of_abs']].describe(percentiles=[.25, .5, .75]).T
    summ.to_csv(OUT / 'mass_concentration_summary.csv')
    # emphasis of each cluster at different cut-offs (cancers), relative to pool share
    pool = pd.crosstab(c.donor_id, c.k, normalize='index'); lab = c.groupby('donor_id').label.first().loc[pool.index]
    c['cum'] = c.cumulative_positive_fraction
    em = {}
    for name, sel in {'top10': c.positive_rank.between(1, 10), 'top50': c.top50, 'mass50': (c.relevance > 0) & (c.cum <= .5 + 1e-12) | (c.positive_rank == 1),
                      'mass75': (c.relevance > 0) & (c.cum <= .75 + 1e-12) | (c.positive_rank == 1), 'mass90': c.mass90}.items():
        a = pd.crosstab(c.donor_id[sel], c.k[sel], normalize='index').reindex(index=pool.index, columns=pool.columns).fillna(0)
        em[name] = np.log2((a[lab == 1].mean() + 1e-6) / (pool[lab == 1].mean() + 1e-6))
    em = pd.DataFrame(em); em.index = em.index.map(NUM); em = em.sort_index(); em.index.name = 'cluster_display'
    em.to_csv(OUT / 'emphasis_by_cutoff.csv')
    rho = {k: spearmanr(em[k], em['mass90']).statistic for k in em}
    print(summ.round(3).to_string()); print(em.round(2).to_string()); print('spearman vs mass90', {k: round(v, 3) for k, v in rho.items()})
    pd.Series(rho).to_csv(OUT / 'emphasis_by_cutoff_spearman.csv')


if __name__ == '__main__':
    main()
