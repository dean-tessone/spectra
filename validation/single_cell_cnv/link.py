"""Link copy-number calls (caller.py) to SPECTRA attribution on held-out test patients (Figure 3 run, locked
ensemble, inference only). A sequenced cell matches a SPECTRA candidate-pool cell on the same slide and frame within
10 px in x and y (the original 40x-match rule). Groups: top-50 attributed, other mass-90 (carry 90% of positive
attribution), other model-input cells, and sequenced cells that never entered the model (failed a filter). Exploratory,
first pass: picked cells are a biased sample (chosen at the microscope as candidate tumour cells)."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / 'model/pipeline'))
import caller as K  # noqa: E402
from common import read  # noqa: E402
F3 = REPO / 'runs/figure3_locked6_attribution_20261111'; OUT = REPO / 'analyses/cnv_attribution_20261112'; OUT.mkdir(parents=True, exist_ok=True)


def main():
    C = K.call_all()
    a = pd.read_parquet(F3 / 'all_cells.parquet', columns=['donor_id', 'slide_id', 'source_row', 'frame_id', 'x', 'y', 'relevance', 'positive_rank', 'top50', 'mass90', 'label', 'cancer_types'])
    a['cluster'] = np.load(F3 / 'mass90_leiden/pool_cluster_all_cells.npy')
    num = pd.read_csv(REPO / 'analyses/figure3_locked6_attribution_20261111/leiden/cluster_display_numbers.csv').set_index('cluster').number
    t = C[C.slide_id.isin(a.slide_id.unique()) & C.frame_id.notna()].copy()
    m = t.merge(a, on=['slide_id', 'frame_id'], suffixes=('_seq', ''))
    m = m[(m.x_seq - m.x).abs().le(10) & (m.y_seq - m.y).abs().le(10)]
    m = m.assign(d=np.hypot(m.x_seq - m.x, m.y_seq - m.y)).sort_values('d').drop_duplicates('cell_name')
    t = t.merge(m[['cell_name', 'donor_id', 'source_row', 'relevance', 'positive_rank', 'top50', 'mass90', 'label', 'cluster']], on='cell_name', how='left')
    donor = a.drop_duplicates('slide_id').set_index('slide_id')
    t['donor_id'] = t.slide_id.map(donor.donor_id); t['patient_label'] = t.slide_id.map(donor.label)
    t['group'] = np.select([t.top50 == True, t.mass90 == True, t.relevance.notna()], ['top-50 attributed', 'other mass-90', 'other model input'], 'not a model input')
    t['display_cluster'] = t.cluster.map(num)
    t.to_csv(OUT / 'test_sequenced_cells_private.csv', index=False)
    C.drop(columns=['patient_id']).to_csv(OUT / 'cnv_calls_all_cells_private.csv', index=False)
    order = ['top-50 attributed', 'other mass-90', 'other model input', 'not a model input']
    g = t.groupby(['patient_label', 'group']).agg(cells=('cell_name', 'size'), patients=('donor_id', 'nunique'), altered=('altered', 'sum'),
                                                  median_altered_fraction=('altered_fraction', 'median')).reset_index()
    g['altered_pct'] = (100 * g.altered / g.cells).round(1)
    print(g.to_string(index=False))
    hc = t[t.mass90 == True].groupby('display_cluster').agg(cells=('cell_name', 'size'), patients=('donor_id', 'nunique'), altered=('altered', 'sum')).reset_index()
    hc['altered_pct'] = (100 * hc.altered / hc.cells).round(0)
    print(hc.sort_values('cells', ascending=False).to_string(index=False))
    g.to_csv(OUT / 'altered_by_attribution_group.csv', index=False); hc.to_csv(OUT / 'altered_by_cluster_mass90.csv', index=False)


if __name__ == '__main__':
    main()
