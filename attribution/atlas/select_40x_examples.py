"""Deterministic 40x example selection for Fig 4a (CK-negative detected cancers) and Fig 5e
(cells most down-/up-weighted by adaptation in test cancer patients), restricted to cached eligible 40x cells."""
from pathlib import Path
import pandas as pd, numpy as np
REPO = Path(__file__).resolve().parents[2]
A = REPO / 'analyses/figure3_locked6_attribution_20261111'
ok = pd.read_csv(A / '40x_extraction_status.csv'); ok = set(ok.cell_key[ok.ok])
c = pd.read_parquet(REPO / 'runs/figure3_locked6_attribution_20261111/all_cells.parquet', columns=['donor_id', 'cancer_types', 'slide_id', 'source_row', 'relevance', 'positive_rank', 'TRITC_mean', 'CY5_mean', 'FITC_mean', 'DAPI_mean'])
c['cell_key'] = [f'{s}_r{int(r):05d}' for s, r in zip(c.slide_id, c.source_row)]
rows = []
for d in pd.read_csv(A / 'fig4_ckneg_examples_private.csv', dtype=str).donor_id:   # four detected CK-negative cancers (private: patient IDs)
    g = c[(c.donor_id == d) & c.cell_key.isin(ok)].nsmallest(7, 'positive_rank')
    rows.append(g.assign(panel='fig4_ckneg_40x', example=d, rank=range(1, len(g) + 1)))
for task in ['lung', 'breast']:
    t = pd.read_parquet(REPO / f'runs/specific_adaptation_20261111/attribution/{task}_cells_private.parquet')
    t = t[t.label == 1].copy()
    for h in ['source', 'probe']:
        t[h + '_rk'] = t.groupby('donor_id')[h].rank(ascending=False, pct=True)
    t['cell_key'] = [f'{s}_r{int(r):05d}' for s, r in zip(t.slide_id, t.source_row)]
    t = t[t.cell_key.isin(ok)]
    t['shift'] = t.source_rk - t.probe_rk          # >0: moved up after adaptation
    lost = t[t.source_rk <= .05].nsmallest(6, 'shift').drop_duplicates('donor_id')
    gained = t[t.probe_rk <= .05].nlargest(6, 'shift').drop_duplicates('donor_id')
    for nm, g in [('lost', lost), ('gained', gained)]:
        rows.append(g.assign(panel=f'fig5_{task}_{nm}', example=task, rank=range(1, len(g) + 1))[['donor_id', 'slide_id', 'source_row', 'cell_key', 'panel', 'example', 'rank', 'source_rk', 'probe_rk']])
sel = pd.concat(rows, ignore_index=True)
sel.to_csv(A / 'examples_40x_private.csv', index=False)
print(sel.groupby('panel').size())
