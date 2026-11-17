"""Fig 5c/5d thumbnails: each displayed cluster is illustrated by a cell from THAT task's test
patients (cancers or diagnostic controls), not the Figure 3 gallery cell.

Candidates: each test patient's top-30 cells by pan-cancer or adapted-probe relevance, assigned to Fig 3
clusters (15-NN, analyze_attribution.assign). Per displayed cluster (six largest mean share shifts), candidates
are ordered by best within-patient rank under either head; up to 8 cells (at most 2 per patient) with a cached
40x re-image, else up to 8 10x cells extracted with the audited review extractor. Final cells are chosen
visually for clarity from these candidates and recorded in fig5cd_thumbs_private.csv."""
from pathlib import Path
import importlib.util, json, sys
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read  # noqa: E402
spec = importlib.util.spec_from_file_location('aa', REPO / 'adaptation/disease_specific/analyze_attribution.py'); AA = importlib.util.module_from_spec(spec); spec.loader.exec_module(AA)
OUT = REPO / 'analyses/specific_adaptation_20261111/attribution'
TENX = REPO / 'analyses/specific_adaptation_20261111/tenx'
A3 = REPO / 'analyses/figure3_locked6_attribution_20261111'
ok = pd.read_csv(A3 / '40x_extraction_status.csv'); ok = set(ok.cell_key[ok.ok])
receipt = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']

rows = []
for task in ['lung', 'breast']:
    sh = pd.read_csv(OUT / f'{task}_cluster_share_shift.csv')
    both = sh.pivot_table(index='cluster', columns='label', values='delta_probe').fillna(0).mean(axis=1)
    shown = list(both.abs().sort_values(ascending=False).index[:6])
    t = pd.read_parquet(REPO / f'runs/specific_adaptation_20261111/attribution/{task}_cells_private.parquet')
    for h in ['source', 'probe']:
        t[h + '_rk'] = t.groupby('donor_id')[h].rank(ascending=False, pct=True)
        t[h + '_r'] = t.groupby('donor_id')[h].rank(ascending=False)
    t = t[(t.source_r <= 30) | (t.probe_r <= 30)].reset_index(drop=True)
    t['cluster'] = AA.assign(t)
    t['cell_key'] = [f'{s}_r{int(r):05d}' for s, r in zip(t.slide_id, t.source_row)]
    t['best_rk'] = t[['source_rk', 'probe_rk']].min(axis=1)
    t['has40'] = t.cell_key.isin(ok)
    for k in shown:
        g = t[t.cluster == k].sort_values('best_rk')
        g = g[g.groupby('donor_id').cumcount() < 2]
        g40 = g[g.has40].head(8)
        pick = g40 if len(g40) >= 3 else pd.concat([g40, g[~g.has40].head(8)])
        rows.append(pick.assign(task=task, delta=float(both[k])))
c = pd.concat(rows, ignore_index=True)
# 10x extraction for non-40x candidates
need = c[~c.has40].drop_duplicates('cell_key')
meta = []
for sid, g in need.groupby('slide_id'):
    src = pd.read_parquet(receipt[sid]['source'], columns=['frame_id', 'cell_id', 'x', 'y', 'area', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']).iloc[g.source_row.to_numpy()]
    meta.append(pd.concat([g[['cell_key', 'slide_id', 'source_row']].reset_index(drop=True), src.reset_index(drop=True)], axis=1))
meta = pd.concat(meta, ignore_index=True)
spec = importlib.util.spec_from_file_location('review_build', REPO / 'imaging/cell_images/build.py')
hb = importlib.util.module_from_spec(spec); spec.loader.exec_module(hb); hb.HERE = TENX
recs = []
for sid, g in meta.groupby('slide_id'):
    try: recs.append(hb.extract_slide((sid, g)))
    except Exception as e: recs.append(dict(slide_id=sid, available_keys=[], error=repr(e)))
TENX.mkdir(parents=True, exist_ok=True)
json.dump(recs, open(TENX / 'provenance.json', 'w'), indent=1)
got = {k for r in recs for k in r.get('available_keys', [])}
c['has10'] = c.cell_key.isin(got)
c = c.merge(meta[['cell_key', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']], on='cell_key', how='left')
c.drop(columns=['source_r', 'probe_r']).to_csv(OUT / 'fig5cd_thumb_candidates_private.csv', index=False)
print(c.groupby(['task', 'cluster']).agg(n=('cell_key', 'size'), n40=('has40', 'sum'), n10=('has10', 'sum'), ctrl=('label', lambda s: int((s == 0).sum()))))
