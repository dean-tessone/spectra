"""Extract exemplar cell crops with the audited review pipeline (read-only raw tiles)."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.util, json, sys
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'analyses/figure3_locked6_attribution_20261111'
spec = importlib.util.spec_from_file_location('review_build', REPO / 'imaging/cell_images/build.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
h.HERE = OUT  # cache under this analysis folder
ex = pd.read_csv(OUT / 'exemplar_cells_private.csv', dtype={'slide_id': str})
ex = ex.drop_duplicates('cell_key')
cols = ['cell_key', 'slide_id', 'frame_id', 'cell_id', 'source_row', 'x', 'y', 'area', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']
tasks = list(ex[cols].groupby('slide_id', sort=True))
recs = []
def safe(t):
    try:
        return h.extract_slide(t)
    except Exception as e:  # unavailable/ambiguous acquisition: record, never substitute
        return dict(slide_id=t[0], available_keys=[], missing=[dict(cell_key=k, reason=repr(e)) for k in t[1].cell_key], error=repr(e))
with ThreadPoolExecutor(max_workers=3) as pool:
    for r in pool.map(safe, tasks):
        recs.append(r); print(len(recs), '/', len(tasks), r['slide_id'], len(r['available_keys']), flush=True)
json.dump(dict(slides=recs, source_writes=False), open(OUT / 'image_provenance.json', 'w'), indent=1)
