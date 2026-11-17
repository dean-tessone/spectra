"""10x candidates (audited review extractor) for Leiden clusters lacking any 40x re-image."""
from pathlib import Path
import importlib.util, sys, json
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'analyses/figure3_locked6_attribution_20261111'; L = OUT / 'leiden'
spec = importlib.util.spec_from_file_location('review_build', REPO / 'imaging/cell_images/build.py')
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h); h.HERE = OUT / 'tenx'
a = pd.read_parquet(L / 'atlas_mass90_private.parquet')
ks = [int(x) for x in sys.argv[1:]]
c = a[a.cluster.isin(ks) & a.called_at_90spec & a.label.eq(1)].sort_values(['cluster', 'relevance'], ascending=[True, False]).groupby('cluster').head(16)
cols = ['cell_key', 'slide_id', 'frame_id', 'cell_id', 'source_row', 'x', 'y', 'area', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']
c[cols + ['cluster']].to_csv(L / 'tenx_candidates_private.csv', index=False)
recs = []
for sid, g in c[cols].groupby('slide_id'):
    try: recs.append(h.extract_slide((sid, g)))
    except Exception as e: recs.append(dict(slide_id=sid, available_keys=[], error=repr(e)))
json.dump(recs, open(L / 'tenx_provenance.json', 'w'), indent=1)
print(sum(len(r['available_keys']) for r in recs), 'of', len(c))
