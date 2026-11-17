"""Contact sheets of 40x candidates per Leiden cluster for requested visual selection (review only)."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / 'imaging/cell_images'))
from cell_ext_display import rgb
OUT = REPO / 'analyses/figure3_locked6_attribution_20261111'; L = OUT / 'leiden'
SHEETS = Path(sys.argv[1]); SHEETS.mkdir(parents=True, exist_ok=True)
atlas = pd.read_parquet(L / 'atlas_mass90_private.parquet')
ok = pd.read_csv(OUT / '40x_extraction_status.csv'); ok = set(ok.cell_key[ok.ok])
c = atlas[atlas.cell_key.isin(ok)].copy()
c = c.sort_values(['cluster', 'label', 'relevance'], ascending=[True, False, False])
rows = []
for k, g in c.groupby('cluster'):
    g = g.head(24)
    fig, axs = plt.subplots(3, 8, figsize=(16, 6.4)); fig.suptitle(f'cluster {k}: {len(c[c.cluster==k])} 40x candidates')
    for ax in axs.flat: ax.axis('off')
    for i, r in enumerate(g.itertuples()):
        im = np.load(OUT / 'cache_40x' / f'{r.cell_key}.npz')['image'][100:301, 100:301]
        axs.flat[i].imshow(rgb(im)); axs.flat[i].set_title(f'{i} {"ca" if r.label else "ct"} {r.cancer_types[:6]}', fontsize=8)
        rows.append(dict(cluster=k, idx=i, cell_key=r.cell_key, label=r.label, cancer=r.cancer_types, relevance=r.relevance))
    fig.savefig(SHEETS / f'cluster_{k:02d}.png', dpi=60, bbox_inches='tight'); plt.close(fig)
pd.DataFrame(rows).to_csv(OUT / 'leiden/gallery_candidates_private.csv', index=False)
print(c.groupby('cluster').size().to_string())
