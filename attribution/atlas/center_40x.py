"""Auto-centre 40x display crops on the matched cell: segment the 401 px context (sum of channels,
Gaussian smoothing, Otsu threshold, connected components), pick the component whose centroid is
best matching the attributed cell's 10x 4-channel profile (cosine of centred log means) minus 0.004/px distance, within 120 px of the context centre (the matched event lies within +/-10 px at 10x = +/-40 px at 40x),
centre on its intensity-weighted centroid; half-width = max(52, 1.45 x equivalent radius).
Writes cx/cy/half back into the selection tables; manual overrides allowed afterwards."""
from pathlib import Path
import sys
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
A = Path(__file__).resolve().parents[2] / 'analyses/figure3_locked6_attribution_20261111'
atlas = pd.read_parquet(A / 'leiden/atlas_mass90_private.parquet', columns=['cell_key', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']).set_index('cell_key')
def centre(key):
    ref = np.log1p(atlas.loc[key, ['DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']].to_numpy(float)); ref = ref - ref.mean()
    im = np.load(A / 'cache_40x' / f'{key}.npz')['image'].astype(np.float32)
    s = ndi.gaussian_filter(im.sum(-1), 3); t = threshold_otsu(s); lab, n = ndi.label(s > t)
    best = None
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 300: continue
        cy, cx = ndi.center_of_mass(s * m); d = np.hypot(cy - 200, cx - 200)
        if d > 120: continue
        v = np.log1p(im[m].mean(0)); v = v - v.mean()
        sim = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9))
        score = sim - .004 * d
        if best is None or score > best[4]: best = (d, cy, cx, m.sum(), score)
    if best is None: return 200, 200, 64, np.nan
    d, cy, cx, area, _ = best; r = np.sqrt(area / np.pi); half = int(np.clip(1.45 * r, 52, 110))
    cy = int(np.clip(round(cy), half, 400 - half)); cx = int(np.clip(round(cx), half, 400 - half))
    return cx, cy, half, d
for f in [A / 'leiden/gallery_selection.csv', A / 'fig1_attribution_cells.csv']:
    t = pd.read_csv(f)
    for i, r in t.iterrows():
        if getattr(r, 'source', '40x') != '40x': continue
        cx, cy, h, d = centre(r.cell_key); t.loc[i, ['cx', 'cy', 'half', 'centre_distance_px']] = [cx, cy, h, d]
    t.to_csv(f, index=False); print(f.name); print(t.drop(columns=[c for c in ['cell_key'] if c in t]).to_string())
