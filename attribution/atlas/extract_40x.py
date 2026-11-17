"""Read-only extraction of eligible 40x re-images for mass-90 attributed cells, using the original
Figure 3 display processing: per-channel native-bit-range -> uint16 once, 75 px full-frame top-hat,
180-degree rotation, central 401 px context. Cached as compressed npz (display arrays only)."""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, sys
import numpy as np, pandas as pd
from PIL import Image
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'imaging/cell_images'))
from cell_ext_display import preprocess_frame
OUT = REPO / 'analyses/figure3_locked6_attribution_20261111'; CACHE = OUT / 'cache_40x'; CACHE.mkdir(exist_ok=True)
CH = ['DAPI', 'TRITC', 'CY5', 'FITC']
r = pd.read_csv(OUT / '40x_matching_audit_private.csv', dtype={'slide_id': str}); r = r[r.eligible]
def one(row):
    key = f'{row.slide_id}_r{int(row.source_row):05d}'; t = CACHE / f'{key}.npz'
    if t.exists(): return key, True
    try:
        folder = Path(os.environ.get('SPECTRA_40X_ROOT', 'raw/40x')) / row.slide_id / 'ClientBin/40x'; arrs = []; bits = []
        for c in CH:
            with Image.open(folder / f'{row.acquisition}-{c}.tif') as im:
                a = np.array(im); b = int(im.tag_v2[258][0])
            arrs.append(a.astype(np.uint16)); bits.append(b)
        raw = np.stack(arrs, -1); disp = preprocess_frame(raw, [65535 / (2 ** b - 1) for b in bits])[::-1, ::-1]
        h, w = disp.shape[:2]; a = disp[h // 2 - 200:h // 2 + 201, w // 2 - 200:w // 2 + 201]
        if a.shape != (401, 401, 4): return key, False
        np.savez_compressed(t, image=a); return key, True
    except Exception as e:
        return key, repr(e)
res = []
with ThreadPoolExecutor(6) as pool:
    for i, (k, ok) in enumerate(pool.map(one, r.itertuples()), 1):
        res.append(dict(cell_key=k, ok=ok is True, note=None if ok is True else str(ok)))
        if i % 100 == 0: print(i, flush=True)
pd.DataFrame(res).to_csv(OUT / '40x_extraction_status.csv', index=False); print(pd.DataFrame(res).ok.sum(), 'extracted')
