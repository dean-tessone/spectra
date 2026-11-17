"""Many-cell gallery per Leiden cluster.

Rule (no visual selection): for each cluster, mass-90 attributed cells of held-out test cancer patients, ranked by attribution.
40x re-images first (cells with a cached, eligible 40x acquisition; one cell per patient, then further cells of the same
patients if needed), then 10x crops of the next most-attributed cells, one per patient, until N tiles. 40x tiles are centred
automatically with the Figure 3 rule (center_40x.py; no manual overrides) and all tiles show the same 35 um field
(40x: +/-98 px at 0.18 um/px; 10x: +/-30 px at 0.59 um/px). Display processing as in Figure 3 (cell_ext convention, sx.boost).

    python gallery_many.py select     # choose cells, extract missing 10x crops (read-only raw frames)
    python gallery_many.py tiles      # write display tiles
"""
from pathlib import Path
import importlib.util, json, sys
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
A = REPO / 'analyses/figure3_locked6_attribution_20261111'; L = A / 'leiden'
N, SPARE = 18, 8
COLS = ['cell_key', 'slide_id', 'frame_id', 'cell_id', 'source_row', 'x', 'y', 'area', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']


def select():
    a = pd.read_parquet(L / 'atlas_mass90_private.parquet'); a = a[a.label == 1]
    have40 = {f.stem for f in (A / 'cache_40x').glob('*.npz')}
    a['x40'] = a.cell_key.isin(have40); rows = []
    for k, g in a.sort_values('relevance', ascending=False).groupby('cluster'):
        g40 = g[g.x40]; first = g40.drop_duplicates('donor_id'); more = g40[~g40.cell_key.isin(first.cell_key)]
        p40 = pd.concat([first, more]).head(N).assign(source='40x', order=lambda d: range(len(d)))
        g10 = g[~g.x40 & ~g.donor_id.isin(p40.donor_id)].drop_duplicates('donor_id').head(N - len(p40) + SPARE)
        rows += [p40, g10.assign(source='10x', order=lambda d: range(len(d)))]
    s = pd.concat(rows, ignore_index=True); s.to_csv(L / 'gallery_many_candidates_private.csv', index=False)
    print(s.groupby(['cluster', 'source']).size().unstack(fill_value=0).T.to_string())
    spec = importlib.util.spec_from_file_location('review_build', REPO / 'imaging/cell_images/build.py')
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h); h.HERE = A / 'tenx_many'; h.HERE.mkdir(exist_ok=True)
    recs = []
    for i, (sid, g) in enumerate(s[s.source == '10x'][COLS].groupby('slide_id')):
        try:
            recs.append(h.extract_slide((sid, g)))
        except Exception as e:
            recs.append(dict(slide_id=sid, available_keys=[], error=repr(e)))
        if i % 25 == 0:
            print('slides', i, flush=True)
    json.dump(recs, open(L / 'gallery_many_tenx_provenance.json', 'w'), indent=1)
    print(sum(len(r['available_keys']) for r in recs), 'of', int((s.source == '10x').sum()), '10x crops available')


def _one(args):
    spec = importlib.util.spec_from_file_location('review_build', REPO / 'imaging/cell_images/build.py')
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h); h.HERE = A / 'tenx_many'
    try:
        return h.extract_slide(args)
    except Exception as e:
        return dict(slide_id=args[0], available_keys=[], error=repr(e))


def extract(workers=8):
    """Parallel 10x extraction for the saved candidates (per-slide caches are reused)."""
    from multiprocessing import Pool
    s = pd.read_csv(L / 'gallery_many_candidates_private.csv', dtype={'slide_id': str}); (A / 'tenx_many').mkdir(exist_ok=True)
    jobs = [(sid, g) for sid, g in s[s.source == '10x'][COLS].groupby('slide_id')]
    with Pool(workers) as pool:
        recs = pool.map(_one, jobs, chunksize=1)
    json.dump(recs, open(L / 'gallery_many_tenx_provenance.json', 'w'), indent=1)
    print(sum(len(r['available_keys']) for r in recs), 'of', int((s.source == '10x').sum()), '10x crops available;', sum('error' in r for r in recs), 'slides with errors')


def centre(im, ref4):
    """Figure 3 centring rule on a 401-px 40x context; returns (cx, cy) or None."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    ref = np.log1p(np.asarray(ref4, float)); ref = ref - ref.mean()
    im = im.astype(np.float32); s = ndi.gaussian_filter(im.sum(-1), 3); lab, n = ndi.label(s > threshold_otsu(s)); best = None
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 300:
            continue
        cy, cx = ndi.center_of_mass(s * m); d = np.hypot(cy - 200, cx - 200)
        if d > 120:
            continue
        v = np.log1p(im[m].mean(0)); v = v - v.mean(); score = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9)) - .004 * d
        if best is None or score > best[2]:
            best = (cx, cy, score)
    return None if best is None else (int(round(best[0])), int(round(best[1])))


def dim(raw, floor=45, contrast=30):
    """True when the crop has no displayable object: after a small median filter (removes speckle) the centre is either dark or
    uniformly textured (no object above its own background); boosting such a crop would only show noise."""
    from scipy.ndimage import median_filter
    m = median_filter(raw.max(axis=-1), size=11 if raw.shape[0] > 100 else 5)   # smaller than any cell at either magnification
    hh, ww = m.shape; c = m[int(hh * .2):int(hh * .8), int(ww * .2):int(ww * .8)]
    hi = float(np.percentile(c, 99.0)); return hi < floor or hi - float(np.median(m)) < contrast


def tiles():
    sys.path.insert(0, str(REPO / 'imaging/cell_images')); sys.path.insert(0, str(REPO / 'figures/code'))
    from cell_ext_display import rgb
    import sx
    from PIL import Image
    s = pd.read_csv(L / 'gallery_many_candidates_private.csv', dtype={'slide_id': str})
    ten = {}
    for f in (A / 'tenx_many/image_cache').glob('*.npz'):
        z = np.load(f); ten.update(dict(zip(z['cell_keys'], z['images'])))
    # favour 40x. After the cancer patients' 40x cells, add 40x cells of test non-cancer controls in the same cluster.
    at = pd.read_parquet(L / 'atlas_mass90_private.parquet', columns=['cell_key', 'cluster', 'label', 'relevance', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean'])
    have40 = {f.stem for f in (A / 'cache_40x').glob('*.npz')}
    ctrl = at[(at.label == 0) & at.cell_key.isin(have40)].sort_values('relevance', ascending=False)
    out, keep = {}, []
    for k, g in s.groupby('cluster'):
        n = 0
        g40 = pd.concat([g[g.source == '40x'].sort_values('order'), ctrl[ctrl.cluster == k]], ignore_index=True)
        for r in g40.itertuples():
            if n >= N:
                break
            im = np.load(A / 'cache_40x' / f'{r.cell_key}.npz')['image']; c = centre(im, [r.DAPI_mean, r.TRITC_mean, r.CY5_mean, r.FITC_mean])
            if c is None:
                continue
            cx, cy = np.clip(c[0], 98, 302), np.clip(c[1], 98, 302)
            raw = rgb(im[cy - 98:cy + 99, cx - 98:cx + 99])                              # uint8 RGB
            if dim(raw):
                continue
            t = sx.boost(raw)
            t = np.asarray(Image.fromarray(np.asarray(t, np.uint8)).resize((120, 120), Image.LANCZOS))
            out[str(len(keep))] = t; keep.append(dict(cluster=k, source='40x', cell_key=r.cell_key, tile=len(keep))); n += 1
        for r in g[g.source == '10x'].sort_values('order').itertuples():
            if n >= N or r.cell_key not in ten:
                continue
            im = ten[r.cell_key]; h = im.shape[0] // 2
            raw = rgb(im[h - 30:h + 31, h - 30:h + 31])
            if dim(raw):
                continue
            t = sx.boost(raw)
            t = np.asarray(Image.fromarray(np.asarray(t, np.uint8)).resize((120, 120), Image.BICUBIC))
            out[str(len(keep))] = t; keep.append(dict(cluster=k, source='10x', cell_key=r.cell_key, tile=len(keep))); n += 1
    k = pd.DataFrame(keep); k.to_csv(L / 'gallery_many_private.csv', index=False)
    np.savez_compressed(L / 'gallery_many_private.npz', field_um=35.3, **out)
    print(k.groupby(['cluster', 'source']).size().unstack(fill_value=0).T.to_string())


if __name__ == '__main__':
    {'select': select, 'extract': extract, 'tiles': tiles}[sys.argv[1]]()
