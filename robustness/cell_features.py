"""Revision step 2.1b: CellProfiler-style image features of cluster 10 cells (2026-11-17). Inference-free; read-only raw tiles.

Cells: mass-90 attributed cells assigned to cluster 10 (paper numbering; internal Leiden 2).
  * primary test cancers: up to 6 randomly chosen cells per patient (seed 20261117);
  * primary test reference controls, lung-workup controls, BRCA-carrier controls: up to 20 per patient.
Images: raw 10x tiles (DAPI, TRITC=pan-CK, CY5=CD45/CD31, FITC=vimentin) staged read-only from the scanner mount to scratch;
segmentation masks from the mask store matched by cell id, area and centroid; mask mean intensities verified against the
source table (as in the audited review pipeline). No display preprocessing: features use raw counts.
Features per cell (skimage): cell shape (area, perimeter, form factor, solidity, eccentricity, extent, axis lengths), nucleus
(Otsu on DAPI within the cell: area, nucleus/cell ratio, eccentricity, solidity, number of nuclear objects, centroid offset),
per channel: mean, SD, CV, 10th/90th percentile, max, integrated intensity, radial distribution (fraction in outer and inner
thirds of the distance-to-edge), GLCM texture (contrast, homogeneity, correlation, energy; 32 levels, distance 1, 4 angles)
and granularity proxy (Laplacian-of-Gaussian variance).
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys, json, tempfile, traceback
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'imaging/cell_images')); sys.path.insert(0, str(REPO / 'model/pipeline'))
import render_pancreatic_model_comparison as X  # noqa: E402
from common import local  # noqa: E402
from spectra.review_intensity import acquisition_gains, apply_gains  # noqa: E402
from skimage.measure import regionprops, label
from skimage.filters import threshold_otsu, laplace, gaussian
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import distance_transform_edt

L = REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
ADA = REPO / 'runs/specific_adaptation_20261111/attribution'
RUN = REPO / 'runs/revision_20261117/cell_features'; RUN.mkdir(parents=True, exist_ok=True)
SCR = Path(tempfile.gettempdir()) / 'spectra_cell_feature_tiles'; SCR.mkdir(parents=True, exist_ok=True)
CH = ['DAPI', 'CK', 'CD45CD31', 'VIM']
INT10 = int(pd.read_csv(L / 'cluster_display_numbers.csv').set_index('number').cluster[10])
rng = np.random.default_rng(20261117)


def select_cells():
    cols = ['donor_id', 'slide_id', 'source_row', 'frame_id', 'cell_id', 'x', 'y', 'area', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']
    at = pd.read_parquet(L / 'atlas_mass90_private.parquet', columns=cols + ['label', 'cluster'])
    at = at[at.cluster == INT10].copy(); at['group'] = np.where(at.label == 1, 'Cancer', 'Reference control')
    inv = pd.read_csv(REPO / 'runs/paper_regeneration_20261109_v1a/cohort/local_cache_inventory.csv', dtype=str, usecols=['slide_id', 'source'])
    src = dict(zip(inv.slide_id, inv.source)); dx = []
    for task, grp in [('lung', 'Lung work-up control'), ('breast', 'BRCA-carrier control')]:
        c = pd.read_parquet(ADA / f'{task}_cells_clustered_private.parquet', columns=['donor_id', 'slide_id', 'source_row', 'source', 'label', 'cluster'])
        c = c[c.label == 0].copy(); c['pos'] = c.source.clip(lower=0)
        c = c.sort_values(['donor_id', 'pos'], ascending=[True, False]); tot = c.groupby('donor_id').pos.transform('sum')
        c = c[(c.groupby('donor_id').pos.cumsum() - c.pos) / tot < .9]; c = c[c.cluster == INT10]
        for s, g in c.groupby('slide_id'):
            f = pd.read_parquet(local(src[s]), columns=['frame_id', 'cell_id', 'x', 'y', 'area', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean'])
            h = pd.concat([g[['donor_id', 'slide_id', 'source_row']].reset_index(drop=True), f.iloc[g.source_row.to_numpy()].reset_index(drop=True)], axis=1)
            h['group'] = grp; dx.append(h)
    cells = pd.concat([at.drop(columns=['label', 'cluster']), pd.concat(dx)], ignore_index=True)
    cap = np.where(cells.group == 'Cancer', 6, 20)
    cells['r'] = rng.random(len(cells)); cells['rank'] = cells.groupby('donor_id').r.rank(method='first')
    cells = cells[cells['rank'] <= cap].drop(columns=['r', 'rank']).reset_index(drop=True)
    cells['cell_key'] = cells.slide_id + '_r' + cells.source_row.astype(str)
    return cells


def glcm(img, m):
    v = img[m]; lo, hi = np.percentile(v, [1, 99])
    q = np.clip((img - lo) / max(hi - lo, 1) * 31, 0, 31).astype(np.uint8); q[~m] = 0
    ys, xs = np.nonzero(m); q = q[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    G = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=False).astype(float)
    G[0, :, :, :] = 0; G[:, 0, :, :] = 0  # drop background pairs
    G = G / np.maximum(G.sum(axis=(0, 1), keepdims=True), 1)
    return {k: float(graycoprops(G, k).mean()) for k in ('contrast', 'homogeneity', 'correlation', 'energy')}


def features(crop, m):
    f = {}; p = regionprops(m.astype(np.uint8))[0]
    f.update(cell_area=p.area, cell_perimeter=p.perimeter, cell_form_factor=4 * np.pi * p.area / max(p.perimeter, 1) ** 2, cell_solidity=p.solidity,
             cell_eccentricity=p.eccentricity, cell_extent=p.extent, cell_major_axis=p.major_axis_length, cell_minor_axis=p.minor_axis_length)
    dist = distance_transform_edt(m); dm = dist[m]; t1, t2 = np.percentile(dm, [33.3, 66.7])
    dapi = crop[..., 0].astype(float)
    try:
        nm = m & (gaussian(dapi, 1, preserve_range=True) > threshold_otsu(dapi[m]))
    except ValueError:
        nm = m.copy()
    lab = label(nm); f['nuclear_objects'] = int(lab.max())
    if nm.any():
        q = max(regionprops(lab), key=lambda r: r.area)
        f.update(nucleus_area=float(nm.sum()), nucleus_cell_ratio=float(nm.sum() / m.sum()), nucleus_eccentricity=q.eccentricity, nucleus_solidity=q.solidity,
                 nucleus_offset=float(np.hypot(*(np.array(q.centroid) - np.array(p.centroid)))))
    for i, ch in enumerate(CH):
        img = crop[..., i].astype(float); v = img[m]
        f.update({f'{ch}_mean': v.mean(), f'{ch}_sd': v.std(), f'{ch}_cv': v.std() / max(v.mean(), 1), f'{ch}_p10': np.percentile(v, 10), f'{ch}_p90': np.percentile(v, 90),
                  f'{ch}_max': v.max(), f'{ch}_integrated': v.sum(),
                  f'{ch}_outer_fraction': img[m & (dist <= t1)].sum() / max(v.sum(), 1), f'{ch}_inner_fraction': img[m & (dist > t2)].sum() / max(v.sum(), 1),
                  f'{ch}_log_var': float(laplace(gaussian(img, 1, preserve_range=True))[m].var())})
        f.update({f'{ch}_glcm_{k}': val for k, val in glcm(img, m).items()})
        if nm.any() and (m & ~nm).any():
            f[f'{ch}_nuc_cyto_ratio'] = img[nm].mean() / max(img[m & ~nm].mean(), 1)
    return f


def run_slide(args):
    slide, part = args
    out = RUN / f'{slide}.parquet'
    if out.exists():
        return slide, 'cached'
    from slideutils.utils.frame import Frame
    rows = []; miss = []
    try:
        raw = X.acquisition(slide, part.frame_id.iloc[0])
    except Exception as e:
        pd.DataFrame([dict(cell_key=k, reason=repr(e)) for k in part.cell_key]).to_csv(RUN / f'{slide}.missing.csv', index=False)
        return slide, 'no acquisition'
    gains = None
    if (raw / 'slideinfo.txt').is_file():
        try: gains = acquisition_gains((raw / 'slideinfo.txt').read_text())
        except (ValueError, KeyError): pass
    for fid, ev in part.groupby('frame_id'):
        ev = ev.reset_index(drop=True)
        try:
            with tempfile.TemporaryDirectory(dir=SCR) as tmp:
                paths = []
                for off in (0, 2304, 4608, 9216):
                    n = int(fid) + off
                    s = next((raw / f'Tile{n:06d}.{e}' for e in ('tif', 'jpg') if (raw / f'Tile{n:06d}.{e}').is_file()), None)
                    if s is None: raise FileNotFoundError(f'tile {n}')
                    d = Path(tmp) / s.name; X.stage(s, d); paths.append(str(d))
                    if s.suffix == '.jpg': X.stage(s.with_suffix('.tags'), d.with_suffix('.tags'))
                fr = Frame(int(fid), X.CHANNELS, paths); fr.readImage(); mc = {}
                for r in ev.itertuples():
                    b, _ = X.mask_for_cell(slide, fid, r, mc)
                    if b is None: miss.append(dict(cell_key=r.cell_key, reason='mask')); continue
                    px = fr.image[b]; exp_ = np.array([r.DAPI_mean, r.TRITC_mean, r.CY5_mean, r.FITC_mean], int)
                    err = int(np.abs(px.mean(0).astype(np.uint16).astype(int) - exp_).max()); g = None
                    if err > 1:
                        if gains is not None and int(np.abs(apply_gains(px, gains).mean(0).astype(np.uint16).astype(int) - exp_).max()) <= 1: g = gains
                        else: miss.append(dict(cell_key=r.cell_key, reason=f'intensity check {err}')); continue
                    img = np.pad(fr.image, ((37, 37), (37, 37), (0, 0)))[int(r.y):int(r.y) + 75, int(r.x):int(r.x) + 75]
                    if g is not None: img = apply_gains(img.reshape(-1, 4), g).reshape(img.shape)
                    img = img.astype(float)
                    cut = np.pad(b, 37)[int(r.y):int(r.y) + 75, int(r.x):int(r.x) + 75]
                    try:
                        rows.append(dict(cell_key=r.cell_key, donor_id=r.donor_id, group=r.group, **features(img, cut)))
                    except Exception as e:
                        miss.append(dict(cell_key=r.cell_key, reason='features ' + repr(e)))
        except Exception as e:
            miss.extend(dict(cell_key=k, reason=repr(e)) for k in ev.cell_key)
    pd.DataFrame(rows).to_parquet(out)
    if miss: pd.DataFrame(miss).to_csv(RUN / f'{slide}.missing.csv', index=False)
    return slide, f'{len(rows)} ok, {len(miss)} missing'


if __name__ == '__main__':
    cells = select_cells(); cells.to_parquet(RUN / 'selected_cells_private.parquet')
    print(cells.groupby('group').agg(cells=('cell_key', 'size'), patients=('donor_id', 'nunique')), flush=True)
    tasks = list(cells.groupby('slide_id')); done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for s, msg in pool.map(lambda t: (lambda: run_slide(t))() if True else None, tasks):
            done += 1
            if done % 50 == 0: print(done, '/', len(tasks), s, msg, flush=True)
    print('done', flush=True)
