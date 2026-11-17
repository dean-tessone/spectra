"""IMC <-> 40x registration for slides whose raw 10x frames are unavailable. Read-only on all source drives. Run with the OpenIMC venv.

Each IMC ROI ({frame}_{cell} in the MCD description / 'Pano 01_1_{frame}_{cell}_{acq}' text name) was targeted at the
same (frame, cell) as a 40x re-image ({cell}-{frame}-{X}-{Y}-{CH}.tif; X, Y = the cell's 10x frame position).
  1. 40x DAPI rotated 180 deg (the Figure 3 convention that matches 10x orientation), resampled to 1 um/px for a grid
     of 40x pixel sizes; its central square, mirrored or not and rotated (2 deg steps), cropped to the inscribed
     square, is the template searched (40x pixel ~0.171 um from validation; grid 0.16-0.18) in the IMC DNA image (normalised cross-correlation).
  2. ICP/RANSAC affine refinement on nuclei (IMC DNA peaks <-> 40x DAPI peaks at 1 um/px), as register.refine.
  3. 10x frame (x, y) -> rotated 40x pixel: u = W/2 + (x - X) k, v = H/2 + (y - Y) k, k = 0.578 / (40x um/px).
Validated on slides with BOTH a 10x registration and 40x images (`python register40.py validate`): the 10x->IMC
position of the same cells via the 40x route vs the 10x route."""
from pathlib import Path
import sys, re, os, json
import numpy as np, pandas as pd, cv2
from PIL import Image
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from register import prep, nuclei  # noqa: E402
DZ = Path(os.environ.get('SPECTRA_40X_ROOT', 'raw/40x')); IF_UM = 0.578
PAT = re.compile(r'(\d+)-(\d+)-(-?\d+)-(-?\d+)-DAPI\.tif')


def find40(slide, frame, cell):
    d = DZ / slide / 'ClientBin/40x'
    for f in os.listdir(d):
        m = PAT.fullmatch(f)
        if m and int(m.group(1)) == cell and int(m.group(2)) == frame:
            return d / f, int(m.group(3)), int(m.group(4))
    return None


def load40(path, ch='DAPI'):
    with Image.open(str(path).replace('-DAPI.tif', f'-{ch}.tif')) as im:
        a = np.array(im).astype(np.float32)
    return a[::-1, ::-1].copy()   # 180 deg, Figure 3 convention


def register40(dna, d40, px_grid=(0.16, 0.165, 0.17, 0.175, 0.18), angles=range(0, 360, 2)):
    """Best (score, px, mirror, angle, A) with A: rotated-40x full-res pixel (u, v, 1) -> IMC pixel (col, row, 1)."""
    big = prep(dna, 1.5); best = dict(score=-1)
    for px in px_grid:
        small = cv2.resize(d40, None, fx=px, fy=px, interpolation=cv2.INTER_AREA)
        s = min(small.shape); y0 = (small.shape[0] - s) // 2; x0 = (small.shape[1] - s) // 2
        sq = prep(small[y0:y0 + s, x0:x0 + s], 1.5); h = int(s / 2 / 1.42); c = s // 2
        for mirror in (False, True):
            src = sq[:, ::-1] if mirror else sq
            for a in angles:
                M = cv2.getRotationMatrix2D((s / 2, s / 2), a, 1.0)
                t = cv2.warpAffine(src, M, (s, s), flags=cv2.INTER_LINEAR)[c - h:c + h, c - h:c + h]
                r = cv2.matchTemplate(big, t, cv2.TM_CCOEFF_NORMED); _, mv, _, ml = cv2.minMaxLoc(r)
                if mv > best['score']:
                    A = np.diag([px, px, 1.0])
                    A = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ A
                    if mirror:
                        A = np.array([[-1, 0, s - 1], [0, 1, 0], [0, 0, 1]], float) @ A
                    A = np.vstack([M, [0, 0, 1]]) @ A
                    A = np.array([[1, 0, ml[0] - (c - h)], [0, 1, ml[1] - (c - h)], [0, 0, 1]], float) @ A
                    best = dict(score=float(mv), px=px, mirror=mirror, angle=a, A=A)
    return best


def refine40(best, dna, d40, iters=15):
    """ICP: IMC nuclei -> 40x nuclei (both at ~1 um/px); returns IMC->rotated-40x-full-res affine and diagnostics
    (matched fraction within 3 um; median residual in um), restricted to IMC nuclei that fall inside the 40x field."""
    from scipy.spatial import cKDTree
    px = best['px']; small = cv2.resize(d40, None, fx=px, fy=px, interpolation=cv2.INTER_AREA)
    pi = nuclei(dna, 1.5, 4, 80, otsu=True); pf = nuclei(small, 1.5, 4, 80, otsu=True) / px   # 40x full-res px
    tree = cKDTree(pf); A = np.linalg.inv(best['A']); tol = 8.0 / px
    for _ in range(iters):
        P = (A @ np.c_[pi, np.ones(len(pi))].T).T[:, :2]
        ins = (P[:, 0] > 0) & (P[:, 0] < d40.shape[1]) & (P[:, 1] > 0) & (P[:, 1] < d40.shape[0])
        d, j = tree.query(P); ok = ins & (d < tol)
        if ok.sum() < 6:
            break
        M, _ = cv2.estimateAffine2D(pi[ok].astype(np.float32), pf[j[ok]].astype(np.float32), method=cv2.RANSAC, ransacReprojThreshold=3.0 / px)
        if M is None:
            break
        A = np.vstack([M, [0, 0, 1]]); tol = max(4.0 / px, tol * .8)
    P = (A @ np.c_[pi, np.ones(len(pi))].T).T[:, :2]; d, _ = tree.query(P)
    ins = (P[:, 0] > 0) & (P[:, 0] < d40.shape[1]) & (P[:, 1] > 0) & (P[:, 1] < d40.shape[0])
    return A, dict(imc_nuclei_in_40x=int(ins.sum()), matched_frac=float((d[ins] * px < 3).mean()) if ins.any() else 0.,
                   median_residual_um=float(np.median(d[ins]) * px) if ins.any() else np.nan, um_per_40x_px=float(1 / np.sqrt(abs(np.linalg.det(A[:2, :2])))))


def tenx_to_imc(x, y, X, Y, W, H, A_imc_to_40, um40):
    k = IF_UM / um40; u = W / 2 + (np.asarray(x, float) - X) * k; v = H / 2 + (np.asarray(y, float) - Y) * k
    p = np.linalg.inv(A_imc_to_40) @ np.vstack([u, v, np.ones(np.size(u))])
    return p[0], p[1]


def validate():
    """Slides with both routes: compare IMC position of linked cells via 40x vs the accepted 10x registration."""
    from link import load_roi
    L = pd.concat([pd.read_parquet(p) for p in (REPO / 'analyses/imc_linkage_20261111').glob('linked_cells*_private.parquet')], ignore_index=True)
    rows = []
    for roi, g in L.groupby('roi'):
        sid = g.slide_id.iloc[0]; m = re.match(rf'{sid}_(\d+)_(\d+)_(\d+)\.(txt|npz)', roi)
        if not m or not (DZ / sid / 'ClientBin/40x').is_dir():
            continue
        hit = find40(sid, int(m.group(1)), int(m.group(2)))
        if hit is None:
            continue
        arr, cols = load_roi(dict(slide_id=sid, file=roi)); dna = arr[[i for i, c in enumerate(cols) if 'DNA' in c]].sum(0)
        d40 = load40(hit[0]); best = register40(dna, d40); A, diag = refine40(best, dna, d40)
        c, r = tenx_to_imc(g.x, g.y, hit[1], hit[2], d40.shape[1], d40.shape[0], A, diag['um_per_40x_px'])
        err = np.hypot(c - g.imc_col.to_numpy(), r - g.imc_row.to_numpy())
        rows.append(dict(roi=roi, score=best['score'], angle=best['angle'], mirror=best['mirror'], **diag, n=len(g), median_err_um=float(np.median(err)), max_err_um=float(err.max())))
        print(rows[-1], flush=True)
        if len(rows) >= 12:
            break
    pd.DataFrame(rows).to_csv(REPO / 'analyses/imc_linkage_20261111/register40_validation.csv', index=False)


if __name__ == '__main__':
    validate()
