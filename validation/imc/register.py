"""Register IMC ROIs to the IF scan frame named in the ROI.

IMC DNA (Ir191+Ir193, 1 um/px) is matched to the IF DAPI tile of that frame (read-only from the raw image store) by
exhaustive search over rotation (2 deg steps, with and without mirror), IF pixel size (um/px grid) and position,
using normalized cross-correlation of the central rotation-invariant 280x280 IMC crop. Output: per ROI the best
transform and its score, and functions to map IF (x, y) into IMC pixel coordinates. Read-only on both drives."""
import os
from pathlib import Path
import sys, re
import numpy as np, pandas as pd, cv2
from scipy import ndimage as ndi
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'imaging/cell_images'))
import render_pancreatic_model_comparison as extraction  # noqa: E402
IMC = Path(os.environ.get('SPECTRA_IMC_ROOT', 'raw/imc'))


def imc_roi(path):
    t = pd.read_csv(path, sep='\t')
    H, W = int(t.Y.max()) + 1, int(t.X.max()) + 1
    cols = [c for c in t.columns[6:]]
    arr = np.zeros((len(cols), H, W), np.float32)
    for i, c in enumerate(cols):
        arr[i, t.Y.to_numpy(), t.X.to_numpy()] = t[c].to_numpy()
    return arr, cols


def if_frame(slide, frame, channel=0):
    raw = extraction.acquisition(slide, frame)
    off = [0, 2304, 4608, 9216][channel]
    p = next(raw / f'Tile{frame + off:06d}.{e}' for e in ['tif', 'jpg'] if (raw / f'Tile{frame + off:06d}.{e}').is_file())
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float32)


def prep(img, sigma):
    img = np.log1p(np.clip(img, 0, None)); img = ndi.gaussian_filter(img, sigma)
    return ((img - img.mean()) / (img.std() + 1e-6)).astype(np.float32)


def register(dna, dapi, px_grid=(0.55, 0.59, 0.63, 0.67, 0.71), angles=range(0, 360, 2)):
    """Returns dict(score, angle, mirror, px, cx, cy): IMC centre maps to IF (cx, cy) in full-res IF pixels."""
    best = dict(score=-1)
    tmpl0 = prep(dna, 1.5); h = int(min(dna.shape) * .35)
    for px in px_grid:
        f = 1.0 / px  # IF px per um... IF image resized to 1 um/px by factor px
        big = prep(cv2.resize(dapi, None, fx=px, fy=px, interpolation=cv2.INTER_AREA), 1.5)
        big = cv2.copyMakeBorder(big, h, h, h, h, cv2.BORDER_CONSTANT, value=0)
        for mirror in (False, True):
            src = tmpl0[:, ::-1] if mirror else tmpl0
            for a in angles:
                M = cv2.getRotationMatrix2D((src.shape[1] / 2, src.shape[0] / 2), a, 1.0)
                rot = cv2.warpAffine(src, M, src.shape[::-1], flags=cv2.INTER_LINEAR)
                c = rot.shape[0] // 2; t = rot[c - h:c + h, c - h:c + h]
                r = cv2.matchTemplate(big, t, cv2.TM_CCOEFF_NORMED)
                _, mv, _, ml = cv2.minMaxLoc(r)
                if mv > best['score']:
                    cx = ml[0] / px; cy = ml[1] / px  # template centre in IF full-res px (pad = half template)
                    best = dict(score=float(mv), angle=a, mirror=mirror, px=px, cx=cx, cy=cy, mlx=int(ml[0]), mly=int(ml[1]),
                                M=M.tolist(), off=int(c - h), W=int(src.shape[1]), h=h)
    return best


def imc_to_if_affine(reg):
    """3x3 affine mapping IMC pixel (col, row, 1) to IF full-resolution (x, y, 1), exactly as used in matching."""
    A = np.eye(3)
    if reg['mirror']:
        A = np.array([[-1, 0, reg['W'] - 1], [0, 1, 0], [0, 0, 1]], float) @ A
    A = np.vstack([np.array(reg['M'], float), [0, 0, 1]]) @ A
    A = np.array([[1, 0, -reg['off'] + reg['mlx'] - reg['h']], [0, 1, -reg['off'] + reg['mly'] - reg['h']], [0, 0, 1]], float) @ A
    A = np.diag([1 / reg['px'], 1 / reg['px'], 1]) @ A
    return A


def if_to_imc(x, y, reg):
    """Map IF full-res (x, y) to IMC pixel coordinates (col, row)."""
    B = np.linalg.inv(imc_to_if_affine(reg))
    p = B @ np.vstack([np.asarray(x, float), np.asarray(y, float), np.ones(np.size(x))])
    return p[0], p[1]


def nuclei(img, sigma, min_distance, q, otsu=False):
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    g = ndi.gaussian_filter(np.log1p(np.clip(img, 0, None)), sigma)
    thr = threshold_otsu(g) if otsu else np.percentile(g, q)
    pk = peak_local_max(g, min_distance=min_distance, threshold_abs=thr)
    return pk[:, ::-1].astype(float)  # (x, y)


def refine(reg, dna, dapi, iters=15):
    """ICP refinement: IMC nuclei (DNA peaks) -> IF nuclei (DAPI peaks), RANSAC affine on nearest neighbours.
    Returns refined 3x3 IMC->IF affine and diagnostics (matched fraction, median residual in IF px)."""
    from scipy.spatial import cKDTree
    pi = nuclei(dna, 1.5, 4, 80, otsu=True); pf = nuclei(dapi, 2.0, 5, 80)
    tree = cKDTree(pf); A = imc_to_if_affine(reg); tol = 14.0
    for it in range(iters):
        P = (A @ np.c_[pi, np.ones(len(pi))].T).T[:, :2]
        d, j = tree.query(P); ok = d < tol
        if ok.sum() < 10:
            break
        M, inl = cv2.estimateAffine2D(pi[ok].astype(np.float32), pf[j[ok]].astype(np.float32), method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if M is None:
            break
        A = np.vstack([M, [0, 0, 1]]); tol = max(6.0, tol * .8)
    P = (A @ np.c_[pi, np.ones(len(pi))].T).T[:, :2]; d, _ = tree.query(P)
    inside = (P[:, 0] > 0) & (P[:, 0] < dapi.shape[1]) & (P[:, 1] > 0) & (P[:, 1] < dapi.shape[0])
    return A, dict(imc_nuclei=int(len(pi)), inside=int(inside.sum()), matched_frac=float((d[inside] < 5).mean()) if inside.any() else 0.,
                   median_residual_if_px=float(np.median(d[inside])) if inside.any() else np.nan,
                   scale=float(np.sqrt(abs(np.linalg.det(A[:2, :2])))))


if __name__ == '__main__':
    import time
    t0 = time.time()
    arr, cols = imc_roi(IMC / '0A73E07/0A73E07_1638_1_3.txt')
    dna = arr[[i for i, c in enumerate(cols) if c.startswith('DNA')]].sum(0)
    dapi = if_frame('0A73E07', 1638)
    print('IF', dapi.shape, 'IMC', dna.shape)
    reg = register(dna, dapi); print(reg, 'sec', round(time.time() - t0, 1))
    print('distance of ROI centre to HDSCA interesting cell (946,557):', np.hypot(reg['cx'] - 946, reg['cy'] - 557))
