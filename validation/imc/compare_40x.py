"""Review sheet (private): segmented high-attention IMC-linked cells that also have a 40x re-image. Columns: 40x
(cell_ext display, original Figure 3 processing, not re-centred), the SPECTRA 10x crop, IMC DNA and the cell's relevant
IMC markers (gallery.py display). 40x matching rule as attribution/atlas/index_40x.py
(same slide and frame, |dx|,|dy|<=10 px, all four channels, exactly one candidate). Run with the OpenIMC venv."""
import os
from pathlib import Path
import sys, re, json
from collections import defaultdict
import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from skimage.segmentation import find_boundaries
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / 'imaging/cell_images'))
import gallery as G  # noqa: E402
from cell_ext_display import preprocess_frame, rgb  # noqa: E402
from register import if_frame  # noqa: E402
X40_HALF = 98   # 40x crop half-width in px (0.18 um/px; matches the 10x crop, 30 px x 0.59 um/px)
PAT = re.compile(r'(\d+)-(\d+)-(-?\d+)-(-?\d+)-(DAPI|TRITC|FITC|CY5)\.tif', re.I)
CH = ['DAPI', 'TRITC', 'CY5', 'FITC']


def forty(slide, frame, x, y):
    p = Path(os.environ.get('SPECTRA_40X_ROOT', 'raw/40x')) / slide / 'ClientBin/40x'
    if not p.is_dir():
        return None
    files = defaultdict(set)
    for f in p.iterdir():
        m = PAT.fullmatch(f.name)
        if m:
            files[tuple(map(int, m.groups()[:4]))].add(m.group(5).upper())
    hits = [k for k, ch in files.items() if k[1] == frame and abs(k[2] - x) <= 10 and abs(k[3] - y) <= 10 and len(ch) == 4]
    if len(hits) != 1:
        return None
    acq = '-'.join(map(str, hits[0])); arrs, bits = [], []
    for c in CH:
        with Image.open(p / f'{acq}-{c}.tif') as im:
            arrs.append(np.array(im).astype(np.uint16)); bits.append(int(im.tag_v2[258][0]))
    disp = preprocess_frame(np.stack(arrs, -1), [65535 / (2 ** b - 1) for b in bits])[::-1, ::-1]
    h, w = disp.shape[:2]; k = X40_HALF   # same field as the 10x crop: 30 px x 0.59 um = 17.7 um = 98 px x 0.18 um
    im = rgb(disp[h // 2 - k:h // 2 + k, w // 2 - k:w // 2 + k]).astype(np.float32)
    return np.clip(im / (np.percentile(im, 99.7) + 1e-6), 0, 1)   # display-only linear stretch (dim re-images)


def main():
    L = pd.concat([pd.read_parquet(p) for p in G.AN.glob('linked_cells*_private.parquet')], ignore_index=True)
    regs = pd.concat([pd.read_csv(p, dtype={'slide_id': str}) for p in G.AN.glob('roi_registration*.csv')]).drop_duplicates('file').set_index('file')
    P = pd.read_csv(G.AN / 'imc_profiles_private.csv', dtype={'slide_id': str})
    P = P[P.match.isin(['inside nucleus', 'inside expanded cell'])].assign(num=lambda d: d.cluster.map(G.NUM)).sort_values(['num', 'slide_id', 'positive_rank'])
    rows = []
    for p in P.itertuples():
        c = L[(L.roi == p.roi) & (L.source_row == p.source_row)].iloc[0]
        im40 = forty(p.slide_id, int(c.frame_id), int(c.x), int(c.y))
        if im40 is not None:
            rows.append((p, c, im40))
    print(len(rows), 'segmented cells with 40x')
    ncol = 9
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.05 * ncol, 1.15 * len(rows)), squeeze=False)
    for ri, (p, c, im40) in enumerate(rows):
        pr = P[(P.slide_id == p.slide_id) & (P.positive_rank == p.positive_rank)].iloc[0]
        markers = G.informative(p.slide_id, p.positive_rank, pr, k=6)
        markers = [m for m in ['CK', 'EpCAM', 'Vimentin', 'CD31', 'CD45'] if pd.notna(pr.get(f'{m}_pct'))][:3] + markers
        markers = list(dict.fromkeys(markers))[:6]
        A = np.array(json.loads(regs.loc[p.roi].affine)); st = G.acquisition(p.slide_id, p.roi); slide_rois = P[P.slide_id == p.slide_id].roi.unique()
        ifimg = np.stack([if_frame(p.slide_id, int(regs.loc[p.roi].frame), ch) for ch in range(4)], -1); HW = G.HW
        x0, y0 = int(c.x), int(c.y)
        q = np.clip(np.pad(ifimg, ((HW, HW), (HW, HW), (0, 0)))[y0:y0 + 2 * HW, x0:x0 + 2 * HW] / (np.percentile(ifimg.reshape(-1, 4), 99.9, axis=0) + 1e-6), 0, 1)
        warp = lambda im, nn=False: np.pad(cv2.warpAffine(im.astype(np.float32), np.linalg.inv(A)[:2], (ifimg.shape[1], ifimg.shape[0]),
                                                          flags=(cv2.INTER_NEAREST if nn else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP), HW)[y0:y0 + 2 * HW, x0:x0 + 2 * HW]
        tm, _ = G.target_mask(p.roi, int(p.source_row), int(round(c.imc_row)), int(round(c.imc_col)))
        ol = find_boundaries(warp(tm.astype(np.float32), nn=True) > .5, mode='outer')
        ax = axes[ri]
        ax[0].imshow(im40); ax[0].set_title('40x', fontsize=6)
        ax[1].imshow(np.clip(np.dstack([q[..., 1] + q[..., 3], q[..., 2] + q[..., 3], q[..., 0] + q[..., 3]]), 0, 1)); ax[1].set_title('10x (SPECTRA)', fontsize=6)
        for a, m in zip(ax[2:], ['DNA'] + markers):
            lo, hi = G.scale(p.slide_id, m, slide_rois); w = np.clip((warp(st[m]) - lo) / (hi - lo + 1e-6), 0, 1)
            g = np.repeat(w[..., None], 3, -1); g[ol] = [0, .8, 1]; a.imshow(g, interpolation='bilinear')
            call = '' if m in ('DNA', 'CD45') else (' +' if pr.get(f'{m}_pos') in (True, 'True') else ' -')
            a.set_title(m + call, fontsize=6)
        ax[0].set_ylabel(f'#{G.NUM[int(p.cluster)]} {G.ctype(p.donor_id)}\n{p.split} r{int(p.positive_rank)}', fontsize=6, rotation=0, ha='right', va='center')
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
    fig.subplots_adjust(left=.1, right=.995, top=.985, bottom=.005, wspace=.05, hspace=.3)
    out = G.AN / 'imc_40x_comparison_private.png'; fig.savefig(out, dpi=150); print('saved', out)


if __name__ == '__main__':
    main()
