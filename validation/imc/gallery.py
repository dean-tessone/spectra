"""Gallery of selected high-attention SPECTRA cells with registered IMC (exploratory: a sample of
high-attribution cells from several clusters on a small set of patients, no statistics). Run with the OpenIMC venv.

Display: no IMC composites; single markers only. IMC panels show RAW ion counts (no hot-pixel
removal; that is applied only for segmentation and features), arcsinh-transformed (cofactor 1) and contrast-scaled with
ONE fixed window per marker per slide (low = slide median pixel, high = 99.9th percentile of pixels over all of that
slide's linked ROIs), so a dark panel means the marker is low relative to that slide's own positive cells. Panels are
warped into the IF frame orientation; the matched segmented cell is outlined. Labels give the call from
segment_openimc.py v3 (+ = cell arcsinh mean >= 1 above off-cell background; else -).
Selection (slide, attribution rank) in SELECT; no patient identifiers are shown.
  python gallery.py review   -> one contact sheet of every segmented linked cell (private, for selection)
  python gallery.py          -> gallery of SELECT"""
from pathlib import Path
import sys, json, re
import numpy as np, pandas as pd, tifffile, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from readimc import MCDFile
from openimc.processing.denoising import apply_channel_denoise
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from register import if_frame, IMC  # noqa: E402
import segment_openimc as so  # noqa: E402
AN = REPO / 'analyses/imc_linkage_20261111'; RUN = REPO / 'runs/imc_linkage_20261111'
MASKS = RUN / 'openimc'; MANUAL = RUN / 'manual_masks'
NUM = pd.read_csv(REPO / 'analyses/figure3_locked6_attribution_20261111/leiden/cluster_display_numbers.csv').set_index('cluster').number
FIXED = ['CK', 'EpCAM', 'Vimentin', 'CD45', 'CD31']
DEN = None   # display raw ion counts (no hot-pixel removal); hot-pixel removal is used for segmentation/features only
CT = {'055048': 'Prostate', '0B8D301': 'Prostate', '0B86902': 'Prostate', '0A73E07': 'Prostate', '0A73E08': 'Prostate', '0750910': 'Breast', '075469': 'Breast',
      '0A63004': 'Breast', '09E3D02': 'Breast', '0A90903': 'Breast', '0A78F08': 'Breast', '0BAB902': 'Pancreatic', '0BB2402': 'Pancreatic', '0BB7401': 'Pancreatic',
      '0B97A01': 'Pancreatic', '078708': 'Prostate', '0ADD609': 'Control', '0ADD705': 'Control', '0ADD708': 'Control', '0533210': 'Prostate', '059588': 'Prostate', '0A9AC01': 'Prostate'}
# (slide, attribution rank, two row-specific markers)
SELECT = [('055048', 3, ['AR', 'PSMA']), ('0750910', 173, ['ER', 'HER2']),          # CK+ EpCAM+ epithelial (#17)
          ('055048', 4, ['AR', 'PSMA']), ('0750910', 113, ['ER', 'Ecad']),          # CK+ EpCAM- epithelial (#21)
          ('09E3D02', 71, ['CD14', 'CD68']),                                       # Vim+ CD31+ CD45- endothelial-like (#10)
          ('075469', 18, ['CD68', 'CD14']), ('075469', 31, ['CD14', 'CD44']),      # Vim+ myeloid-like (#10)
          ('0A78F08', 114, ['CD14', 'CD68']),                                      # CD14+ CD68+ monocyte (#22)
          ('0A63004', 165, ['CD3', 'CD45RO']), ('0B8D301', 43, ['CD45RA', 'CD3']), # lymphocytes (#16)
          ('075469', 10, ['CD14', 'CD68'])]                                        # IF CK-bright, IMC Vim+ CD31+ (#16, discordant)
HW = 30   # half-width of the crop in 10x IF pixels (0.59 um/px, 17.7 um)
X40_HALF = 98   # matching 40x half-width (0.18 um/px)
_A, _SCALE = {}, {}
_LEDGER = pd.read_csv(REPO / 'cohort/reference_policy/population_patient_ledger.csv', dtype=str).set_index('donor_id')


def ctype(donor):
    """Cancer type or 'Control' from the study ledger (replaces a hand-typed slide map that had errors, 2026-11-12)."""
    r = _LEDGER.loc[donor]
    return 'Control' if r.label == '0' else r.cancer_types


def acquisition(sid, roi):
    """Denoised IMC stack (C,H,W) and marker -> channel index for this ROI, from the MCD (verified by Well name)."""
    if roi not in _A:
        frame_cell = '_'.join(roi.split('_')[1:3]); acq_n = int(re.search(r'_(\d+)\.(txt|npz)$', roi).group(1))
        with MCDFile(IMC / sid / f'{sid}.mcd') as f:
            hits = [a for sl in f.slides for a in sl.acquisitions if str(a.description) == frame_cell]
            if len(hits) > 1:
                hits = [a for a in hits if int(a.id) == acq_n] or hits
            assert len(hits) == 1, roi
            a = hits[0]; img = f.read_acquisition(a); names = [f'{l}_{n}' for l, n in zip(a.channel_labels, a.channel_names)]
        idx = {}
        for k, alts in so.KEY + [('DNA', ['dna1', 'dna2'])]:
            ix = [i for i, n in enumerate(names) if so.norm(n) in alts]
            if ix:
                idx[k] = ix
        den = {i: (apply_channel_denoise(img[i].astype(np.float32), DEN) if DEN else img[i].astype(np.float32)) for v in idx.values() for i in v}
        _A[roi] = {k: np.arcsinh(sum(den[i] for i in ix)) for k, ix in idx.items()}
    return _A[roi]


def scale(sid, marker, rois):
    """One fixed display window per marker per slide."""
    if (sid, marker) not in _SCALE:
        px = np.concatenate([acquisition(sid, r)[marker].ravel() for r in rois if marker in acquisition(sid, r)])
        _SCALE[(sid, marker)] = (float(np.percentile(px, 50)), float(np.percentile(px, 99.9)))
    return _SCALE[(sid, marker)]


def if_stack(sid, frame):
    """10x IF frame (H, W, 4); slides without raw 10x frames (40x-registered, link40.py) get a blank frame of the
    standard size so IMC can still be warped into 10x frame coordinates (the 40x image is shown instead)."""
    try:
        return np.stack([if_frame(sid, frame, ch) for ch in range(4)], -1)
    except Exception:
        return np.zeros((1004, 1362, 4), np.float32)


def target_mask(roi, source_row, r0, c0):
    m = MANUAL / f'{roi.rsplit(".", 1)[0]}_{source_row}.npy'
    if m.exists():
        return np.load(m), 'manual'
    mask = tifffile.imread(MASKS / roi.rsplit('.', 1)[0] / 'cells_expanded3.tif')
    lab = mask[r0, c0] if 0 <= r0 < mask.shape[0] and 0 <= c0 < mask.shape[1] else 0
    return (mask == lab) if lab else None, 'cellpose'


def render(cells, out, extras_for, title_rows=True, dpi=220):
    L = pd.concat([pd.read_parquet(p) for p in AN.glob('linked_cells*_private.parquet')], ignore_index=True)
    regs = pd.concat([pd.read_csv(p, dtype={'slide_id': str}) for p in AN.glob('roi_registration*.csv')]).drop_duplicates('file').set_index('file')
    P = pd.read_csv(AN / 'imc_profiles_private.csv', dtype={'slide_id': str})
    ncol = 2 + len(FIXED) + 2
    fig, axes = plt.subplots(len(cells), ncol, figsize=(0.76 * ncol, 0.84 * len(cells)), squeeze=False)
    for ri, (sid, rk) in enumerate(cells):
        p = P[(P.slide_id == sid) & (P.positive_rank == rk)].iloc[0]; roi = p.roi
        c = L[(L.roi == roi) & (L.source_row == p.source_row)].iloc[0]; A = np.array(json.loads(regs.loc[roi].affine))
        st = acquisition(sid, roi); slide_rois = P[P.slide_id == sid].roi.unique()
        ifimg = if_stack(sid, int(regs.loc[roi].frame))
        x0, y0 = int(c.x), int(c.y)
        crop = np.pad(ifimg, ((HW, HW), (HW, HW), (0, 0)))[y0:y0 + 2 * HW, x0:x0 + 2 * HW]
        q = np.clip(crop / (np.percentile(ifimg.reshape(-1, 4), 99.9, axis=0) + 1e-6), 0, 1)
        warp = lambda im, nn=False: np.pad(cv2.warpAffine(im.astype(np.float32), np.linalg.inv(A)[:2], (ifimg.shape[1], ifimg.shape[0]),
                                                          flags=(cv2.INTER_NEAREST if nn else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP), HW)[y0:y0 + 2 * HW, x0:x0 + 2 * HW]
        tm, how = target_mask(roi, int(p.source_row), int(round(c.imc_row)), int(round(c.imc_col)))
        outline = find_boundaries(warp(tm.astype(np.float32), nn=True) > .5, mode='outer') if tm is not None else None
        ax = axes[ri]
        ax[0].imshow(np.clip(np.dstack([q[..., 1] + q[..., 3], q[..., 2] + q[..., 3], q[..., 0] + q[..., 3]]), 0, 1), interpolation='lanczos')
        for a, m in zip(ax[1:], ['DNA'] + FIXED + list(extras_for(sid, rk, p))):
            if m not in st:
                a.set_facecolor('#E6E6E6'); a.text(.5, .5, f'{m}\nnot in panel', ha='center', va='center', fontsize=4.5, transform=a.transAxes); continue
            lo, hi = scale(sid, m, slide_rois); w = np.clip((warp(st[m]) - lo) / (hi - lo + 1e-6), 0, 1)
            rgb = np.repeat(w[..., None], 3, -1)
            if outline is not None:
                rgb[outline] = [0.0, 0.85, 1.0]
            a.imshow(rgb, interpolation='bilinear')
            call = '' if m in ('DNA', 'CD45') else (' +' if p.get(f'{m}_pos') is True or p.get(f'{m}_pos') == 'True' else ' −')
            a.set_title(m + call, fontsize=5.5, pad=1.5, color='black')
        lab = f"#{NUM[int(c.cluster)]} {ctype(p.donor_id)}\n{p.split}, rank {int(rk)}"
        if how == 'manual':
            lab += '\nmanual mask'
        if ri == len(cells) - 1:   # 10 um scale bar (10x 0.59 um/px)
            ax[0].plot([2 * HW - 4 - 10 / .59, 2 * HW - 4], [2 * HW - 4] * 2, color='white', lw=1.2)
        ax[0].set_ylabel(lab, fontsize=5.5, rotation=0, ha='right', va='center', labelpad=3)
        if ri == 0:
            ax[0].set_title('IF 10x', fontsize=5.5, pad=1.5)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(.3)
    fig.subplots_adjust(left=.13, right=.995, top=1 - .12 / len(cells), bottom=.005, wspace=.04, hspace=.28)
    fig.savefig(out, dpi=dpi); plt.close(fig); print('saved', out)


def informative(sid, rk, p, k=2):
    """Row-specific markers: positive calls outside the fixed set first, then the highest-percentile ones."""
    cand = [m for m, _ in so.KEY if m not in FIXED + ['CD45RO'] and pd.notna(p.get(f'{m}_pct'))]
    pos = [m for m in cand if p.get(f'{m}_pos') in (True, 'True')]
    rest = sorted([m for m in cand if m not in pos], key=lambda m: -p[f'{m}_pct'])
    return (pos + rest)[:k]


# Manuscript figure (Supplementary Fig. 4), revised 2026-11-12: IMC markers that are also stained in IF (CK,
# vimentin, CD45) are NOT shown, because the IF antibodies may block IMC antibody binding; CD31 is kept, then markers
# not in the IF panel. Only slides with usable IMC marker signal (imc_signal_by_roi_private.csv; 0A90903, 0B8D301,
# 0B97A01, 0BB7401, 0BB2402, 0BAB902 excluded). 40x re-image shown where one exists (compare_40x.forty).
# (slide, attribution rank, IMC markers after DNA, provisional label; the figure label is re-derived from the v3 calls)
FIGURE = [  # chosen after reviewing all linked cells (review sheets v4b); label None = derived from calls
    ('0A9AC01', 108, ['CD31', 'EpCAM', 'CD44', 'CD14', 'CD68', 'CD3'], None),                 # [60] CK-bright at 40x, EpCAM+
    ('055048', 3, ['CD31', 'EpCAM', 'CD44', 'AR', 'PSMA', 'CD14'], None),                     # [46]
    ('0A9AC01', 266, ['CD31', 'EpCAM', 'CD68', 'CD14', 'CD61', 'CD3'], 'EpCAM+ cell beside a CD68+ cell'),  # [42] doublet
    ('0A9AC01', 194, ['CD31', 'CD14', 'CD68', 'EpCAM', 'CD3', 'CD61'], None),                 # [24] elongated, CD31+
    ('0A63004', 93, ['CD31', 'CD14', 'CD68', 'CD3', 'SMA', 'FAP'], None),                     # [17]
    ('059588', 117, ['CD31', 'CD68', 'CD14', 'Ecad', 'EpCAM', 'CD3'], None),                  # [4]
    ('0ADD705', 15, ['CD31', 'CD68', 'CD14', 'CD66b', 'CD3', 'CD45RA'], None),                # [25] reference control
    ('0A78F08', 114, ['CD31', 'CD14', 'CD68', 'CD66b', 'CD3', 'CD45RA'], None),               # [67]
    ('0A63004', 165, ['CD31', 'CD3', 'CD45RA', 'CD45RO', 'CD14', 'CD68'], None),              # [40]
    ('0ADD705', 187, ['CD31', 'CD45RA', 'CD3', 'CD14', 'CD68', 'CD66b'], None),               # [44] reference control
    ('0ADD708', 239, ['CD31', 'CD61', 'CD14', 'CD68', 'CD3', 'CD45RA'], 'CD31+ CD61+ (platelet-like aggregate)')]  # [27] reference control
FIG_TILES = AN / 'figure_tiles'


def export_tiles():
    """Denoised, registered, contrast-fixed tiles for the manuscript figure (no patient or slide identifiers)."""
    FIG_TILES.mkdir(exist_ok=True)
    for f in FIG_TILES.glob('cell*.npz'):
        f.unlink()
    L = pd.concat([pd.read_parquet(p) for p in AN.glob('linked_cells*_private.parquet')], ignore_index=True)
    regs = pd.concat([pd.read_csv(p, dtype={'slide_id': str}) for p in AN.glob('roi_registration*.csv')]).drop_duplicates('file').set_index('file')
    P = pd.read_csv(AN / 'imc_profiles_private.csv', dtype={'slide_id': str})
    meta, private = [], []
    for i, (sid, rk, markers, pheno) in enumerate(FIGURE):
        p = P[(P.slide_id == sid) & (P.positive_rank == rk)].iloc[0]; roi = p.roi
        c = L[(L.roi == roi) & (L.source_row == p.source_row)].iloc[0]; A = np.array(json.loads(regs.loc[roi].affine))
        st = acquisition(sid, roi); slide_rois = P[P.slide_id == sid].roi.unique()
        ifimg = if_stack(sid, int(regs.loc[roi].frame))
        x0, y0 = int(c.x), int(c.y)
        q = np.clip(np.pad(ifimg, ((HW, HW), (HW, HW), (0, 0)))[y0:y0 + 2 * HW, x0:x0 + 2 * HW] / (np.percentile(ifimg.reshape(-1, 4), 99.9, axis=0) + 1e-6), 0, 1)
        warp = lambda im, nn=False: np.pad(cv2.warpAffine(im.astype(np.float32), np.linalg.inv(A)[:2], (ifimg.shape[1], ifimg.shape[0]),
                                                          flags=(cv2.INTER_NEAREST if nn else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP), HW)[y0:y0 + 2 * HW, x0:x0 + 2 * HW]
        tm, how = target_mask(roi, int(p.source_row), int(round(c.imc_row)), int(round(c.imc_col)))
        arrs = dict(IF=np.clip(np.dstack([q[..., 1] + q[..., 3], q[..., 2] + q[..., 3], q[..., 0] + q[..., 3]]), 0, 1).astype(np.float32),
                    outline=find_boundaries(warp(tm.astype(np.float32), nn=True) > .5, mode='outer'))
        calls = {}
        for m in ['DNA'] + markers:
            assert m in st, (sid, m)
            lo, hi = scale(sid, m, slide_rois); arrs[m] = np.clip((warp(st[m]) - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
            if m != 'DNA':
                calls[m] = dict(pct=round(float(p[f'{m}_pct']), 1), asinh=round(float(p[f'{m}_asinh']), 3), background_asinh=round(float(p[f'{m}_bg']), 3),
                                delta=round(float(p[f'{m}_delta']), 3), positive=bool(p[f'{m}_pos'] in (True, 'True')))
        from compare_40x import forty
        im40 = forty(sid, int(c.frame_id), int(c.x), int(c.y))
        if im40 is not None:
            arrs['X40'] = im40.astype(np.float32)
        pos = [m for m in markers if calls[m]['positive']]
        pheno = pheno or (' '.join(f'{m}+' for m in pos) if pos else 'No non-IF marker detected')   # derived from the calls unless set
        np.savez_compressed(FIG_TILES / f'cell{i + 1:02d}.npz', **arrs)
        meta.append(dict(cell=i + 1, cluster_display=int(NUM[int(c.cluster)]), cancer_type=ctype(p.donor_id), split=p.split, attribution_rank=int(rk), match=p.match,
                         mask=how, phenotype=pheno, markers=markers, calls=calls, has_40x=im40 is not None))
        private.append(dict(cell=i + 1, slide_id=sid, donor_id=p.donor_id, roi=roi, source_row=int(p.source_row)))
    (FIG_TILES / 'tiles.json').write_text(json.dumps(dict(um_per_px=0.59, um_per_px_40x=0.18, imc_um_per_px=1.0, half_width_40x_px=X40_HALF, half_width_px=HW, display=__doc__.split('Display')[1].split('Selection')[0].strip(), cells=meta), indent=1))
    pd.DataFrame(private).to_csv(AN / 'figure_tiles_private.csv', index=False); print('exported', len(meta), 'cells')


IF_MARKERS = {'CK', 'Vimentin', 'CD45'}   # stained in IF; IMC binding may be blocked - not shown
GOOD_SLIDES = ['055048', '0750910', '075469', '09E3D02', '0A63004', '0A73E07', '0A73E08', '0A78F08', '0B86902', '078708', '0ADD705', '0ADD708', '059588', '0A9AC01']


def review_all(per_page=10):
    """Numbered review sheets of EVERY linked high-attention cell.
    Row: 40x (if any), 10x IF, IMC DNA, CD31 and the 6 non-IF markers with the largest arcsinh difference over
    off-cell background (v4 features). Tile titles: marker, call (segment_openimc.call_positive), cell mean counts/pixel (off-cell background).
    Unsegmented cells are shown at their registered position (crosshair, no outline). Raw-count display."""
    from compare_40x import forty
    L = pd.concat([pd.read_parquet(p) for p in AN.glob('linked_cells*_private.parquet')], ignore_index=True)
    regs = pd.concat([pd.read_csv(p, dtype={'slide_id': str}) for p in AN.glob('roi_registration*.csv')]).drop_duplicates('file').set_index('file')
    P = pd.read_csv(AN / 'imc_profiles_private.csv', dtype={'slide_id': str})
    P = P.assign(num=P.cluster.map(NUM), good=P.slide_id.isin(GOOD_SLIDES)).sort_values(['good', 'num', 'slide_id', 'positive_rank'], ascending=[False, True, True, True]).reset_index(drop=True)
    P['review_id'] = np.arange(1, len(P) + 1)
    P[['review_id', 'slide_id', 'donor_id', 'roi', 'source_row', 'positive_rank', 'split', 'cluster', 'num', 'match', 'good']].to_csv(AN / 'review_index_private.csv', index=False)
    ncol = 10
    for page in range(0, len(P), per_page):
        sub = P.iloc[page:page + per_page]
        fig, axes = plt.subplots(len(sub), ncol, figsize=(1.0 * ncol + .9, 1.08 * len(sub)), squeeze=False)
        for ri, p in enumerate(sub.itertuples()):
            ax = axes[ri]; roi = p.roi; sid = p.slide_id
            c = L[(L.roi == roi) & (L.source_row == p.source_row)].iloc[0]; A = np.array(json.loads(regs.loc[roi].affine))
            st = acquisition(sid, roi); slide_rois = P[P.slide_id == sid].roi.unique()
            ifimg = if_stack(sid, int(regs.loc[roi].frame)); x0, y0 = int(c.x), int(c.y)
            q = np.clip(np.pad(ifimg, ((HW, HW), (HW, HW), (0, 0)))[y0:y0 + 2 * HW, x0:x0 + 2 * HW] / (np.percentile(ifimg.reshape(-1, 4), 99.9, axis=0) + 1e-6), 0, 1)
            warp = lambda im, nn=False: np.pad(cv2.warpAffine(im.astype(np.float32), np.linalg.inv(A)[:2], (ifimg.shape[1], ifimg.shape[0]),
                                                              flags=(cv2.INTER_NEAREST if nn else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP), HW)[y0:y0 + 2 * HW, x0:x0 + 2 * HW]
            seg = p.match in ('inside nucleus', 'inside expanded cell')
            tm = target_mask(roi, int(p.source_row), int(round(c.imc_row)), int(round(c.imc_col)))[0] if seg else None
            ol = find_boundaries(warp(tm.astype(np.float32), nn=True) > .5, mode='outer') if tm is not None else None
            im40 = forty(sid, int(c.frame_id), int(c.x), int(c.y))
            if im40 is not None:
                ax[0].imshow(im40)
            else:
                ax[0].set_facecolor('#EEEEEE'); ax[0].text(.5, .5, 'no 40x', ha='center', va='center', fontsize=6, transform=ax[0].transAxes)
            if ifimg.any():
                ax[1].imshow(np.clip(np.dstack([q[..., 1] + q[..., 3], q[..., 2] + q[..., 3], q[..., 0] + q[..., 3]]), 0, 1))
            else:
                ax[1].set_facecolor('#EEEEEE'); ax[1].text(.5, .5, 'no 10x frame\n(40x-registered)', ha='center', va='center', fontsize=5.5, transform=ax[1].transAxes)
            row = P.loc[p.Index]
            cand = [m for m, _ in so.KEY if m not in IF_MARKERS | {'CD31'} and m in st and pd.notna(row.get(f'{m}_delta'))]
            cand = sorted(cand, key=lambda m: -row[f'{m}_delta'])[:6]
            if not cand:   # unsegmented: no features; show a fixed non-IF set by eye
                cand = [m for m in ['EpCAM', 'CD14', 'CD68', 'CD3', 'CD45RA', 'CD44', 'CD16', 'CD66b'] if m in st][:6]
            for a, m in zip(ax[2:], ['DNA', 'CD31'] + cand):
                if m not in st:
                    a.set_facecolor('#EEEEEE'); a.text(.5, .5, f'{m}\nnot in panel', ha='center', va='center', fontsize=5.5, transform=a.transAxes); continue
                lo, hi = scale(sid, m, slide_rois); w = np.clip((warp(st[m]) - lo) / (hi - lo + 1e-6), 0, 1); g = np.repeat(w[..., None], 3, -1)
                if ol is not None:
                    g[ol] = [0, .8, 1]
                else:
                    g[HW, HW - 6:HW - 2] = g[HW, HW + 3:HW + 7] = g[HW - 6:HW - 2, HW] = g[HW + 3:HW + 7, HW] = [1, .3, .3]
                a.imshow(g, interpolation='bilinear')
                t = m if m == 'DNA' or pd.isna(row.get(f'{m}_asinh')) else f"{m} {'+' if row[f'{m}_pos'] in (True, 'True') else '-'} {np.sinh(row[f'{m}_asinh']):.2f} ({np.sinh(row[f'{m}_bg']):.2f})"
                a.set_title(t, fontsize=6, pad=2)
            qual = '' if p.good else '\nLOW IMC SIGNAL'
            ax[0].set_ylabel(f"[{p.review_id}]  #{p.num} {ctype(row.donor_id)}\n{p.split}, rank {p.positive_rank}\n{p.match}{qual}", fontsize=6.5, rotation=0, ha='right', va='center', labelpad=4)
            if ri == 0:
                ax[0].set_title('40x', fontsize=6, pad=2); ax[1].set_title('10x IF (SPECTRA)', fontsize=6, pad=2)
            for a in ax:
                a.set_xticks([]); a.set_yticks([])
        fig.subplots_adjust(left=.13, right=.995, top=.985, bottom=.005, wspace=.05, hspace=.32)
        out = AN / f'imc_review_all_{page // per_page + 1}_private.png'; fig.savefig(out, dpi=140); plt.close(fig); print('saved', out)


def main(mode):
    if mode == 'export':
        return export_tiles()
    if mode == 'review_all':
        return review_all()
    P = pd.read_csv(AN / 'imc_profiles_private.csv', dtype={'slide_id': str})
    if mode == 'review':
        P = P[P.match.isin(['inside nucleus', 'inside expanded cell'])].assign(num=lambda d: d.cluster.map(NUM)).sort_values(['num', 'slide_id', 'positive_rank'])
        cells = list(zip(P.slide_id, P.positive_rank))
        for i in range(0, len(cells), 16):
            render(cells[i:i + 16], AN / f'imc_review_sheet_{i // 16 + 1}_private.png', informative, dpi=170)
    else:
        ex = {(s, r): e for s, r, e in SELECT}
        render([(s, r) for s, r, _ in SELECT], AN / 'imc_gallery_draft.png', lambda s, r, p: ex[(s, r)])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'final')
