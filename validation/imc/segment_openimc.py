"""OpenIMC (headless, OpenIMC v1.1.3 in its own environment) segmentation and per-cell features for registered
ROIs that contain a linked high-attention (mass-90) SPECTRA cell. Run with the OpenIMC venv python. Read-only on the IMC drive.

Segmentation (optimized on 5 test ROIs with known target cells, 2026-11-11): OpenIMC hot-pixel removal (median3) on all
channels, Cellpose nuclei model on DNA1+DNA2 (OpenIMC --arcsinh image preprocessing, cell-probability threshold -3) at
diameter 7 px and 14 px, merged (14-px nuclei added where they overlap existing ones by <=20%), expanded by 3 px
(= 3 um) to include cytoplasm. Outlier-only hot-pixel removal (n_sd_local_median) was compared and rejected
(seg_compare_hotpixel.py: 4640 vs 7508 nuclei).

Featurization (v4, arcsinh normalisation, precise), per ROI, computed here from the MCD pixels:
  1. Hot-pixel removal: OpenIMC's outlier-only step (processing.denoising._apply_hot_pixel_step, n_sd_local_median,
     N = 5: a pixel above its 8-neighbour median + 5 x 1.4826 x MAD and above all neighbours is replaced by that
     median). NOT OpenIMC's apply_channel_denoise / extract-features --denoise: its _restore_output_dtype rescales
     every filtered channel so its maximum equals the original maximum, i.e. multiplies each channel of each ROI by
     (raw max / filtered max) - ~10x for DNA and ~100x for EpCAM in one ROI - which corrupts absolute values and any
     arcsinh transform (v3 features_asinh.csv are therefore not used; ROI-internal ranks were unaffected). median3 is
     also avoided for features because it erases sparse true IMC counts (cell means are often <1 count/pixel).
  2. Per cell and channel: mean hot-pixel-removed ion counts over the cells_expanded3 mask (label = OpenIMC cell_id,
     verified), then arcsinh(mean / 1) (cofactor 1, the usual IMC convention).
  3. Background per channel: arcsinh(mean hot-pixel-removed counts / 1) over pixels >=2 px outside every cell mask.
  4. Per linked cell and marker: asinh, bg, delta = asinh - bg, pct = percentile among the ROI's segmented cells.
     Call (v4b, `call_positive`): the cell's total counts (mean x area) exceed the off-cell background expectation
     over the same area beyond Poisson counting noise (one-sided p < 0.001), AND mean >= 3 x background, AND
     mean >= 0.3 counts/pixel (stricter than a first 2x / 0.1 version, which called faint spillover from
     neighbouring cells inside the 3-um expansion). Independent of how many cells carry the marker (the v1/v2 >=90th pct and >=3x-median
     rule failed for abundant markers such as CD16/CD45; a fixed arcsinh-delta threshold failed for low counts).
Morphology (area) from OpenIMC extract-features (unaffected by the rescaling).
Linked-cell match quality: inside nucleus / inside expanded cell / nearest within 4 um / not segmented. A cell lying
in two overlapping ROIs keeps its best-matched ROI."""
import os
from pathlib import Path
import sys, json, re, subprocess
import numpy as np, pandas as pd, tifffile
from skimage.segmentation import expand_labels
from scipy.ndimage import binary_dilation
COFACTOR = 1.0
REPO = Path(__file__).resolve().parents[2]
OI = Path(os.environ.get('OPENIMC', 'openimc'))
IMC = Path(os.environ.get('SPECTRA_IMC_ROOT', 'raw/imc')); CACHE = REPO / 'runs/imc_linkage_20261111/mcd_cache'
OUT = REPO / 'runs/imc_linkage_20261111/openimc'; OUT.mkdir(parents=True, exist_ok=True)
DENOISE = ['--denoise', 'all', '--denoise-method', 'median3']   # OpenIMC hot-pixel removal, 3x3 median (outlier-only n_sd_local_median tested 2026-11-12: 4640 vs 7508 nuclei, loses 23/42 targets; seg_compare_hotpixel.py)
AN = REPO / 'analyses/imc_linkage_20261111'
CYTO = ['Cytokeratin_8_18', 'CK8_and_CK18', 'CK8_18', 'Vimentin', 'CD45', 'CD31', 'CD45RA', 'CD45RO']   # segmentation channels
KEY = [('CK', ['panck', 'pankeratin', 'cytokeratin818', 'ck8and18', 'ck8andck18', 'ck8ck18', 'ck818', 'ck7']), ('EpCAM', ['epcam', 'epcamcd326']), ('Vimentin', ['vimentin']),
       ('CD45', ['cd45']), ('CD45RA', ['cd45ra']), ('CD45RO', ['cd45ro']), ('CD31', ['cd31']), ('CD61', ['cd61', 'integrinbeta3']), ('CD14', ['cd14']),
       ('CD66b', ['cd66', 'cd66b', 'cd66bceacam']), ('CD3', ['cd3']), ('CD68', ['cd68']), ('AR', ['ar']), ('PSMA', ['psmafolh1naaladase1', 'psma']),
       ('Ki67', ['ki67']), ('HER2', ['her2', 'her2cerbb2', 'erbb2']), ('ER', ['er', 'era']), ('CD44', ['cd44']), ('Ncad', ['ncadherin']),
       ('Ecad', ['ecadherincd324', 'ecadherin']), ('SMA', ['sma']), ('FAP', ['fap']), ('CD16', ['cd16'])]


def norm(ch):
    """Marker base name from an OpenIMC channel name ('191Ir_DNA1_Ir191', 'CD45_Y89', '89Y_CD45_Y89')."""
    b = re.sub(r'^\d+[A-Z][a-z]?_', '', ch); b = re.sub(r'_[A-Z][a-z]?\d+$', '', b)
    return re.sub(r'[^a-z0-9]', '', b.lower())


_ACQ = {}


def find_acquisition(mcd, frame_cell, acq_n):
    """OpenIMC acquisition id ('slide_{i}_acq_{j}' by position) whose description equals the ROI's frame_cell;
    ties broken by the MCD acquisition id matching the ROI's acquisition number."""
    from readimc import MCDFile
    if mcd not in _ACQ:
        rows = []
        with MCDFile(mcd) as f:
            for i, sl in enumerate(f.slides):
                for j, a in enumerate(sl.acquisitions):
                    rows.append((f'slide_{i}_acq_{j}', str(a.description), int(a.id), [f'{l}_{n}' for l, n in zip(a.channel_labels, a.channel_names)]))
        _ACQ[mcd] = rows
    hits = [r for r in _ACQ[mcd] if r[1] == frame_cell]
    if len(hits) > 1:
        hits = [r for r in hits if r[2] == acq_n] or hits
    return (hits[0][0], hits[0][3]) if len(hits) == 1 else (None, None)


def merge(small, large, maxov=.2):
    """Keep all 7-px nuclei; add 14-px (large) nuclei that overlap existing ones by <=20%."""
    out = small.copy(); nxt = out.max() + 1
    for lab in np.unique(large[large > 0]):
        reg = large == lab
        if (small[reg] > 0).mean() <= maxov:
            out[reg & (out == 0)] = nxt; nxt += 1
    return out


def call_positive(mean, bg, area, alpha=1e-3):
    """Marker call on hot-pixel-removed ion counts (see module docstring)."""
    from scipy.stats import poisson
    if not np.isfinite(mean) or not np.isfinite(area) or area <= 0:
        return None
    lam = max(bg, 1e-3) * area
    return bool(poisson.sf(np.ceil(mean * area) - 1, lam) < alpha and mean >= 3 * bg and mean >= 0.3)


def recall(path=AN / 'imc_profiles_private.csv'):
    """Re-apply call_positive to an existing profile table (no image access)."""
    P = pd.read_csv(path, dtype={'slide_id': str})
    for c in [c for c in P.columns if c.endswith('_asinh')]:
        k = c[:-6]
        P[f'{k}_pos'] = [call_positive(np.sinh(v) * COFACTOR, np.sinh(b) * COFACTOR, a) if pd.notna(v) else None for v, b, a in zip(P[c], P[f'{k}_bg'], P.area_um2)]
    P.to_csv(path, index=False); return P


def features_v4(mcd, acq, chans, mask):
    """Per-cell arcsinh(mean outlier-hot-pixel-removed counts / COFACTOR) and per-channel off-cell background."""
    from readimc import MCDFile
    from scipy.ndimage import mean as label_mean
    from openimc.processing.denoising import _apply_hot_pixel_step
    i, j = map(int, re.fullmatch(r'slide_(\d+)_acq_(\d+)', acq).groups())
    with MCDFile(mcd) as fh:
        a = fh.slides[i].acquisitions[j]; img = fh.read_acquisition(a)
        names = [f'{l}_{n}' for l, n in zip(a.channel_labels, a.channel_names)]
    assert names == list(chans)
    labs = np.unique(mask[mask > 0]); off = ~binary_dilation(mask > 0, iterations=2)
    feat, bg = {'cell_id': labs}, {}
    for k, n in enumerate(names):
        x = _apply_hot_pixel_step(img[k].astype(np.float32), {'method': 'n_sd_local_median', 'n_sd': 5.0})
        feat[f'{n}_counts'] = label_mean(x, mask, labs); feat[f'{n}_asinh'] = np.arcsinh(feat[f'{n}_counts'] / COFACTOR)
        bg[n] = float(np.arcsinh(x[off].mean() / COFACTOR))
    return pd.DataFrame(feat), bg


def channels(roi_file, slide):
    """OpenIMC channel names (label_metal) for this ROI."""
    if roi_file.endswith('.npz'):
        z = np.load(CACHE / roi_file)
        return [f"{re.sub(r'^[0-9]+[A-Za-z]+_', '', str(l))}_{n}" for l, n in zip(z['labels'], z['names'])]
    head = open(IMC / slide / roi_file).readline().rstrip('\n').split('\t')[6:]
    return [re.sub(r'\(([A-Z][a-z]?)(\d+)Di\)', r'_\1\2', h) for h in head]


def main(only=None):
    L = pd.concat([pd.read_parquet(p) for p in AN.glob('linked_cells*_private.parquet')], ignore_index=True)
    if only:
        L = L[L.slide_id.isin(only)]
    regs = pd.concat([pd.read_csv(p) for p in AN.glob('roi_registration*.csv')], ignore_index=True)
    regs = regs[regs.qc_pass == True].drop_duplicates('file').set_index('file')
    H = L[L.mass90.fillna(False).astype(bool) & L.roi.isin(regs.index)]
    rows = []
    for roi, g in H.groupby('roi'):
        sid = g.slide_id.iloc[0]; acq_n = int(re.search(r'_(\d+)\.(txt|npz)$', roi).group(1)); frame_cell = '_'.join(roi.split('_')[1:3])
        mcd = IMC / sid / f'{sid}.mcd'; d = OUT / roi.rsplit('.', 1)[0]; d.mkdir(exist_ok=True)
        acq, chans = find_acquisition(mcd, frame_cell, acq_n)
        dna = [c for c in (chans or []) if norm(c) in ('dna1', 'dna2')]
        if acq is None:
            print(roi, 'no unique MCD acquisition for', frame_cell, flush=True); continue
        feat = d / 'features.csv'
        try:
          if not feat.exists():
            masks = {}
            for diam in (7, 14):
                sd = d / f'nuclei_d{diam}'
                if not list(sd.glob('*_segmentation_masks.tif')):
                    subprocess.run([str(OI), 'segment', str(mcd), str(sd), '--acquisition', acq, '--method', 'cellpose', '--model', 'nuclei', '--nuclear-channels', ','.join(dna),
                                    '--diameter', str(diam), '--cellprob-threshold', '-3', '--arcsinh', *DENOISE, '--gpu-id', '0'], check=True, capture_output=True)
                masks[diam] = tifffile.imread(next(sd.glob('*_segmentation_masks.tif')))
            nuc = merge(masks[7], masks[14]); tifffile.imwrite(d / 'nuclei_merged.tif', nuc.astype(np.int32))
            cell = expand_labels(nuc, 3); tifffile.imwrite(d / 'cells_expanded3.tif', cell.astype(np.int32))
            subprocess.run([str(OI), 'extract-features', str(mcd), str(feat), '--mask', str(d / 'cells_expanded3.tif'), '--acquisition', acq, '--morphological', '--intensity', *DENOISE],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(roi, 'OpenIMC failed:', (e.stderr or b'')[-300:], flush=True); continue
        f = pd.read_csv(feat); mask = tifffile.imread(d / 'cells_expanded3.tif'); nucm = tifffile.imread(d / 'nuclei_merged.tif')
        well = str(f.source_well.iloc[0])
        ok = well == frame_cell
        fv, bg = features_v4(mcd, acq, chans, mask); fv.to_csv(d / 'features_v4.csv', index=False)
        f = f[['cell_id', 'area_um2', 'centroid_x', 'centroid_y']].merge(fv, on='cell_id', validate='1:1')
        for _, c in g.iterrows():
            r0, c0 = int(round(c.imc_row)), int(round(c.imc_col))
            inb = 0 <= r0 < mask.shape[0] and 0 <= c0 < mask.shape[1]
            lab = int(mask[r0, c0]) if inb else 0
            how = 'inside nucleus' if inb and nucm[r0, c0] > 0 and lab else ('inside expanded cell' if lab else '')
            if lab == 0:
                dd = np.hypot(f.centroid_x - c.imc_col, f.centroid_y - c.imc_row); j = dd.idxmin()
                lab, how = (int(f.cell_id[j]), 'nearest<=4um') if dd[j] <= 4 else (0, 'not segmented')
            row = dict(split=c.get('split', 'test'), slide_id=sid, donor_id=c.donor_id, roi=roi, acquisition_verified=ok, well=well, source_row=int(c.source_row),
                       positive_rank=int(c.positive_rank), top50=bool(c.top50), cluster=int(c.cluster), match=how, roi_cells=len(f),
                       IF_CK=c.TRITC_mean, IF_CD45_CD31=c.CY5_mean, IF_VIM=c.FITC_mean, IF_DAPI=c.DAPI_mean, IF_area=c.area)
            if lab:
                cell = f[f.cell_id == lab].iloc[0]; row.update(area_um2=cell.area_um2)
                for k, alts in KEY:
                    col = next((f'{ch}_asinh' for ch in chans if norm(ch) in alts and f'{ch}_asinh' in f.columns), None)
                    if col:
                        v = float(cell[col]); b = bg[col[:-6]]; pct = float((f[col] < v).mean() * 100)
                        row.update({f'{k}_asinh': v, f'{k}_bg': b, f'{k}_delta': v - b, f'{k}_pct': pct, f'{k}_pos': call_positive(np.sinh(v) * COFACTOR, np.sinh(b) * COFACTOR, cell.area_um2)})
            rows.append(row)
        print(roi, 'well', well, 'verified', ok, 'cells', len(f), flush=True)
    P = pd.DataFrame(rows)
    # a cell can lie in two overlapping ROIs: keep its best-matched ROI (nucleus > expanded cell > nearest > not segmented)
    order = {'inside nucleus': 0, 'inside expanded cell': 1, 'nearest<=4um': 2, 'not segmented': 3}
    P = P.assign(_o=P.match.map(order)).sort_values(['_o', 'roi']).drop_duplicates(['slide_id', 'source_row']).drop(columns='_o')
    P.to_csv(AN / ('imc_profiles_private.csv' if not only else 'pilot_openimc_profiles_private.csv'), index=False)
    print(P.match.value_counts())


if __name__ == '__main__':
    main(sys.argv[1:] or None)
