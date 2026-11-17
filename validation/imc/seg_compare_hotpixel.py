"""Segmentation comparison of OpenIMC hot-pixel removal methods. The current run used OpenIMC
`--denoise all` with its default hot-pixel method (median3: 3x3 median on every pixel). This re-segments the same
ROIs with OpenIMC's outlier-only hot-pixel removal (n_sd_local_median, N SD = 5: only pixels above the local
median + 5 robust SD and above all 8 neighbours are replaced), keeping everything else identical (Cellpose nuclei
d7 + d14 merged, expanded 3 px), and reports whether each linked high-attention cell falls in a segmented cell.
Also records denoised DNA at the target relative to the ROI's segmented nuclei (to separate missed nuclei from
cells absent in IMC). Run with the OpenIMC venv python. Read-only on the IMC drive."""
from pathlib import Path
import sys, re, subprocess
import numpy as np, pandas as pd, tifffile
from skimage.segmentation import expand_labels
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import segment_openimc as so  # noqa: E402
from openimc.processing.denoising import apply_channel_denoise  # noqa: E402
OUT = REPO / 'runs/imc_linkage_20261111/openimc_nsd'; OUT.mkdir(parents=True, exist_ok=True)
AN = REPO / 'analyses/imc_linkage_20261111'


def where(mask, nuc, r0, c0):
    if not (0 <= r0 < mask.shape[0] and 0 <= c0 < mask.shape[1]):
        return 'out of ROI'
    return 'inside nucleus' if nuc[r0, c0] else ('inside expanded cell' if mask[r0, c0] else 'not segmented')


def main():
    P = pd.read_csv(AN / 'high_attention_openimc_profiles_dedup_private.csv', dtype={'slide_id': str})
    L = pd.concat([pd.read_parquet(p) for p in AN.glob('linked_cells*_private.parquet')], ignore_index=True)
    rows = []
    for roi, g in P.groupby('roi'):
        sid = g.slide_id.iloc[0]; acq_n = int(re.search(r'_(\d+)\.(txt|npz)$', roi).group(1)); frame_cell = '_'.join(roi.split('_')[1:3])
        mcd = so.IMC / sid / f'{sid}.mcd'; acq, chans = so.find_acquisition(mcd, frame_cell, acq_n)
        dna = [c for c in chans if so.norm(c) in ('dna1', 'dna2')]
        d = OUT / roi.rsplit('.', 1)[0]; d.mkdir(exist_ok=True); masks = {}
        for diam in (7, 14):
            sd = d / f'nuclei_d{diam}'
            if not list(sd.glob('*_segmentation_masks.tif')):
                subprocess.run([str(so.OI), 'segment', str(mcd), str(sd), '--acquisition', acq, '--method', 'cellpose', '--model', 'nuclei', '--nuclear-channels', ','.join(dna),
                                '--diameter', str(diam), '--cellprob-threshold', '-3', '--arcsinh', '--denoise', 'all', '--denoise-method', 'n_sd_local_median',
                                '--denoise-n-sd', '5', '--gpu-id', '0'], check=True, capture_output=True)
            masks[diam] = tifffile.imread(next(sd.glob('*_segmentation_masks.tif')))
        nuc = so.merge(masks[7], masks[14]); cell = expand_labels(nuc, 3)
        tifffile.imwrite(d / 'nuclei_merged.tif', nuc.astype(np.int32)); tifffile.imwrite(d / 'cells_expanded3.tif', cell.astype(np.int32))
        old = so.OUT / roi.rsplit('.', 1)[0]; onuc = tifffile.imread(old / 'nuclei_merged.tif'); ocell = tifffile.imread(old / 'cells_expanded3.tif')
        # denoised DNA (outlier-only hot-pixel removal) for the presence check
        from readimc import MCDFile
        with MCDFile(mcd) as f:
            a = next(a for sl in f.slides for a in sl.acquisitions if str(a.description) == frame_cell and (a.id == acq_n or True))
            img = f.read_acquisition(a); names = [f'{l}_{n}' for l, n in zip(a.channel_labels, a.channel_names)]
        dimg = sum(apply_channel_denoise(img[i], {'hot': {'method': 'n_sd_local_median', 'n_sd': 5}}) for i, n in enumerate(names) if so.norm(n) in ('dna1', 'dna2'))
        nuc_med = np.median([dimg[nuc == k].mean() for k in np.unique(nuc[nuc > 0])]) if nuc.max() else np.nan
        for _, p in g.iterrows():
            c = L[(L.roi == roi) & (L.source_row == p.source_row)].iloc[0]; r0, c0 = int(round(c.imc_row)), int(round(c.imc_col))
            yy, xx = np.ogrid[:dimg.shape[0], :dimg.shape[1]]; disk = (yy - r0) ** 2 + (xx - c0) ** 2 <= 9
            rows.append(dict(roi=roi, slide_id=sid, positive_rank=int(p.positive_rank), split=p.split, cluster=int(p.cluster),
                             median3=where(ocell, onuc, r0, c0), n_sd_local_median=where(cell, nuc, r0, c0),
                             dna_at_target_vs_nuclei=float(dimg[disk].mean() / nuc_med), nuclei_median3=int(len(np.unique(onuc)) - 1), nuclei_nsd=int(len(np.unique(nuc)) - 1)))
        print(roi, flush=True)
    R = pd.DataFrame(rows); R.to_csv(AN / 'segmentation_hotpixel_comparison_private.csv', index=False)
    print(pd.crosstab(R.median3, R.n_sd_local_median)); print(R.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
