"""Link IMC ROIs to SPECTRA cells via 40x registration (register40.py) for test slides without raw 10x frames. Per ROI: matching 40x re-image by (frame, cell), coarse NCC + nucleus
ICP; QC pass if >=50% of IMC nuclei inside the 40x field match a 40x nucleus within 3 um and the median residual is
<3 um (validated against the 10x route on 055048: 11/12 ROIs pass, cells agree within a median ~3 um). The stored
'affine' maps IMC pixel -> 10x frame coordinates (composed through the 40x field), so downstream code is unchanged.
Candidate-pool cells of that frame mapped inside the ROI are joined to Figure 3 attribution. Run with repo venv."""
from pathlib import Path
import sys, re, json, time
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import link as LK  # noqa: E402  (attribution table, source receipts, load_roi)
from register40 import find40, load40, register40, refine40, IF_UM  # noqa: E402
SLIDES = ['0533210', '059588', '0A9AC01']


def main():
    regs, linked = [], []
    for sid in SLIDES:
        rois = [f for f in sorted(LK.os.listdir(LK.CACHE)) if re.fullmatch(rf'{sid}_\d+_\d+_\d+\.npz', f)]
        src_all = pd.read_parquet(LK.rec[sid]['source'], columns=['frame_id', 'x', 'y', 'area', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']).reset_index(names='source_row')
        for roi in rois:
            t0 = time.time(); fr, cell, acq = map(int, re.fullmatch(rf'{sid}_(\d+)_(\d+)_(\d+)\.npz', roi).groups())
            r = dict(slide_id=sid, file=roi, frame=fr, acq=acq, route='40x')
            hit = find40(sid, fr, cell)
            if hit is None:
                regs.append(dict(r, error='no 40x for frame/cell', qc_pass=False)); continue
            arr, cols = LK.load_roi(dict(slide_id=sid, file=roi)); dna = arr[[i for i, c in enumerate(cols) if 'DNA' in c]].sum(0)
            d40 = load40(hit[0]); best = register40(dna, d40); A40, diag = refine40(best, dna, d40)
            X, Y = hit[1], hit[2]; H, W = d40.shape; k = IF_UM / diag['um_per_40x_px']
            T = np.array([[1 / k, 0, X - W / 2 / k], [0, 1 / k, Y - H / 2 / k], [0, 0, 1]])   # rotated-40x px -> 10x frame
            A = T @ A40
            qc = bool(diag['matched_frac'] >= .5 and diag['median_residual_um'] < 3)
            regs.append(dict(r, x40=X, y40=Y, coarse_score=best['score'], angle=best['angle'], mirror=best['mirror'], **diag, affine=json.dumps(A.tolist()),
                             qc_pass=qc, seconds=round(time.time() - t0, 1)))
            print(roi, 'QC', qc, round(diag['matched_frac'], 2), round(diag['median_residual_um'], 2), flush=True)
            if not qc:
                continue
            src = src_all[src_all.frame_id == fr]; B = np.linalg.inv(A)
            P = (B @ np.c_[src.x, src.y, np.ones(len(src))].T).T
            src = src.assign(imc_col=P[:, 0], imc_row=P[:, 1])
            src = src[(src.imc_col > 8) & (src.imc_col < arr.shape[2] - 8) & (src.imc_row > 8) & (src.imc_row < arr.shape[1] - 8)]
            if len(src):
                linked.append(src.assign(slide_id=sid, roi=roi))
        pd.DataFrame(regs).to_csv(LK.OUT / 'roi_registration_40x.csv', index=False)
    L = pd.concat(linked, ignore_index=True).merge(LK.cells, on=['slide_id', 'source_row'], how='left')
    L['model_input'] = L.relevance.notna()
    L.to_parquet(LK.OUT / 'linked_cells_40x_private.parquet', index=False)
    print(L.groupby('slide_id').agg(cells=('source_row', 'size'), model_input=('model_input', 'sum'), mass90=('mass90', 'sum')))


if __name__ == '__main__':
    main()
