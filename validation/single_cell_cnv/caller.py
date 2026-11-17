"""First-pass, noise-aware copy-number call per sequenced single cell. Read-only on
the QC-approved genomic profile file (SPECTRA_CNV_H5) (one QC-approved profile per cell).

Per cell (autosomes only; each assay keeps its own bin grid):
  log2 ratio r and log2 segment value s, both relative to the cell's autosomal median of segment_mean.
  noise sigma = 1.4826 x MAD(r - s) (bin-level scatter around the segmentation).
  segments = runs of constant segment_mean within a chromosome; a segment is CALLED if it spans >= MIN_BINS bins
  (~10 Mb at ~0.6 Mb bins), |mean s| >= MIN_SHIFT log2, and |mean s| / (sigma / sqrt(bins)) >= MIN_Z.
  altered_fraction = fraction of autosomal bins in called segments; a cell is ALTERED if altered_fraction >=
  MIN_FRAC. Thresholds are calibrated so that known-normal cells (normal blood donors, picked white blood cells,
  cells from reference-control patients) are almost never altered; the false-altered rate on them is reported."""
import os
from pathlib import Path
import numpy as np, pandas as pd, h5py
H5 = Path(os.environ.get('SPECTRA_CNV_H5', 'raw/qc_approved_genomic_profiles.h5'))
MIN_BINS, MIN_SHIFT, MIN_Z, MIN_FRAC = 17, 0.2, 6.0, 0.05
CELL_COLS = ['cell_name', 'slide_id', 'frame_id', 'x', 'y', 'cancer_type', 'patient_id', 'wbc', 'profile_assay', 'profile_row', 'mad_score', 'total_binned_reads']


def cells():
    with h5py.File(H5, 'r') as f:
        c = f['cells']
        return pd.DataFrame({k: (c[k].asstr()[:] if c[k].dtype.kind in 'SO' else c[k][:]) for k in CELL_COLS})


def call_assay(g, min_bins=MIN_BINS, min_shift=MIN_SHIFT, min_z=MIN_Z):
    sm = g['segment_mean'][:].astype(float); ra = g['ratio'][:].astype(float); pos = g['pos_cols'][:]
    auto = pos[:, 0] <= 22; chrom = pos[auto, 0]; sm = sm[:, auto]; ra = ra[:, auto]
    med = np.median(sm, axis=1, keepdims=True)
    s = np.log2(np.clip(sm, 1e-3, None) / med); r = np.log2(np.clip(ra, 1e-3, None) / med)
    sigma = 1.4826 * np.median(np.abs((r - s) - np.median(r - s, axis=1, keepdims=True)), axis=1)
    out = []
    for i in range(len(s)):
        brk = np.r_[0, np.flatnonzero((np.diff(sm[i]) != 0) | (np.diff(chrom) != 0)) + 1, len(chrom)]
        called = np.zeros(len(chrom), bool); events = []
        for a, b in zip(brk[:-1], brk[1:]):
            n = b - a; m = s[i, a:b].mean()
            if n >= min_bins and abs(m) >= min_shift and abs(m) / (sigma[i] / np.sqrt(n) + 1e-9) >= min_z:
                called[a:b] = True; events.append((int(chrom[a]), 'gain' if m > 0 else 'loss', int(n), round(float(m), 3)))
        out.append(dict(profile_row=i, noise_sigma=float(sigma[i]), altered_fraction=float(called.mean()), n_events=len(events),
                        gain_fraction=float(np.mean(called & (s[i] > 0))), loss_fraction=float(np.mean(called & (s[i] < 0))),
                        events=';'.join(f'chr{c}{"+" if t == "gain" else "-"}' for c, t, _, _ in events)))
    return pd.DataFrame(out)


def call_all(**kw):
    C = cells(); parts = []
    with h5py.File(H5, 'r') as f:
        for name in f['assays']:
            parts.append(call_assay(f['assays'][name], **kw).assign(profile_assay=name))
    C = C.merge(pd.concat(parts), on=['profile_assay', 'profile_row'], how='left', validate='1:1')
    C['altered'] = C.altered_fraction >= MIN_FRAC
    return C
