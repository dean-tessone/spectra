"""Data for Supplementary Fig. 5 (cluster biology: IMC and single-cell low-pass WGS), no patient identifiers in outputs.
  cluster_evidence.csv: per Fig. 3 cluster, IMC high-attention cells on usable-signal slides (segmented; calls from
    imc_linkage segment_openimc.call_positive) with the fraction positive per marker group, and sequenced high-attention
    (mass-90) test cells with the fraction copy-number altered (caller.py, first pass).
  cnv_examples.npz: one sequenced high-attention test cell per main cluster (lowest-noise cell with the cluster's majority
    call that has a 40x re-image): rotated 40x crop (cell_ext display) and log2 ratio / segment profile.
Run with the repository venv. Read-only on all source data."""
from pathlib import Path
import sys, re, os, json
from collections import defaultdict
import numpy as np, pandas as pd, h5py
from PIL import Image
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'imaging/cell_images'))
from cell_ext_display import preprocess_frame, rgb  # noqa: E402
OUT = REPO / 'analyses/cnv_attribution_20261112/figure_data'; OUT.mkdir(parents=True, exist_ok=True)
H5 = os.environ.get('SPECTRA_CNV_H5', 'raw/qc_approved_genomic_profiles.h5')
LEI = REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
GOOD = ['055048', '0750910', '075469', '09E3D02', '0A63004', '0A73E07', '0A73E08', '0A78F08', '0B86902', '078708', '0ADD705', '0ADD708', '059588', '0A9AC01']
CLUSTERS = [20, 21, 17, 10, 16]
GROUPS = [('EpCAM', ['EpCAM']), ('CD44', ['CD44']), ('CD31', ['CD31']), ('Myeloid', ['CD14', 'CD68']), ('Lymphoid', ['CD3', 'CD45RA']), ('CD61', ['CD61'])]
PAT = re.compile(r'(\d+)-(\d+)-(-?\d+)-(-?\d+)-(DAPI|TRITC|FITC|CY5)\.tif', re.I)


def forty(slide, frame, x, y, k=120):
    p = Path(os.environ.get('SPECTRA_40X_ROOT', 'raw/40x')) / slide / 'ClientBin/40x'
    if not p.is_dir():
        return None
    files = defaultdict(set)
    for f in os.listdir(p):
        m = PAT.fullmatch(f)
        if m:
            files[tuple(map(int, m.groups()[:4]))].add(m.group(5).upper())
    hits = [h for h, ch in files.items() if h[1] == frame and abs(h[2] - x) <= 10 and abs(h[3] - y) <= 10 and len(ch) == 4]
    if len(hits) != 1:
        return None
    acq = '-'.join(map(str, hits[0])); arrs, bits = [], []
    for c in ['DAPI', 'TRITC', 'CY5', 'FITC']:
        with Image.open(p / f'{acq}-{c}.tif') as im:
            arrs.append(np.array(im).astype(np.uint16)); bits.append(int(im.tag_v2[258][0]))
    d = preprocess_frame(np.stack(arrs, -1), [65535 / (2 ** b - 1) for b in bits])[::-1, ::-1]
    h, w = d.shape[:2]; im = rgb(d[h // 2 - k:h // 2 + k, w // 2 - k:w // 2 + k]).astype(np.float32)
    return np.clip(im / (np.percentile(im, 99.7) + 1e-6), 0, 1)


def main():
    num = pd.read_csv(LEI / 'cluster_display_numbers.csv').set_index('cluster').number
    names = pd.read_csv(LEI / 'cluster_names.csv', index_col=0).name
    tab = pd.read_csv(LEI / 'cluster_table.csv', index_col=0)
    P = pd.read_csv(REPO / 'analyses/imc_linkage_20261111/imc_profiles_private.csv', dtype={'slide_id': str})
    P = P[P.slide_id.isin(GOOD) & P.match.isin(['inside nucleus', 'inside expanded cell'])].assign(num=lambda d: d.cluster.map(num))
    T = pd.read_csv(REPO / 'analyses/cnv_attribution_20261112/test_sequenced_cells_private.csv', dtype={'slide_id': str})
    S = T[T.mass90 == True]
    rows = []
    inv = {v: k for k, v in num.items()}
    for n in CLUSTERS:
        g = P[P.num == n]; s = S[S.display_cluster == n]
        r = dict(cluster=n, name=names[inv[n]], emphasis_log2=float(tab.loc[inv[n], 'emphasis_log2_cancer']),
                 imc_cells=len(g), imc_patients=g.donor_id.nunique(), seq_cells=len(s), seq_patients=s.donor_id.nunique(),
                 seq_altered=int(s.altered.sum()))
        for lab, ms in GROUPS:
            cols = [f'{m}_pos' for m in ms if f'{m}_pos' in g]
            meas = g[cols].notna().any(axis=1) if cols else pd.Series(False, index=g.index)
            pos = g[cols].isin([True, 'True']).any(axis=1) if cols else pd.Series(False, index=g.index)
            r[f'{lab}_measured'] = int(meas.sum()); r[f'{lab}_positive'] = int((pos & meas).sum())
        rows.append(r)
    E = pd.DataFrame(rows); E.to_csv(OUT / 'cluster_evidence.csv', index=False); print(E.to_string(index=False))
    # example cells
    with h5py.File(H5, 'r') as f:
        ex = {}
        for n in CLUSTERS:
            s = S[S.display_cluster == n]; maj = s.altered.mean() >= .5
            cand = s[(s.altered == maj) & ((s.altered_fraction >= .2) if maj else True)]   # unambiguous examples
            for r in cand.sort_values('noise_sigma').itertuples():
                im = forty(r.slide_id, int(r.frame_id), int(r.x_seq) if 'x_seq' in s else int(r.x), int(r.y_seq) if 'y_seq' in s else int(r.y))
                if im is None:
                    continue
                g = f['assays'][r.profile_assay]; row = int(r.profile_row); pos = g['pos_cols'][:]
                keep = pos[:, 0] <= 23; auto = pos[:, 0] <= 22
                sm = g['segment_mean'][row].astype(float); ra = g['ratio'][row].astype(float); med = np.median(sm[auto])
                ex[n] = dict(img=im, chrom=pos[keep, 0], gpos=pos[keep, 2].astype(float), r=np.log2(np.clip(ra[keep], 1e-3, None) / med),
                             s=np.log2(np.clip(sm[keep], 1e-3, None) / med), altered_fraction=float(r.altered_fraction), rank=int(r.positive_rank),
                             altered=bool(r.altered), cancer=str(r.cancer_type))
                print(n, 'example rank', r.positive_rank, 'altered', r.altered, round(r.altered_fraction, 2)); break
    np.savez_compressed(OUT / 'cnv_examples.npz', **{f'{k}_{n}': v for n, d in ex.items() for k, v in d.items()})
    (OUT / 'cnv_examples.json').write_text(json.dumps({n: dict(rank=d['rank'], altered=d['altered'], altered_fraction=d['altered_fraction'], cancer=d['cancer']) for n, d in ex.items()}))


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


# ---------------- v2: per-cell views ----------------
MARKERS = ['EpCAM', 'CD44', 'CD31', 'CD14', 'CD68', 'CD66b', 'CD3', 'CD45RA', 'CD61']
GRID_MB = 2.0


def per_cell():
    """imc_cells.csv: one row per segmented high-attention IMC cell (clusters in CLUSTERS, usable slides) with each
    marker's call (1 positive, 0 negative, NaN not in panel) and cancer/control. cnv_matrix.npz: segment log2 values of
    every sequenced high-attention test cell in CLUSTERS on a common autosomal 2-Mb grid, rows ordered by cluster, then
    patient, then altered fraction; with cluster, patient index (no identifiers) and altered call per row."""
    num = pd.read_csv(LEI / 'cluster_display_numbers.csv').set_index('cluster').number
    P = pd.read_csv(REPO / 'analyses/imc_linkage_20261111/imc_profiles_private.csv', dtype={'slide_id': str})
    P = P[P.slide_id.isin(GOOD) & P.match.isin(['inside nucleus', 'inside expanded cell'])].assign(num=lambda d: d.cluster.map(num))
    P = P[P.num.isin(CLUSTERS)]
    led = pd.read_csv(REPO / 'cohort/reference_policy/population_patient_ledger.csv', dtype=str).set_index('donor_id').label
    rows = []
    for r in P.itertuples():
        d = dict(cluster=int(r.num), split=r.split, control=led[r.donor_id] == '0')
        for m in MARKERS:
            v = getattr(r, f'{m}_pos', None)
            d[m] = np.nan if (v is None or pd.isna(v)) else float(v in (True, 'True'))
        rows.append(d)
    I = pd.DataFrame(rows); I.to_csv(OUT / 'imc_cells.csv', index=False)
    T = pd.read_csv(REPO / 'analyses/cnv_attribution_20261112/test_sequenced_cells_private.csv', dtype={'slide_id': str})
    S = T[(T.mass90 == True) & T.display_cluster.isin(CLUSTERS)].copy()
    S['cl_order'] = S.display_cluster.map({c: i for i, c in enumerate(CLUSTERS)})
    S = S.sort_values(['cl_order', 'donor_id', 'altered_fraction'], ascending=[True, True, False])
    with h5py.File(H5, 'r') as f:
        ref = f['assays']['june_PE']['pos_cols'][:]
        chrom_len = {c: ref[ref[:, 0] == c, 1].max() for c in range(1, 23)}
        grid = [(c, s) for c in range(1, 23) for s in np.arange(0, chrom_len[c] + 1, GRID_MB * 1e6)]
        gc = np.array([g[0] for g in grid]); gs = np.array([g[1] for g in grid])
        M = np.full((len(S), len(grid)), np.nan, np.float32)
        for i, r in enumerate(S.itertuples()):
            g = f['assays'][r.profile_assay]; pos = g['pos_cols'][:]; sm = g['segment_mean'][int(r.profile_row)].astype(float)
            auto = pos[:, 0] <= 22; med = np.median(sm[auto]); lv = np.log2(np.clip(sm, 1e-3, None) / med)
            for c in range(1, 23):
                k = np.flatnonzero(pos[:, 0] == c); sel = gc == c
                j = np.clip(np.searchsorted(pos[k, 1], gs[sel], side='right') - 1, 0, len(k) - 1)
                M[i, sel] = lv[k[j]]
    pid = pd.factorize(S.donor_id)[0]
    np.savez_compressed(OUT / 'cnv_matrix.npz', M=M, chrom=gc, start=gs, cluster=S.display_cluster.to_numpy().astype(int), patient=pid,
                        altered=S.altered.to_numpy().astype(bool), cancer=np.array(S.cancer_type.astype(str).tolist(), dtype='U40'))
    print(I.groupby('cluster').size().to_dict(), M.shape, S.groupby('display_cluster').donor_id.nunique().to_dict())


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'per_cell':
    per_cell()


def tenx(slide, frame, x, y, hw=30):
    """10x IF crop (cell_ext channel convention, frame-percentile scaling as in the IMC gallery), or None."""
    sys.path.insert(0, str(REPO / 'validation/imc'))
    from register import if_frame
    try:
        im = np.stack([if_frame(slide, frame, ch) for ch in range(4)], -1)
    except Exception:
        return None
    c = np.pad(im, ((hw, hw), (hw, hw), (0, 0)))[y:y + 2 * hw, x:x + 2 * hw] / (np.percentile(im.reshape(-1, 4), 99.9, axis=0) + 1e-6)
    q = np.clip(c, 0, 1); return np.clip(np.dstack([q[..., 1] + q[..., 3], q[..., 2] + q[..., 3], q[..., 0] + q[..., 3]]), 0, 1)


IMC_PREF = {21: ['EpCAM', 'CD14'], 17: ['CD44'], 10: ['CD31', 'CD68'], 16: ['CD44', 'CD45RA'], 20: []}


def thumbs(n_imc=0, n_seq=3):
    """thumbs.npz/json: per cluster, n_seq sequenced high-attention test cells (cluster-majority call, distinct patients,
    lowest noise) and IMC cells showing each cluster's dominant marker(s) with the fewest other positives (IMC_PREF; cluster 20 has none, so it gets n_seq + n_imc sequenced),
    each as a 40x crop where a re-image exists, else a 10x crop. Labels carry no identifiers."""
    from PIL import Image as PI
    num = pd.read_csv(LEI / 'cluster_display_numbers.csv').set_index('cluster').number
    T = pd.read_csv(REPO / 'analyses/cnv_attribution_20261112/test_sequenced_cells_private.csv', dtype={'slide_id': str})
    S = T[(T.mass90 == True)]
    P = pd.read_csv(REPO / 'analyses/imc_linkage_20261111/imc_profiles_private.csv', dtype={'slide_id': str})
    P = P[P.slide_id.isin(GOOD) & P.match.isin(['inside nucleus', 'inside expanded cell'])].assign(num=lambda d: d.cluster.map(num))
    L = pd.concat([pd.read_parquet(p, columns=['roi', 'source_row', 'frame_id', 'x', 'y']) for p in (REPO / 'analyses/imc_linkage_20261111').glob('linked_cells*_private.parquet')]).drop_duplicates(['roi', 'source_row'])
    P = P.merge(L, on=['roi', 'source_row'], how='left')
    out, meta = {}, []
    sys.path.insert(0, str(REPO / 'model/pipeline'))
    from common import read as _read
    rec = _read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']; _src = {}
    def ref4(slide, row):
        if slide not in _src:
            _src[slide] = pd.read_parquet(rec[slide]['source'], columns=['DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean'])
        return _src[slide].iloc[int(row)].to_numpy(float)
    def img(slide, frame, x, y, row=None):
        a = forty_centred(slide, frame, x, y, ref4(slide, row), fixed_half=98) if row is not None and slide in rec else None   # centred on the target cell; fixed 35 um field (98 px x 0.18 um, = 10x 30 px x 0.59 um)
        if a is not None:
            return a, '40x'
        b = tenx(slide, frame, x, y)
        return (b, '10x') if b is not None else (None, None)
    for c in CLUSTERS:
        s = S[S.display_cluster == c]; maj = s.altered.mean() >= .5; k = 0; used = set()
        want = n_seq + (n_imc if not (P.num == c).any() else 0)
        for r in s[s.altered == maj].sort_values('noise_sigma').itertuples():
            if r.donor_id in used or k >= want:
                continue
            im, mag = img(r.slide_id, int(r.frame_id), int(r.x), int(r.y), r.source_row)
            if im is None:
                continue
            key = f'c{c}_{len(meta)}'; out[key] = np.asarray(PI.fromarray((im * 255).astype(np.uint8)).resize((128, 128), PI.LANCZOS))
            meta.append(dict(key=key, cluster=c, kind='WGS', label='Altered' if r.altered else 'Flat', mag=mag, field_um=2 * 98 * .18 if mag == '40x' else 2 * 30 * .59)); used.add(r.donor_id); k += 1
        g = P[P.num == c].copy()
        if len(g):
            pos = [f'{m}' for m in MARKERS]
            g['_n'] = g[[f'{m}_pos' for m in pos if f'{m}_pos' in g]].isin([True, 'True']).sum(axis=1)
            prefs = IMC_PREF[c]; k = 0
            for want_m in prefs:   # one cell per dominant marker: has it, fewest other positives
                h = g[g[f'{want_m}_pos'].isin([True, 'True'])].sort_values('_n') if f'{want_m}_pos' in g else g.iloc[:0]
                for r in h.itertuples():
                    im, mag = img(r.slide_id, int(r.frame_id), int(r.x), int(r.y), r.source_row)
                    if im is None:
                        continue
                    lab = ' '.join(f'{m}+' for m in pos if getattr(r, f'{m}_pos', None) in (True, 'True')) or 'No marker'
                    key = f'c{c}_{len(meta)}'; out[key] = np.asarray(PI.fromarray((im * 255).astype(np.uint8)).resize((128, 128), PI.LANCZOS))
                    meta.append(dict(key=key, cluster=c, kind='IMC', label=lab, mag=mag, field_um=2 * 98 * .18 if mag == '40x' else 2 * 30 * .59)); k += 1; break
    np.savez_compressed(OUT / 'thumbs.npz', **out); (OUT / 'thumbs.json').write_text(json.dumps(meta)); print(pd.DataFrame(meta).to_string(index=False))



def forty_centred(slide, frame, x, y, ref4, size=128, fixed_half=None):
    """40x thumbnail centred on the target cell (Figure 3 rule, center_40x.py): 401-px 40x context (Fig. 3 display
    processing, 180 deg), Otsu components of the smoothed channel sum, choose the component whose 4-channel profile
    (cosine of centred log means vs the cell's 10x DAPI/TRITC/CY5/FITC means) minus 0.004/px distance is best within
    120 px of centre; crop half-width max(52, 1.45 x equivalent radius) on its intensity-weighted centroid."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    from PIL import Image as PI
    p = Path(os.environ.get('SPECTRA_40X_ROOT', 'raw/40x')) / slide / 'ClientBin/40x'
    if not p.is_dir():
        return None
    files = defaultdict(set)
    for f in os.listdir(p):
        m = PAT.fullmatch(f)
        if m:
            files[tuple(map(int, m.groups()[:4]))].add(m.group(5).upper())
    hits = [h for h, ch in files.items() if h[1] == frame and abs(h[2] - x) <= 10 and abs(h[3] - y) <= 10 and len(ch) == 4]
    if len(hits) != 1:
        return None
    acq = '-'.join(map(str, hits[0])); arrs, bits = [], []
    for c in ['DAPI', 'TRITC', 'CY5', 'FITC']:
        with Image.open(p / f'{acq}-{c}.tif') as im:
            arrs.append(np.array(im).astype(np.uint16)); bits.append(int(im.tag_v2[258][0]))
    d = preprocess_frame(np.stack(arrs, -1), [65535 / (2 ** b - 1) for b in bits])[::-1, ::-1]
    h, w = d.shape[:2]; im = d[h // 2 - 200:h // 2 + 201, w // 2 - 200:w // 2 + 201].astype(np.float32)
    ref = np.log1p(np.asarray(ref4, float)); ref = ref - ref.mean()
    s = ndi.gaussian_filter(im.sum(-1), 3); lab, n = ndi.label(s > threshold_otsu(s)); best = None
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < 300:
            continue
        cy, cx = ndi.center_of_mass(s * m); dist = np.hypot(cy - 200, cx - 200)
        if dist > 120:
            continue
        v = np.log1p(im[m].mean(0)); v = v - v.mean(); sim = float(v @ ref / (np.linalg.norm(v) * np.linalg.norm(ref) + 1e-9))
        if best is None or sim - .004 * dist > best[3]:
            best = (cy, cx, m.sum(), sim - .004 * dist)
    cy, cx, area = (200, 200, 3000) if best is None else best[:3]
    half = int(fixed_half) if fixed_half else int(np.clip(1.45 * np.sqrt(area / np.pi), 52, 110)); cy = int(np.clip(round(cy), half, 400 - half)); cx = int(np.clip(round(cx), half, 400 - half))
    rgbim = rgb(im[cy - half:cy + half, cx - half:cx + half].astype(np.uint16)).astype(np.float32); rgbim = np.clip(rgbim / (np.percentile(rgbim, 99.7) + 1e-6), 0, 1)
    return np.asarray(PI.fromarray((rgbim * 255).astype(np.uint8)).resize((size, size), PI.LANCZOS)) / 255.0


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'thumbs':
    thumbs()
