"""Link IMC ROIs to SPECTRA cells on primary test slides.

Per ROI (name {slide}_{frame}_{cellid}_{acq}; cell id ignored): coarse registration of IMC DNA to the IF
DAPI tile of that frame (register.register), then nucleus ICP/RANSAC affine refinement (register.refine). ROIs pass
QC if >=50% of IMC nuclei match an IF nucleus within 5 IF px. Every candidate-pool cell of that frame (top-10k source
parquet, with x, y) is mapped into the ROI; cells inside (8 px margin) get IMC marker means in a 4-px-radius disk
around the nearest IMC DNA peak within 4 px of the mapped point (else the mapped point). Joined to Figure 3
attribution (model-input membership, relevance rank, top-50, mass-90, Leiden cluster). Read-only on all source drives."""
from pathlib import Path
import sys, re, json, os, time, traceback
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / 'model/pipeline'))
from register import imc_roi, if_frame, register, refine, nuclei, IMC  # noqa: E402
from common import read  # noqa: E402
OUT = REPO / 'analyses/imc_linkage_20261111'; OUT.mkdir(parents=True, exist_ok=True)
SLIDES = ['055048', '0A73E07', '0A73E08', '0B86902', '0B8D301']   # test slides with IMC and raw IF in the raw image store
rec = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
F3 = REPO / 'runs/figure3_locked6_attribution_20261111'
cells = pd.read_parquet(F3 / 'all_cells.parquet', columns=['donor_id', 'slide_id', 'source_row', 'relevance', 'positive_rank', 'top50', 'mass90', 'label', 'called_at_90spec'])
cells['cluster'] = np.load(F3 / 'mass90_leiden/pool_cluster_all_cells.npy')
TRAIN = REPO / 'runs/imc_linkage_20261111/train_attribution'
if TRAIN.exists() and any(TRAIN.glob('*.parquet')):
    tr = pd.concat([pd.read_parquet(p, columns=['donor_id', 'slide_id', 'source_row', 'relevance', 'positive_rank', 'top50', 'mass90', 'cluster']) for p in TRAIN.glob('*.parquet')])
    cells = pd.concat([cells, tr.assign(label=np.nan, called_at_90spec=np.nan, split='train')], ignore_index=True)
cells['split'] = cells.get('split', pd.Series(index=cells.index, dtype=object)).fillna('test')
HIGH_FRAMES_ONLY = False


CACHE = REPO / 'runs/imc_linkage_20261111/mcd_cache'


def load_roi(r):
    if r['file'].endswith('.npz'):
        z = np.load(CACHE / r['file']); labels = [str(x) for x in z['labels']]
        return z['image'], labels
    return imc_roi(IMC / r['slide_id'] / r['file'])


def rois(sid):
    out = []
    for f in sorted(os.listdir(CACHE)) if CACHE.exists() else []:
        m = re.fullmatch(rf'{sid}_(\d+)_(\d+)_(\d+)\.npz', f)
        if m:
            out.append(dict(slide_id=sid, file=f, frame=int(m.group(1)), acq=int(m.group(3))))
    for f in sorted(os.listdir(IMC / sid)):
        m = re.fullmatch(rf'{sid}_(\d+)_(\d+)_(\d+)\.txt', f)
        if m:
            out.append(dict(slide_id=sid, file=f, frame=int(m.group(1)), acq=int(m.group(3))))
    return out


def main(slides=SLIDES, tag=''):
    regs, linked = [], []
    prev = OUT / f'roi_registration{tag}.csv'
    prev = pd.read_csv(prev).set_index('file') if prev.exists() else None   # reuse saved registrations
    todo = [r for s in slides for r in rois(s)]
    if HIGH_FRAMES_ONLY:
        src_frames = {}
        for s_ in slides:
            g = cells[(cells.slide_id == s_) & cells.mass90.fillna(False).astype(bool)]
            fr = pd.read_parquet(rec[s_]['source'], columns=['frame_id']).frame_id.to_numpy()[g.source_row.to_numpy()] if len(g) else []
            src_frames[s_] = set(map(int, fr))
        todo = [r for r in todo if r['frame'] in src_frames[r['slide_id']]]
    print('ROIs', len(todo), flush=True)
    for k, r in enumerate(todo, 1):
        t0 = time.time()
        try:
            arr, cols = load_roi(r)
            dna = arr[[i for i, c in enumerate(cols) if 'DNA' in c]].sum(0)
            dapi = if_frame(r['slide_id'], r['frame'])
            reuse = prev is not None and r['file'] in prev.index and isinstance(prev.loc[r['file']].get('affine'), str)
            if reuse:   # registration already computed and saved: reuse it unchanged
                p_ = prev.loc[r['file']]; A = np.array(json.loads(p_.affine))
                rr = {**p_.to_dict(), **r}; diag = dict(matched_frac=float(p_.matched_frac)); rr['seconds'] = 0.0
            else:
                reg = register(dna, dapi, px_grid=(0.57, 0.59, 0.61))
                A, diag = refine(reg, dna, dapi); constrained = False
                if not (diag['matched_frac'] >= .4 and diag['median_residual_if_px'] < 4):
                    # fallback: instrument geometry is constant across all registered slides (rotation 88-90 deg, mirrored,
                    # ~1.73 IF px per IMC px); small ROIs fail the free search, so constrain rotation and keep QC unchanged
                    reg2 = register(dna, dapi, px_grid=(0.57, 0.58, 0.59), angles=range(86, 93, 2)); A2, d2 = refine(reg2, dna, dapi)
                    if d2['median_residual_if_px'] < diag['median_residual_if_px']:
                        reg, A, diag, constrained = reg2, A2, d2, True
                rr = dict(r, constrained_rotation=constrained, coarse_score=reg['score'], angle=reg['angle'], mirror=reg['mirror'], **diag, affine=json.dumps(A.tolist()),
                          qc_pass=bool(diag['matched_frac'] >= .4 and diag['median_residual_if_px'] < 4), seconds=round(time.time() - t0, 1))
            regs.append(rr)
            if not bool(rr['qc_pass']):
                print(k, r['file'], 'QC fail', round(diag['matched_frac'], 2), flush=True); continue
            src = pd.read_parquet(rec[r['slide_id']]['source'], columns=['frame_id', 'x', 'y', 'area', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']).reset_index(names='source_row')
            src = src[src.frame_id == r['frame']]
            B = np.linalg.inv(A)
            P = (B @ np.c_[src.x, src.y, np.ones(len(src))].T).T
            src = src.assign(imc_col=P[:, 0], imc_row=P[:, 1])
            src = src[(src.imc_col > 8) & (src.imc_col < arr.shape[2] - 8) & (src.imc_row > 8) & (src.imc_row < arr.shape[1] - 8)]
            if len(src):
                pk = nuclei(dna, 1.0, 3, 80); d, j = cKDTree(pk).query(src[['imc_col', 'imc_row']].to_numpy())
                cc = np.where(d < 4, pk[j, 0], src.imc_col); rr_ = np.where(d < 4, pk[j, 1], src.imc_row)
                yy, xx = np.mgrid[:arr.shape[1], :arr.shape[2]]
                vals = []
                for c0, r0 in zip(cc, rr_):
                    m = (xx - c0) ** 2 + (yy - r0) ** 2 <= 16
                    vals.append(arr[:, m].mean(1))
                names = [re.sub(r'\(.*\)', '', c) for c in cols]
                names = [n if names[:i].count(n) == 0 else f'{n}_{names[:i].count(n) + 1}' for i, n in enumerate(names)]   # unique marker names
                v = pd.DataFrame(np.array(vals), columns=names, index=src.index)
                src = pd.concat([src.assign(slide_id=r['slide_id'], roi=r['file'], nucleus_snap_um=d), v], axis=1)
                linked.append(src)
            print(k, r['file'], 'ok', round(diag['matched_frac'], 2), 'cells', len(src), rr['seconds'], 's', flush=True)
        except Exception as e:
            regs.append(dict(r, error=repr(e))); print(k, r['file'], 'ERROR', repr(e), flush=True)
        pd.DataFrame(regs).to_csv(OUT / f'roi_registration{tag}.csv', index=False)
    L = pd.concat(linked, ignore_index=True) if linked else pd.DataFrame()
    if len(L):
        L = L.merge(cells, on=['slide_id', 'source_row'], how='left')
        L['model_input'] = L.relevance.notna()
        L.to_parquet(OUT / f'linked_cells{tag}_private.parquet', index=False)
        print(L.groupby('slide_id').agg(cells=('source_row', 'size'), model_input=('model_input', 'sum'), mass90=('mass90', 'sum'), top50=('top50', 'sum')))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1].startswith('_train'):
            HIGH_FRAMES_ONLY = True
        main(sys.argv[2:], sys.argv[1])
    else:
        main()
