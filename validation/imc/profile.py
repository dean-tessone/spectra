"""Per-cell IMC profiles for linked high-attention SPECTRA cells (exploratory; a sample of
high-attribution cells from various clusters on a small set of patients, no statistics).

For each registered ROI containing a linked mass-90 cell: all IMC cells are detected as DNA peaks; each cell gets
marker values = mean over a 7-um-radius disk (nucleus + cytoplasm). A linked cell is 'present in IMC' when an IMC
nucleus lies within 4 um of its mapped position; its marker values are then expressed as percentiles among all IMC
cells of the same ROI (panel- and ROI-independent). Also writes an image sheet per cell: IF 10x composite
(cell_ext_display convention) and IMC marker crops warped into the IF frame orientation."""
from pathlib import Path
import sys, json, re
import numpy as np, pandas as pd, cv2
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO / 'imaging/cell_images')); sys.path.insert(0, str(REPO / 'figures/code'))
from register import imc_roi, if_frame, nuclei, IMC  # noqa: E402
from link import load_roi  # noqa: E402
import sx  # noqa: E402
OUT = REPO / 'analyses/imc_linkage_20261111'
SHEETS = OUT / 'cell_sheets_private'; SHEETS.mkdir(exist_ok=True)
NUM = sx.cluster_numbers(); names = pd.read_csv(REPO / 'analyses/figure3_locked6_attribution_20261111/leiden/cluster_names.csv', index_col=0).name
KEY = [('CK', ['PanCK', 'Cytokeratin_8_18', 'CK8_and_CK18', 'CK7', 'Pan_Keratin', 'panCK']), ('EpCAM', ['EpCAM', 'EpCAM_CD326']), ('Vimentin', ['Vimentin', 'VIM']),
       ('CD45', ['CD45']), ('CD31', ['CD31']), ('CD61', ['CD61']), ('CD14', ['CD14']), ('CD66', ['CD66', 'CD66b']), ('CD3', ['CD3']), ('CD68', ['CD68']),
       ('AR', ['AR']), ('PSMA', ['PSMA_FOLH1__NAALADase1', 'PSMA']), ('Ki67', ['Ki_67', 'Ki67'])]


def clean(c):
    c = re.sub(r'\(.*\)', '', c)
    return re.sub(r'^\d+[A-Z][a-z]?_', '', c)


def main():
    L = pd.concat([pd.read_parquet(p) for p in OUT.glob('linked_cells*_private.parquet')], ignore_index=True)
    regs = pd.concat([pd.read_csv(p) for p in OUT.glob('roi_registration*.csv')], ignore_index=True).set_index('file')
    H = L[L.mass90.fillna(False).astype(bool)].copy()
    rows = []
    for roi, g in H.groupby('roi'):
        sid = g.slide_id.iloc[0]; r = regs.loc[roi]
        arr, cols = load_roi(dict(slide_id=sid, file=roi)); cols = [clean(c) for c in cols]
        dna = arr[[i for i, c in enumerate(cols) if 'DNA' in c]].sum(0)
        pk = nuclei(dna, 1.5, 4, 80, otsu=True)
        yy, xx = np.mgrid[:arr.shape[1], :arr.shape[2]]
        def prof(c0, r0):
            m = (xx - c0) ** 2 + (yy - r0) ** 2 <= 49
            return arr[:, m].mean(1)
        allc = np.array([prof(c0, r0) for c0, r0 in pk])
        d, j = cKDTree(pk).query(g[['imc_col', 'imc_row']].to_numpy())
        A = np.array(json.loads(r.affine)); dapi = None
        for (_, c), dd, jj in zip(g.iterrows(), d, j):
            present = dd < 4
            row = dict(split=c.get('split', 'test'), slide_id=sid, donor_id=c.donor_id, roi=roi, source_row=int(c.source_row), positive_rank=int(c.positive_rank), top50=bool(c.top50),
                       cluster=int(c.cluster), display_number=NUM[int(c.cluster)], cluster_name=names[int(c.cluster)], present_in_imc=bool(present),
                       nucleus_distance_um=float(dd), roi_cells=len(pk), IF_CK=c.TRITC_mean, IF_CD45_CD31=c.CY5_mean, IF_VIM=c.FITC_mean, IF_DAPI=c.DAPI_mean)
            if present:
                v = allc[jj]
                for k, alts in KEY:
                    ix = [i for i, cc in enumerate(cols) if cc in alts]
                    if ix:
                        pct = float((allc[:, ix[0]] < v[ix[0]]).mean() * 100); fold = float(v[ix[0]] / (np.median(allc[:, ix[0]]) + .05))
                        row[f'{k}_pct'] = pct; row[f'{k}_fold'] = fold; row[f'{k}_pos'] = bool(pct >= 90 and fold >= 3)
            rows.append(row)
            # image sheet
            if dapi is None:
                fr = int(r.frame); ifimg = np.stack([if_frame(sid, fr, ch) for ch in range(4)], -1); dapi = True
            from cell_ext_display import rgb
            x0, y0 = int(c.x), int(c.y); hw = 40
            padimg = np.pad(ifimg, ((hw, hw), (hw, hw), (0, 0))); crop = padimg[y0:y0 + 2 * hw, x0:x0 + 2 * hw]
            warp = lambda im: cv2.warpAffine(im.astype(np.float32), np.linalg.inv(A)[:2], (ifimg.shape[1], ifimg.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            show = [('DNA', [i for i, cc in enumerate(cols) if 'DNA' in cc])] + [(k, [i for i, cc in enumerate(cols) if cc in alts]) for k, alts in KEY[:6]]
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1 + len(show), figsize=(2 * (1 + len(show)), 2.3))
            scl = np.percentile(ifimg.reshape(-1, 4), 99.9, axis=0) + 1e-6; q = np.clip(crop / scl, 0, 1)
            ax[0].imshow(np.clip(np.dstack([q[..., 1] + q[..., 3], q[..., 2] + q[..., 3], q[..., 0] + q[..., 3]]), 0, 1), interpolation='lanczos')
            ax[0].set_title(f'IF 10x ({c.get("split", "test")})  rank {int(c.positive_rank)}  #{NUM[int(c.cluster)]}', fontsize=7)
            for a, (k, ix) in zip(ax[1:], show):
                if not ix:
                    a.set_title(f'{k}: n/a', fontsize=7); a.axis('off'); continue
                w = np.pad(warp(arr[ix].sum(0)), hw)[y0:y0 + 2 * hw, x0:x0 + 2 * hw]
                a.imshow(np.arcsinh(w / 1.0), cmap='magma', vmin=0, vmax=np.percentile(np.arcsinh(arr[ix].sum(0)), 99.5) + 1e-6)
                pct = row.get(f'{k}_pct'); a.set_title(f'{k}' + (f' {pct:.0f}th pct' + (' +' if row.get(f"{k}_pos") else '') if pct is not None else ''), fontsize=7)
            for a in ax:
                a.set_xticks([]); a.set_yticks([])
            fig.savefig(SHEETS / f'{sid}_{int(c.source_row)}.png', dpi=110, bbox_inches='tight'); plt.close(fig)
    P = pd.DataFrame(rows).sort_values(['display_number', 'positive_rank'])
    P.to_csv(OUT / 'high_attention_cell_profiles_private.csv', index=False)
    pd.set_option('display.width', 250); print(P.drop(columns=['roi']).round(0).to_string(index=False))


if __name__ == '__main__':
    main()
