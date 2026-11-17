"""Review sheets for the manual many-cell gallery.

    python gallery_review.py pool     # candidate pool (40x first) + one contact sheet per cluster with numbered tiles
    python gallery_review.py build    # gallery tiles from gallery_manual_selection.csv (picks and centre shifts)
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
import gallery_many as G

A, L = G.A, G.L
REV = L / 'gallery_review_private'; REV.mkdir(exist_ok=True)
sys.path.insert(0, str(G.REPO / 'imaging/cell_images')); sys.path.insert(0, str(G.REPO / 'figures/code'))
N40, T = 60, 132


def _ten():
    ten = {}
    for f in (A / 'tenx_many/image_cache').glob('*.npz'):
        z = np.load(f); ten.update(dict(zip(z['cell_keys'], z['images'])))
    return ten


def crop40(key, cx, cy, half=98):
    from cell_ext_display import rgb
    import sx
    im = np.load(A / 'cache_40x' / f'{key}.npz')['image']; cx, cy = int(np.clip(cx, half, 400 - half)), int(np.clip(cy, half, 400 - half))
    return sx.boost(rgb(im[cy - half:cy + half + 1, cx - half:cx + half + 1]))


def crop10(im, dx=0, dy=0):
    from cell_ext_display import rgb
    import sx
    h = im.shape[0] // 2; cx, cy = int(np.clip(h + dx, 30, im.shape[1] - 31)), int(np.clip(h + dy, 30, im.shape[0] - 31))
    return sx.boost(rgb(im[cy - 30:cy + 31, cx - 30:cx + 31]))


def pool():
    at = pd.read_parquet(L / 'atlas_mass90_private.parquet', columns=['cell_key', 'donor_id', 'cluster', 'label', 'relevance', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean'])
    have40 = {f.stem for f in (A / 'cache_40x').glob('*.npz')}; ten = _ten()
    cand = pd.read_csv(L / 'gallery_many_candidates_private.csv'); NUM = pd.read_csv(L / 'cluster_display_numbers.csv', index_col=0).number
    rows = []
    for k in sorted(at.cluster.unique()):
        g = at[(at.cluster == k) & at.cell_key.isin(have40)].sort_values(['label', 'relevance'], ascending=False)
        first = g.drop_duplicates('donor_id'); g = pd.concat([first, g[~g.cell_key.isin(first.cell_key)]]).head(N40)
        i = 0
        for r in g.itertuples():
            im = np.load(A / 'cache_40x' / f'{r.cell_key}.npz')['image']; c = G.centre(im, [r.DAPI_mean, r.TRITC_mean, r.CY5_mean, r.FITC_mean])
            cx, cy = (200, 200) if c is None else c
            rows.append(dict(cluster=k, display=NUM[k], idx=i, source='40x', cell_key=r.cell_key, label=r.label, cx=cx, cy=cy)); i += 1
        for r in cand[(cand.cluster == k) & (cand.source == '10x') & cand.cell_key.isin(ten)].sort_values('order').itertuples():
            rows.append(dict(cluster=k, display=NUM[k], idx=i, source='10x', cell_key=r.cell_key, label=1, cx=0, cy=0)); i += 1
    p = pd.DataFrame(rows); p.to_csv(L / 'gallery_review_pool_private.csv', index=False)
    for k, g in p.groupby('cluster'):
        ncol = 12; nrow = int(np.ceil(len(g) / ncol)); sheet = Image.new('RGB', (ncol * T, nrow * T), (40, 40, 40)); d = ImageDraw.Draw(sheet)
        for j, r in enumerate(g.itertuples()):
            # 40x tiles show a wider context (+/-130 px) with the +/-98 px gallery crop outlined, so off-centre cells can be shifted
            t = crop40(r.cell_key, r.cx, r.cy, half=130) if r.source == '40x' else crop10(ten[r.cell_key])
            t = Image.fromarray(np.asarray(t, np.uint8)).resize((T - 2, T - 2), Image.LANCZOS); x, y = (j % ncol) * T, (j // ncol) * T
            sheet.paste(t, (x + 1, y + 1)); dd = ImageDraw.Draw(sheet)
            if r.source == '40x':
                m = (T - 2) * (1 - 98 / 130) / 2; dd.rectangle([x + 1 + m, y + 1 + m, x + T - 1 - m, y + T - 1 - m], outline=(255, 255, 0))
            dd.text((x + 4, y + 2), f'{r.idx}' + ('' if r.source == '40x' else ' 10x'), fill=(255, 255, 255))
        sheet.save(REV / f'cluster_{int(g.display.iloc[0]):02d}.png')
    print(p.groupby(['display', 'source']).size().unstack(fill_value=0).T.to_string())


def build(n=18):
    """Tiles from the manual selection: columns display, idx, dx, dy (shift of the crop centre as a fraction of the yellow box,
    +x right, +y down; blank = 0). Order in the file is the order in the figure (40x first)."""
    p = pd.read_csv(L / 'gallery_review_pool_private.csv').set_index(['display', 'idx']); s = pd.read_csv(L / 'gallery_manual_selection.csv').fillna(0)
    ten = _ten(); out, keep = {}, []
    for r in s.itertuples():
        q = p.loc[(r.display, r.idx)]
        if q.source == '40x':
            t = crop40(q.cell_key, q.cx + r.dx * 196, q.cy + r.dy * 196)
            t = np.asarray(Image.fromarray(np.asarray(t, np.uint8)).resize((120, 120), Image.LANCZOS))
        else:
            t = crop10(ten[q.cell_key], r.dx * 61, r.dy * 61); t = np.asarray(Image.fromarray(np.asarray(t, np.uint8)).resize((120, 120), Image.BICUBIC))
        out[str(len(keep))] = t; keep.append(dict(cluster=int(q.cluster), source=q.source, cell_key=q.cell_key, tile=len(keep), manual_dx=r.dx, manual_dy=r.dy))
    k = pd.DataFrame(keep); k['o'] = (k.source == '10x').astype(int); k = k.sort_values(['cluster', 'o', 'tile']).drop(columns='o')
    k.to_csv(L / 'gallery_many_private.csv', index=False); np.savez_compressed(L / 'gallery_many_private.npz', field_um=35.3, **out)
    print(k.groupby(['cluster', 'source']).size().unstack(fill_value=0).T.to_string())


if __name__ == '__main__' and sys.argv[1] in ('pool', 'build'):
    {'pool': pool, 'build': build}[sys.argv[1]]()


def check(n=16, T=118):
    """Centring check sheets: the tiles as they appear in the figure (first n per cluster, 40x first), with edge ticks at the centre."""
    k = pd.read_csv(L / 'gallery_many_private.csv'); z = np.load(L / 'gallery_many_private.npz'); NUM = pd.read_csv(L / 'cluster_display_numbers.csv', index_col=0).number
    k['display'] = k.cluster.map(NUM); order = sorted(k.display.unique())
    for part in range(0, len(order), 6):
        ds = order[part:part + 6]; sheet = Image.new('RGB', (n * T + 40, len(ds) * T), (60, 60, 60)); d = ImageDraw.Draw(sheet)
        for i, dnum in enumerate(ds):
            g = k[k.display == dnum].sort_values('tile').head(n); d.text((4, i * T + T // 2 - 6), f'{dnum}', fill=(255, 255, 0))
            for j, r in enumerate(g.itertuples()):
                t = Image.fromarray(z[str(r.tile)]).resize((T - 2, T - 2), Image.LANCZOS); x, y = 40 + j * T, i * T; sheet.paste(t, (x + 1, y + 1))
                c = T // 2
                for (x0, y0, x1, y1) in [(x + c, y + 1, x + c, y + 7), (x + c, y + T - 8, x + c, y + T - 2), (x + 1, y + c, x + 7, y + c), (x + T - 8, y + c, x + T - 2, y + c)]:
                    d.line([x0, y0, x1, y1], fill=(255, 255, 0), width=1)
                d.text((x + 4, y + 2), f'{j + 1}', fill=(255, 255, 255))
        sheet.save(REV / f'check_{part // 6 + 1}.png')


if __name__ == '__main__' and sys.argv[1] == 'check':
    check()
