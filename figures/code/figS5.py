"""Supplementary Fig. 7 - imaging mass cytometry (IMC) of selected high-attention cells.

Tiles are produced by validation/imc/gallery.py export (OpenIMC venv: IF->IMC registration, OpenIMC
raw counts for display (no hot-pixel removal), arcsinh, one fixed display window per marker per slide, Cellpose/OpenIMC cell outline),
stored without patient or slide identifiers in analyses/imc_linkage_20261111/figure_tiles. Each row shows the markers
relevant to that cell (defining positives, then informative negatives)."""
import json
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import sx

T = sx.REPO / 'analyses/imc_linkage_20261111/figure_tiles'
UP = 4   # display upsampling (bilinear image, nearest outline)
OUTLINE = np.array([0.0, 0.80, 1.0])
GROUP_GAP = {4: .05, 6: .05, 9: .05, 11: .05}   # extra space before these rows (epithelial / CD31+ / myeloid / lymphoid / platelet-like)


def tile(img, outline=None):
    g = np.clip(zoom(img, (UP, UP) + (1,) * (img.ndim - 2), order=1), 0, 1)
    rgb = np.repeat(g[..., None], 3, -1) if g.ndim == 2 else g
    if outline is not None:
        rgb[zoom(outline.astype(np.uint8), UP, order=0).astype(bool)] = OUTLINE
    return rgb


def build():
    sx.style()
    meta = json.loads((T / 'tiles.json').read_text()); cells = meta['cells']; hw = meta['half_width_px']; um = meta['um_per_px']; um40 = meta['um_per_px_40x']   # 10x 0.59, 40x 0.18 um/px; IMC (1 um/px) is warped onto the 10x grid, so IMC tiles use 0.59
    ts, gap, lab_w, top = .44, .07, 1.95, .225   # page height <= 170 mm (Nature)
    row_h = ts + .12
    H = top + len(cells) * row_h + sum(GROUP_GAP.values()) + .02
    pg = sx.Page(7.0, H)
    x40 = lab_w; x_if = lab_w + ts + gap; x_imc = x_if + ts + .16; lx = lab_w - .08
    pg.text(x40 + ts / 2, .0, '40x\nre-image', ha='center', fontsize=5.8, linespacing=1.1)
    pg.text(x_if + ts / 2, .0, 'IF 10x\n(SPECTRA)', ha='center', fontsize=5.8, linespacing=1.1)
    pg.text(x_imc + (7 * ts + 6 * gap) / 2, .06, 'IMC, same cell (markers not in the IF panel, plus CD31; outline, segmented cell)', ha='center', fontsize=5.8)
    y = top; src = []
    for c in cells:
        y += GROUP_GAP.get(c['cell'], 0)
        z = np.load(T / f"cell{c['cell']:02d}.npz")
        pg.text(lx, y + .16, f"Cluster {c['cluster_display']}, " + ('non-cancer control' if c['cancer_type'] == 'Control' else f"{c['cancer_type'].lower()} cancer"), fontsize=6, fontweight='bold', ha='right')
        pg.text(lx, y + .29, f"{'Test' if c['split'] == 'test' else 'Training'} patient, attribution rank {c['attribution_rank']}", fontsize=5.6, ha='right')
        pg.text(lx, y + .41, c['phenotype'].replace('; ', ';\n'), fontsize=5.6, linespacing=1.15, ha='right', multialignment='right')
        a40 = pg.ax(x40, y + .12, ts, ts)
        if 'X40' in z.files:
            a40.imshow(z['X40'], interpolation='lanczos')
            sx.scalebar(a40, z['X40'].shape[0], um40, um=10, lw=.8)
        else:
            a40.set_facecolor(sx.FAINT); a40.text(.5, .5, 'No 40x', transform=a40.transAxes, ha='center', va='center', fontsize=5.4)
        a = pg.ax(x_if, y + .12, ts, ts)
        if z['IF'].max() > 0:
            a.imshow(tile(z['IF']), interpolation='none')
        else:   # slide registered through its 40x image; raw 10x frames unavailable
            a.set_facecolor(sx.FAINT); a.text(.5, .5, 'No 10x\nframe', transform=a.transAxes, ha='center', va='center', fontsize=5.4, linespacing=1.1)
        cols = ['DNA'] + c['markers']
        for j, m in enumerate(cols):
            a2 = pg.ax(x_imc + j * (ts + gap), y + .12, ts, ts); a2.imshow(tile(z[m], z['outline']), interpolation='none')
            call = '' if m in ('DNA', 'CD45') else (' +' if c['calls'][m]['positive'] else ' −')
            a2.set_title(m.replace('Vimentin', 'VIM').replace('Ecad', 'E-cad') + call, fontsize=5.8, pad=1.5)
            sx.scalebar(a2, UP * 2 * hw, um / UP, um=10, lw=.8)
            if m != 'DNA':
                src.append(dict(cell=c['cell'], cluster=c['cluster_display'], cancer_type=c['cancer_type'], split=c['split'], attribution_rank=c['attribution_rank'],
                                match=c['match'], marker=m, percentile_in_roi=c['calls'][m]['pct'], arcsinh_mean=c['calls'][m]['asinh'], background_arcsinh=c['calls'][m]['background_asinh'],
                                delta_over_background=c['calls'][m]['delta'],
                                called_positive=c['calls'][m]['positive']))
        if z['IF'].max() > 0:
            sx.scalebar(a, UP * 2 * hw, um / UP, um=10, lw=.8)
        for ax_ in pg.fig.axes[-(len(cols) + 2):]:
            ax_.set_xticks([]); ax_.set_yticks([])
            for s in ax_.spines.values():
                s.set_visible(True); s.set_linewidth(.4); s.set_color(sx.MUTED)
        y += row_h
    sx.export(pg, 'supplementary', 'SupplementaryFig7_imc_high_attention_cells')
    pd.DataFrame(src).to_csv(sx.ROOT / 'supplementary/SupplementaryFig7_imc_marker_calls_source.csv', index=False)


if __name__ == '__main__':
    build()
