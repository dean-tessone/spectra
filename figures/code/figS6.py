"""Supplementary Fig. 5 - what the main high-attention clusters are: per-cell imaging mass cytometry (a) and
single-cell low-pass WGS copy-number profiles (b). Data: validation/single_cell_cnv/figure_data.py per_cell
(no patient identifiers)."""
import json
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import sx
from sx import INK, MUTED, RULE, SPECTRA, FAINT, CONTROL

D = sx.REPO / 'analyses/cnv_attribution_20261112/figure_data'
MARKERS = ['EpCAM', 'CD44', 'CD31', 'CD14', 'CD68', 'CD66b', 'CD3', 'CD45RA', 'CD61']
ORDER = [20, 21, 17, 10, 16]
CMAP = LinearSegmentedColormap.from_list('cn', ['#2C6FAD', '#FFFFFF', '#C0392B'])


def names():
    L = sx.REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
    num = pd.read_csv(L / 'cluster_display_numbers.csv').set_index('cluster').number
    return {int(num[k]): v for k, v in pd.read_csv(L / 'cluster_names.csv', index_col=0).name.items()}


def emphasis():
    L = sx.REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
    num = pd.read_csv(L / 'cluster_display_numbers.csv').set_index('cluster').number
    t = pd.read_csv(L / 'cluster_table.csv', index_col=0)
    return {int(num[k]): float(v) for k, v in t.emphasis_log2_cancer.items()}


def build():
    sx.style(); nm = names(); emph = emphasis()
    TH = json.loads((D / 'thumbs.json').read_text()); tz = np.load(D / 'thumbs.npz')
    I = pd.read_csv(D / 'imc_cells.csv'); z = np.load(D / 'cnv_matrix.npz')
    pg = sx.Page(7.0, 6.68)   # <= 170 mm tall (Nature)

    # ---------- a: IMC, one column per cell ----------
    pg.letter(.02, .02, 'a')
    cl_imc = [c for c in ORDER if (I.cluster == c).any()]
    gap, x0, w_avail = .14, 1.0, 5.9
    n = sum((I.cluster == c).sum() for c in cl_imc); sq = (w_avail - gap * (len(cl_imc) - 1)) / n
    rows = ['Non-cancer control'] + MARKERS; top = 1.12; ti = .38
    a = pg.ax(0, 0, 7.0, 2.55, panel='a'); a.set_xlim(0, 7.0); a.set_ylim(2.55, 0); a.axis('off')
    for yy, lab in [(.25, 'Cluster'), (.40, 'Model emphasis (log2)'), (.55, 'IMC cells'), (.66 + ti / 2, 'Example cells')]:
        a.text(x0 - .06, yy, lab, fontsize=5.8, ha='right', va='center')
    for j, r in enumerate(rows):
        a.text(x0 - .06, top + (j + .5) * sq, r, fontsize=5.8, ha='right', va='center')
    x = x0
    for c in cl_imc:
        g = I[I.cluster == c].copy()
        g['_k'] = g[MARKERS].fillna(0).dot(2 ** np.arange(len(MARKERS))[::-1]); g = g.sort_values(['control', '_k'], ascending=[True, False])
        a.text(x, .25, f'{c}', fontsize=6.5, fontweight='bold', va='center')
        a.text(x, .40, f'{emph[c]:+.2f}', fontsize=5.8, va='center')
        a.text(x, .55, f'{len(g)}', fontsize=5.8, va='center')
        wgrp = len(g) * sq
        for j_, t in enumerate([t for t in TH if t['cluster'] == c and t['kind'] == 'IMC'][:2]):
            tw = min(ti, wgrp); tx = x + j_ * (tw + .04)
            if tx + tw > x + max(wgrp, tw) + .001:
                break
            ta = pg.ax(tx, .66, tw, tw); ta.imshow(tz[t['key']], interpolation='lanczos'); ta.set_xticks([]); ta.set_yticks([])
            for s_ in ta.spines.values():
                s_.set_visible(True); s_.set_linewidth(.6); s_.set_color(SPECTRA)
            ta.text(.04, .04, t['label'].replace(' ', '\n'), transform=ta.transAxes, fontsize=5, color='white', va='bottom', linespacing=1.0)
            sx.scalebar(ta, tz[t['key']].shape[0], t['field_um'] / tz[t['key']].shape[0], um=10, lw=.7, y=.1)
            if t['mag'] == '10x':
                ta.text(.96, .96, '10x', transform=ta.transAxes, fontsize=5, color='white', va='top', ha='right')
        for i, r in enumerate(g.itertuples()):
            cx = x + i * sq
            a.add_patch(Rectangle((cx + .008, top + .008), sq - .016, sq - .016, color=CONTROL if r.control else FAINT, lw=0))
            for j, m in enumerate(MARKERS, start=1):
                v = getattr(r, m)
                if pd.isna(v):
                    continue
                a.add_patch(Rectangle((cx + .008, top + j * sq + .008), sq - .016, sq - .016, color=SPECTRA if v == 1 else FAINT, lw=0))
        x += len(g) * sq + gap
    yl = top + len(rows) * sq + .1
    a.add_patch(Rectangle((x0, yl), .1, .1, color=SPECTRA, lw=0)); a.text(x0 + .14, yl + .05, 'Positive', fontsize=5.6, va='center')
    a.add_patch(Rectangle((x0 + .8, yl), .1, .1, color=FAINT, lw=0)); a.text(x0 + .94, yl + .05, 'Negative (blank, not in panel)', fontsize=5.6, va='center')
    a.add_patch(Rectangle((x0 + 2.9, yl), .1, .1, color=CONTROL, lw=0)); a.text(x0 + 3.04, yl + .05, 'Cell from a non-cancer control', fontsize=5.6, va='center')
    a.text(x0 + 4.75, yl + .05, 'Cluster 20: no IMC cells', fontsize=5.6, va='center', color=MUTED)

    # ---------- b: copy-number heatmap, one row per sequenced cell ----------
    pg.letter(.02, 2.62, 'b')
    M, ch, cl, pid, alt = z['M'], z['chrom'], z['cluster'], z['patient'], z['altered']
    gapb, btop, bh, hmin = .1, 2.9, 3.40, .55
    counts = np.array([(cl == c).sum() for c in ORDER], float); avail = bh - gapb * (len(ORDER) - 1)
    rh = avail / counts.sum()
    for _ in range(20):   # proportional heights with a minimum, so each cluster has room for its example images
        hs = np.maximum(hmin, counts * rh); rh *= (avail - hs[counts * rh < hmin].sum()) / max((counts * rh)[counts * rh >= hmin].sum(), 1e-9)
    hs = np.maximum(hmin, counts * rh)
    ts, tx0 = .38, 5.36
    pg.text(tx0, btop - .26, 'Sequenced example cells\n(40x, centred on the cell)', fontsize=5.5, linespacing=1.05)
    y = btop; hx, hw = 1.5, 3.3
    canc = z['cancer']
    import matplotlib as _mpl   # one distinct colour per patient (no reuse): tab20 + tab20b, 40 colours for 28 patients
    PPAL = [_mpl.colors.to_hex(c_) for c_ in list(_mpl.colormaps['tab20'].colors) + list(_mpl.colormaps['tab20b'].colors)]
    assert pid.max() < len(PPAL), 'more patients than distinct colours'
    from matplotlib.colors import ListedColormap, to_rgb
    sx0 = hx + hw + .05; sw = .09
    def strip(x, rows_rgb, top_, hh):
        sa = pg.ax(x, top_, sw, hh); sa.imshow(np.array(rows_rgb)[:, None, :], aspect='auto', interpolation='nearest'); sa.set_xticks([]); sa.set_yticks([])
        for s_ in sa.spines.values():
            s_.set_visible(False)
    for c, h in zip(ORDER, hs):
        k = np.flatnonzero(cl == c)
        if not len(k):
            continue
        for j, t in enumerate([t for t in TH if t['cluster'] == c and t['kind'] == 'WGS'][:3]):
            ta = pg.ax(tx0 + j * (ts + .02), y + .02, ts, ts); ta.imshow(tz[t['key']], interpolation='lanczos'); ta.set_xticks([]); ta.set_yticks([])
            sx.scalebar(ta, tz[t['key']].shape[0], t['field_um'] / tz[t['key']].shape[0], um=10, lw=.7)
            for s_ in ta.spines.values():
                s_.set_visible(True); s_.set_linewidth(.8); s_.set_color(INK if t['kind'] == 'WGS' else SPECTRA)
            if t['mag'] == '10x':
                ta.text(.95, .05, '10x', transform=ta.transAxes, ha='right', va='bottom', fontsize=5, color='white')
            lab = t['label']
            pg.text(tx0 + j * (ts + .02) + ts / 2, y + ts + .04, lab, fontsize=5, ha='center', linespacing=1.05)
        ax = pg.ax(hx, y, hw, h, panel=f'b{c}')
        ax.imshow(M[k], aspect='auto', cmap=CMAP, vmin=-1, vmax=1, interpolation='nearest')
        for b_ in np.flatnonzero(np.diff(pid[k]) != 0):
            ax.axhline(b_ + .5, color=INK, lw=.5)
        for cb in np.flatnonzero(np.diff(ch) != 0):
            ax.axvline(cb + .5, color=RULE, lw=.3)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(True); s_.set_linewidth(.4); s_.set_color(MUTED)
        strip(sx0, [to_rgb(PPAL[p_ % len(PPAL)]) for p_ in pid[k]], y, h)
        strip(sx0 + .12, [to_rgb(sx.CANCER_COL.get(c_, '#9A9A9A')) for c_ in canc[k]], y, h)
        strip(sx0 + .24, [to_rgb(INK if a_ else FAINT) for a_ in alt[k]], y, h)
        two = 'DAPI dim' in nm[c]
        pg.text(.05, y + h / 2 - (.2 if two else .14), f'{c}  {nm[c]}'.replace(', DAPI dim, large', ',\nDAPI dim, large').replace(', VIM+, DAPI dim', ',\nVIM+, DAPI dim'), fontsize=5.8, fontweight='bold', linespacing=1.1)
        pg.text(.05, y + h / 2 + (.06 if two else -.02), f'Emphasis {emph[c]:+.2f}\n{int(alt[k].sum())} of {len(k)} altered, {len(np.unique(pid[k]))} patients', fontsize=5.6, linespacing=1.1)
        y += h + gapb
    last = ax
    mids = [np.flatnonzero(ch == c).mean() for c in range(1, 23)]
    last.set_xticks(mids); last.set_xticklabels([str(c) if c in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22) else '' for c in range(1, 23)], fontsize=5.6)
    last.tick_params(axis='x', length=0, pad=2); last.set_xlabel('Chromosome (autosomes)', fontsize=6)
    for i_, lab_ in enumerate(['Patient', 'Cancer', 'Altered']):
        pg.text(sx0 + .045 + i_ * .12, btop - .04, lab_, fontsize=5.5, rotation=90, ha='center', va='bottom')
    ly = y - .02; pg.text(tx0, ly - .08, 'Cancer type', fontsize=5.5)
    for i_, ct_ in enumerate([c_ for c_ in ['Prostate', 'Breast', 'Lung', 'Bladder'] if c_ in set(canc)]):
        xx, yy = tx0 + (i_ % 2) * .62, ly + .06 + (i_ // 2) * .13
        pg.fig.patches.append(Rectangle((xx / pg.w, 1 - (yy + .09) / pg.h), .08 / pg.w, .08 / pg.h, transform=pg.fig.transFigure, color=sx.CANCER_COL[ct_], lw=0))
        pg.text(xx + .11, yy, ct_, fontsize=5.5)
    cb = pg.ax(hx, btop - .2, .9, .07); cb.imshow(np.linspace(-1, 1, 100)[None], aspect='auto', cmap=CMAP); cb.set_yticks([])
    cb.set_xticks([0, 49.5, 99]); cb.set_xticklabels(['−1', '0', '+1'], fontsize=5.6); cb.tick_params(length=1.5, pad=1)
    pg.text(hx + .98, btop - .23, 'Copy number, log2 ratio to cell median;\nrows, cells; lines separate patients', fontsize=5.6, linespacing=1.05)
    sx.export(pg, 'supplementary', 'SupplementaryFig5_cluster_biology', column_groups=[[f'b{c}' for c in ORDER if (cl == c).any()]])
    I.to_csv(sx.ROOT / 'supplementary/SupplementaryFig5_imc_cells_source.csv', index=False)
    pd.DataFrame(dict(cluster=cl, patient_index=pid, cancer_type=canc, altered=alt)).to_csv(sx.ROOT / 'supplementary/SupplementaryFig5_sequenced_cells_source.csv', index=False)


if __name__ == '__main__':
    build()
