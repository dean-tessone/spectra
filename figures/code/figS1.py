"""Supplementary Fig. 1 - every held-out patient: SPECTRA discrimination, operating points and clinical difficulty."""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import sx
from sx import INK, MUTED, RULE, FAINT, SPECTRA, SPECTRA_D, SPECTRA_L, CONTROL, CK, CTC


def swarm_offsets(x, width=.36, bin_w=.16, seed=0):
    """Deterministic density-scaled vertical spread (sina-style) for a strip of points."""
    x = np.asarray(x); off = np.zeros(len(x))
    bins = np.floor(x / bin_w).astype(int)
    counts = pd.Series(bins).value_counts()
    peak = counts.max()
    rng = np.random.default_rng(seed)
    for b, idx in pd.Series(np.arange(len(x))).groupby(bins):
        k = len(idx); half = width * np.sqrt(k / peak)
        off[idx.to_numpy()] = (rng.permutation(k) / max(k - 1, 1) - .5) * 2 * half if k > 1 else 0
    return off


def draw_strip(pg, X, Y, Wd, Ht, letter_xy=(0.02, 0.10), legend_xy=None, table=True, letter='a', show_n=False):
    p = sx.primary_predictions(); ops = sx.operating_points(); t90, t95 = sx.logit(ops[.9]), sx.logit(ops[.95])
    rep = pd.read_csv(sx.TEST / 'primary_per_cancer_reporting.csv').set_index('cancer_type')
    order = rep.sort_values(['cases'], ascending=False).index.tolist()
    rows = ['Control'] + order

    # ---------------- a: every patient ----------------
    a = pg.ax(X, Y, Wd, Ht, panel=letter); pg.letter(letter_xy[0], letter_xy[1], letter)
    a.axvspan(t90, 8.2, color=SPECTRA, alpha=.06, lw=0, zorder=0)
    for i, t in enumerate(rows):
        g = p[p.cancer_types.eq(t)] if t != 'Control' else p[p.label.eq(0)]
        y = i + swarm_offsets(g.score.to_numpy(), seed=i)
        if t == 'Control':
            c = np.where(g.called90, INK, CONTROL)
            a.scatter(g.score, y, s=5.5, c=c, lw=0, alpha=.95, zorder=3)
        else:
            c = np.where(g.called90, SPECTRA, SPECTRA_L)
            a.scatter(g.score, y, s=3.2 if len(g) > 100 else 5, c=c, lw=0, alpha=.9, zorder=3)
        a.plot([g.score.median()] * 2, [i - .42, i + .42], color=INK, lw=.9, zorder=4, solid_capstyle='butt')
        if i == 0:
            a.axhline(.5, color=RULE, lw=.6, zorder=1)
    a.set_ylim(len(rows) - .45, -.6)
    a.set_yticks(range(len(rows)))
    nrow = {t: int((p.label.eq(0) if t == 'Control' else p.cancer_types.eq(t)).sum()) for t in rows}
    lab = [f'{sx.short(t)} (n = {nrow[t]})' if show_n else f'{sx.short(t)}' for t in rows]
    a.set_yticklabels(lab)
    a.tick_params(axis='y', length=0, pad=3)
    a.spines['left'].set_visible(False)
    for ln, lab2, yy in [(t90, '90%', -.95), (t95, '95%', -.95)]:
        a.axvline(ln, color=SPECTRA_D, lw=.6, ls=(0, (2, 1.6)), zorder=2)
    a.text(t90 - .12, -.62, '90%', ha='right', va='bottom', fontsize=5.6, color=SPECTRA_D)
    a.text(t95 + .12, -.62, '95%', ha='left', va='bottom', fontsize=5.6, color=SPECTRA_D)
    a.set_xlim(-8.2, 8.2); a.set_xticks([-8, -4, 0, 4, 8])
    a.set_xlabel('SPECTRA patient score (log-odds)')
    if table:
        # inline table to the right of the strips (x in inches from the axes' right edge)
        from matplotlib.transforms import blended_transform_factory
        tr = blended_transform_factory(a.transAxes, a.transData)
        W = Wd
        cols = [(1 + .20 / W, 'n'), (1 + .52 / W, 'AUROC'), (1 + .84 / W, 'Sens.\n90%'), (1 + 1.13 / W, 'Sens.\n95%')]
        for xx, h in cols:
            a.text(xx, -.62, h, ha='center', va='bottom', fontsize=5.6, color=MUTED, clip_on=False, linespacing=1.2, transform=tr)
        for i, t in enumerate(rows):
            if t == 'Control':
                vals = ['44', '', '40/44', '42/44']; colr = INK
            else:
                r = rep.loc[t]; colr = INK if r.macro_eligible else MUTED
                vals = [f'{int(r.cases)}', f'{r.auroc:.2f}', f'{r.sensitivity_at_90spec_target*100:.0f}%', f'{r.sensitivity_at_95spec_target*100:.0f}%']
            for j, ((xx, _), v) in enumerate(zip(cols, vals)):
                a.text(xx, i, v, ha='center', va='center', fontsize=5.9, color=colr, clip_on=False, transform=tr,
                       fontweight='bold' if (j == 1 and t != 'Control' and rep.loc[t].macro_eligible) else 'normal')
        yy = len(rows) - .05
        for (xx, _), v in zip(cols, ['713', '0.82', '57%', '40%']):
            a.text(xx, yy, v, ha='center', va='center', fontsize=5.9, color=SPECTRA_D, fontweight='bold', clip_on=False, transform=tr)
        a.text(-.03, yy, 'All cancers', ha='right', va='center', fontsize=6, color=SPECTRA_D, fontweight='bold', clip_on=False, transform=tr)

    a.set_ylim(len(rows) + .35 if table else len(rows) - .45, -.6)
    h = [Line2D([], [], marker='o', ls='', ms=2.8, mfc=SPECTRA, mec='none', label='Cancer, called at 90% specificity'),
         Line2D([], [], marker='o', ls='', ms=2.8, mfc=SPECTRA_L, mec='none', label='Cancer, not called'),
         Line2D([], [], marker='o', ls='', ms=2.8, mfc=CONTROL, mec='none', label='Non-cancer control'),
         Line2D([], [], color=INK, lw=.9, label='Median')]
    if legend_xy == 'top':
        a.legend(handles=h, loc='lower left', bbox_to_anchor=(0, 1.07), ncol=4, handletextpad=.3, columnspacing=1.2, fontsize=6, borderaxespad=0)
    elif legend_xy is None:
        a.legend(handles=h, loc='upper left', bbox_to_anchor=(-.30, -.115), ncol=4, handletextpad=.3, columnspacing=1.0, fontsize=5.8)
    else:
        a.legend(handles=h, loc='upper left', bbox_to_anchor=(legend_xy[0] / pg.w, 1 - legend_xy[1] / pg.h), bbox_transform=pg.fig.transFigure, ncol=1, handletextpad=.4, fontsize=5.8)

    return p, a


def build():
    sx.style(); pg = sx.Page(5.2, 4.55)
    p, _ = draw_strip(pg, 1.02, 0.42, 2.55, 3.43)
    sx.export(pg, 'supplementary', 'SupplementaryFig1_patient_scores')


if __name__ == '__main__':
    build()
