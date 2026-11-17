"""Supplementary Fig. 3 - attribution sanity checks: cascading model randomization and seed stability."""
import json
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import sx
from sx import INK, MUTED, RULE, SPECTRA, CONTROL

A = sx.REPO / 'analyses/attribution_sanity_20261111'


def build():
    sx.style()
    pg = sx.Page(7.0, 4.95)
    s = pd.read_csv(A / 'summary_median.csv', index_col=0)
    ref = pd.read_csv(A / 'between_seed_reference_median.csv', index_col=0).loc['seed vs seed']

    # a: cascading randomization
    pg.letter(.02, .04, 'a')
    a = pg.ax(.55, .25, 3.15, 1.75, panel='a')
    x = np.arange(len(s))
    for col, lab, c1, mk in [('spearman_signed', 'Rank correlation, signed', SPECTRA, 'o'), ('spearman_abs', 'Rank correlation, magnitude', '#2C7FB8', 's'),
                             ('top50_overlap', 'Top-50 cell overlap', INK, 'D')]:
        a.plot(x, s[col], color=c1, lw=1.2, marker=mk, ms=3.2, label=lab)
    a.axhline(ref.top50_overlap, color=INK, lw=.7, ls=(0, (3, 2)))
    a.text(.9, ref.top50_overlap - .03, 'Top-50 overlap between\nindependently trained seeds', ha='left', va='top', fontsize=5.2, linespacing=1.15)
    a.axhline(0, color=RULE, lw=.6)
    a.set_xticks(x); a.set_xticklabels(['Trained', 'Classifier', '+Norm', '+B6', '+B5', '+B4', '+B3', '+B2', '+B1', '+Input'], rotation=45, ha='right', fontsize=5.6)
    a.set_ylim(-.2, 1.05); a.set_yticks([0, .5, 1]); a.set_ylabel('Agreement with\ntrained-model map', linespacing=1.2)
    a.set_xlabel('Layers re-initialized (cumulative, from the top)')
    a.legend(loc='upper right', fontsize=5.4, handlelength=1.4, borderaxespad=.2, labelspacing=.3)

    # b: Figure 3b emphasis per seed vs ensemble
    pg.letter(4.15, .04, 'b')
    b = pg.ax(4.75, .25, 1.75, 1.75, panel='b')
    e = pd.read_csv(A / 'figure3_emphasis_by_seed.csv', index_col=0)
    for sd, c1, mk in [('17', SPECTRA, 'o'), ('29', CONTROL, 's'), ('43', '#6A4C93', 'D')]:
        b.scatter(e.ensemble, e[sd], s=9, color=c1, marker=mk, lw=0, alpha=.85, label=f'Seed {sd}')
    lim = (-2.2, 1.4)
    b.plot(lim, lim, color=RULE, lw=.6, ls=(0, (2, 2))); b.axhline(0, color=RULE, lw=.5); b.axvline(0, color=RULE, lw=.5)
    b.set_xlim(lim); b.set_ylim(lim); b.set_aspect('equal')
    b.set_xlabel('Emphasis, three-seed ensemble\n(log2, as Fig. 3b)', linespacing=1.2); b.set_ylabel('Emphasis, single seed (log2)')
    b.legend(loc='upper left', fontsize=5.4, handletextpad=.2, borderaxespad=.2, labelspacing=.3)
    # c: how concentrated the evidence is (median and IQR over patients)
    pg.letter(.02, 2.80, 'c')
    c = pg.ax(.55, 3.00, 3.15, 1.55, panel='c')
    cells = pd.read_parquet(sx.ATTR / 'all_cells.parquet', columns=['donor_id', 'label', 'relevance'])
    xs = np.unique(np.round(np.logspace(0, 3.3, 60)).astype(int)); cur = {0: [], 1: []}; need = []
    for _, g in cells.groupby('donor_id'):
        pos = np.sort(g.relevance.clip(lower=0).to_numpy())[::-1]; cum = np.cumsum(pos) / pos.sum()
        cur[int(g.label.iloc[0])].append(cum[np.minimum(xs, len(cum)) - 1] * 100)
        need.append(dict(label=int(g.label.iloc[0]), cells=len(g), **{f'cells_{q}': int(np.searchsorted(cum, q / 100) + 1) for q in (50, 75, 90)}))
    need = pd.DataFrame(need); rows = []
    for lab, col, name in [(1, SPECTRA, 'Cancers'), (0, CONTROL, 'Non-cancer controls')]:
        arr = np.array(cur[lab]); q = np.percentile(arr, [25, 50, 75], axis=0)
        c.fill_between(xs, q[0], q[2], color=col, alpha=.18, lw=0); c.plot(xs, q[1], color=col, lw=1.2, label=f'{name} ({len(arr)})')
        rows += [dict(group=name, cells=int(x), q25=a_, median=b_, q75=c_) for x, a_, b_, c_ in zip(xs, *q)]
    for qq in (50, 75, 90):
        n = need[need.label == 1][f'cells_{qq}'].median()
        c.plot([1, n, n], [qq, qq, 0], color=INK, lw=.5, ls=(0, (1, 1.5)))
        c.text(n * 1.12, qq - 3, f'{n:.0f} cells', fontsize=5.6, va='top')
    c.set_xscale('log'); c.set_xlim(1, 2000); c.set_ylim(0, 100); c.set_xticks([1, 10, 100, 1000]); c.set_xticklabels(['1', '10', '100', '1,000'])
    c.set_yticks([0, 50, 75, 90, 100]); c.set_xlabel("Cells, ranked by attribution (median {:,.0f} per patient)".format(need.cells.median()))
    c.set_ylabel('Positive evidence\ncaptured (%)', linespacing=1.2); c.legend(loc='lower right', fontsize=5.6, handlelength=1.4, borderaxespad=.2)
    pd.DataFrame(rows).to_csv(sx.ROOT / 'supplementary/SupplementaryFig3_evidence_concentration_source.csv', index=False)

    sx.export(pg, 'supplementary', 'SupplementaryFig3_attribution_sanity', row_groups=[['a', 'b']], column_groups=[['a', 'c']],
              exemptions=[dict(panels=['a', 'b', 'c'], checks=['panel-width', 'horizontal-gutter', 'column', 'vertical-gutter'], reason='line plots and square scatter')])
    s.to_csv(sx.ROOT / 'supplementary/SupplementaryFig3_randomization_source.csv'); e.to_csv(sx.ROOT / 'supplementary/SupplementaryFig3_emphasis_by_seed_source.csv')


if __name__ == '__main__':
    build()
