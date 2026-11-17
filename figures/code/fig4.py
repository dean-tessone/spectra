"""Figure 4 - SPECTRA detection does not depend on CK-gated or automated CTC-like cells."""
import glob
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score
import sx
from sx import INK, RULE, SPECTRA, SPECTRA_L, CONTROL, CK, CTC

sys.path.insert(0, str(sx.REPO / 'imaging/cell_images'))
from cell_ext_display import rgb

EX = sx.REPO / 'analyses/figure_examples_20261111'
CLASSES = ['Large Nuclei', 'imCTCs', 'CTCs', 'pcCTCs', 'CECs', 'Megakaryocytes', 'Fibroblasts', 'Lymph', 'Gran', 'Mono']


def build():
    sx.style()
    p = sx.primary_predictions(); m = sx.markers()
    d = p.merge(m[['donor_id', 'CK_gate_per_slide', 'CTC_like_per_slide'] + [c + '_per_slide' for c in CLASSES]], on='donor_id', validate='one_to_one')
    cases = d.label.eq(1)
    pg = sx.Page(7.0, 5.25)

    # ---------- a: detected cancers with no CK-gated cell, and the cells SPECTRA used ----------
    pg.letter(.02, .04, 'a')
    A3 = sx.REPO / 'analyses/figure3_locked6_attribution_20261111'
    ex = pd.read_csv(A3 / 'examples_40x_private.csv', dtype={'slide_id': str}); ex = ex[ex.panel == 'fig4_ckneg_40x']
    tw, gx = .50, .05
    order = pd.read_csv(sx.REPO / 'analyses/figure3_locked6_attribution_20261111/fig4_ckneg_examples_private.csv', dtype=str).donor_id.tolist()
    dd = d.set_index('donor_id')
    for r_, donor in enumerate(order):
        g = ex[ex.example == donor].sort_values('rank').head(7)
        row = dd.loc[donor]
        y0 = .30 + r_ * (tw + .12)
        pg.text(.12, y0 + tw / 2 - .07, sx.short(row.cancer_types), fontsize=6.2, fontweight='bold', va='center')
        pg.text(.12, y0 + tw / 2 + .08, 'CK-gated cells: 0', fontsize=5.6, va='center')
        for j, c_ in enumerate(g.itertuples()):
            ax = pg.ax(1.05 + j * (tw + gx), y0, tw, tw)
            im = np.load(A3 / 'cache_40x' / f'{c_.cell_key}.npz')['image']; h = int(c_.half)
            ax.imshow(rgb(im[c_.cy - h:c_.cy + h + 1, c_.cx - h:c_.cx + h + 1]), interpolation='lanczos'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(SPECTRA); sp.set_linewidth(.9)
            sx.scalebar(ax, 2 * h + 1, sx.UM_PER_PX['40x'], um=5)
    pg.text(1.05, .14, 'Most-attributed cells with 40x re-images (ranked)', fontsize=6.2, va='center')

    # ---------- b: ROC vs marker burden ----------
    pg.letter(4.98, .04, 'b')
    b = pg.ax(5.32, .30, 1.55, 1.55, panel='b')
    curves = [('SPECTRA', d.score, SPECTRA, 1.6, '-'), ('CK gate', d.CK_gate_per_slide, CK, 1.0, '-'),
              ('CTC-like', d.CTC_like_per_slide, CTC, 1.0, '-'), ('CTCs only', d.CTCs_per_slide, CTC, .9, (0, (2, 1.2)))]
    hs = []
    for i, (nm, s, c1, lw, ls) in enumerate(curves):
        f, t, auc = sx.roc(d.label, s); b.step(f * 100, t * 100, where='post', color=c1, lw=lw, ls=ls, zorder=5 - i)
        hs.append(Line2D([], [], color=c1, lw=lw, ls=ls, label=f'{nm} ({auc:.2f})'))
    b.plot([0, 100], [0, 100], color=RULE, lw=.5, ls=(0, (2, 2)))
    b.set_xlim(0, 100); b.set_ylim(0, 100); b.set_xticks([0, 50, 100]); b.set_yticks([0, 50, 100]); b.set_aspect('equal')
    b.set_xlabel('False-positive rate (%)'); b.set_ylabel('Sensitivity (%)')
    b.legend(handles=hs, loc='lower right', fontsize=5.4, handlelength=1.3, labelspacing=.3, borderaxespad=.2)

    # ---------- c: score by CK burden ----------
    pg.letter(.02, 2.95, 'c')
    c = pg.ax(.55, 3.20, 2.25, 1.45, panel='c')
    bins = [(0, 0, '0'), (.5, 1.5, '0.5-1'), (2, 5, '2-5'), (5.5, 1e9, '>5')]
    for i, (lo, hi, lab) in enumerate(bins):
        sel = d.CK_gate_per_slide.between(lo, hi)
        for k, (lab_, off, col) in enumerate([(1, -.18, SPECTRA), (0, .18, CONTROL)]):
            v = d.score[sel & d.label.eq(lab_)].to_numpy()
            if len(v) < 2:
                continue
            vp = c.violinplot([v], positions=[i + off], widths=.33, showextrema=False, bw_method=.35)
            for body in vp['bodies']:
                body.set_facecolor(col); body.set_alpha(.8); body.set_edgecolor('none')
            c.plot([i + off - .1, i + off + .1], [np.median(v)] * 2, color=INK, lw=.9)
            c.text(i + off, -9.2, f'{len(v)}', ha='center', va='bottom', fontsize=5.2)
    c.set_xticks(range(len(bins))); c.set_xticklabels([b_[2] for b_ in bins]); c.set_xlim(-.55, len(bins) - .45)
    c.set_ylim(-9.5, 8.5); c.set_yticks([-8, -4, 0, 4, 8]); c.set_ylabel('SPECTRA score (log-odds)')
    c.set_xlabel('CK-gated cells per slide'); c.tick_params(axis='x', length=0)
    c.legend(handles=[Patch(fc=SPECTRA, alpha=.8, label='Cancers'), Patch(fc=CONTROL, alpha=.8, label='Controls'), Line2D([], [], color=INK, lw=.9, label='Median')],
             loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=5.6, borderaxespad=0, handlelength=1.2)

    # ---------- d: breast stage ----------
    pg.letter(2.95, 2.95, 'd')
    e = pg.ax(3.3, 3.20, 1.2, 1.45, panel='d')
    st = pd.read_csv(sx.TEST / 'clinical_stratum_AUROC.csv'); st = st[(st.endpoint == 'primary') & (st.stratification == 'breast_early_late')].set_index('stratum')
    mk = pd.read_csv(sx.REPO / 'analyses/clinical_cohort_annotations_20261110/markers/breast_stage_CTC_vs_CK_AUROC.csv'); mk = mk[mk.stratification.eq('breast_early_late')].set_index(['stratum', 'marker'])
    series = [('SPECTRA', SPECTRA, lambda s: st.loc[s, 'AUROC']), ('CK gate', CK, lambda s: mk.loc[(s, 'CK_gate'), 'AUROC']), ('CTC-like', CTC, lambda s: mk.loc[(s, 'CTC_like'), 'AUROC'])]
    for i, s in enumerate(['Early', 'Late']):
        for j, (nm, c1, fn) in enumerate(series):
            e.bar(i + (j - 1) * .26, fn(s) - .5, bottom=.5, width=.24, color=c1)
    e.axhline(.5, color=INK, lw=.6)
    e.set_xticks([0, 1]); e.set_xticklabels([f'Early\n(n={int(st.loc["Early", "cases"])})', f'Late\n(n={int(st.loc["Late", "cases"])})'], linespacing=1.2)
    e.set_ylim(.4, .9); e.set_yticks([.4, .5, .6, .7, .8, .9]); e.set_ylabel('Breast AUROC'); e.tick_params(axis='x', length=0)
    e.legend(handles=[Patch(fc=c1, label=nm) for nm, c1, _ in series], loc='lower left', bbox_to_anchor=(-.1, 1.0), ncol=3, fontsize=5.2, borderaxespad=0,
             handlelength=.8, handletextpad=.25, columnspacing=.4)

    # ---------- e: cell removal ----------
    R = pd.read_csv(sx.REPO / 'analyses/figure3_locked6_attribution_20261111/removal_summary.csv').set_index('condition')
    conds = [('CK_gate', 'CK-gated cells'), ('state8', 'CK-bright, CD45-low'), ('CTC_like', 'All CTC-like'), ('top10', 'Top-10 attributed'), ('top50', 'Top-50 attributed')]
    pg.letter(4.62, 2.95, 'e')
    f2 = pg.ax(5.62, 3.20, 1.28, 1.45, panel='e')
    for i, (c_, nm) in enumerate(conds):
        f2.barh(i - .19, R.loc[c_].sens90 * 100, height=.36, color=SPECTRA)
        f2.barh(i + .19, R.loc[c_ + '_matched_random'].sens90 * 100, height=.36, color='#C9D1D6')
    f2.axvline(R.loc['none'].sens90 * 100, color=INK, lw=.7, ls=(0, (3, 2)))
    f2.set_yticks(range(len(conds))); f2.set_yticklabels([c[1] for c in conds], fontsize=5.6); f2.set_ylim(len(conds) - .5, -.6)
    f2.tick_params(axis='y', length=0); f2.spines['left'].set_visible(False)
    f2.set_xlim(0, 70); f2.set_xticks([0, 25, 50]); f2.set_xlabel('Sensitivity at 90%\nspecificity (%)', linespacing=1.2)
    f2.legend(handles=[Patch(fc=SPECTRA, label='Remove these cells'), Patch(fc='#C9D1D6', label='Remove random cells'), Line2D([], [], color=INK, lw=.7, ls=(0, (3, 2)), label='Intact')],
              loc='lower left', bbox_to_anchor=(-.62, 1.0), ncol=2, fontsize=5.3, borderaxespad=0, handlelength=1.0, columnspacing=.5)
    sx.export(pg, 'figure4', 'Figure4', row_groups=[['c', 'd', 'e']],
              exemptions=[dict(panels=['c', 'd', 'e'], checks=['panel-width', 'horizontal-gutter'], reason='different chart types')])
    R.to_csv(sx.ROOT / 'figure4/cell_removal_summary_source.csv')
    d[['label', 'cancer_types', 'score', 'CK_gate_per_slide', 'CTC_like_per_slide', 'CTCs_per_slide']].to_csv(sx.ROOT / 'figure4/patient_score_vs_burden_deidentified_source.csv', index=False)


if __name__ == '__main__':
    build()
