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
    # marker counts per patient: summed over the patient's analysed slides (two for 732 of 757 test patients)
    d = p.merge(m[['donor_id', 'CK_gate', 'CTC_like'] + CLASSES], on='donor_id', validate='one_to_one')
    pg = sx.Page(7.0, 5.62)
    Y1, H1 = .32, 1.62            # top row: quantitative panels
    Y2 = 2.78                     # bottom row: removal and images

    # ---------- a: canonical CTC counts versus SPECTRA, identical patients ----------
    pg.letter(.02, .04, 'a')
    a = pg.ax(.50, Y1, H1, H1, panel='a')
    curves = [('SPECTRA', d.score, SPECTRA, 1.6, '-'), ('CK-gated', d.CK_gate, CK, 1.0, '-'),
              ('CTC-like', d.CTC_like, CTC, 1.0, '-'), ('CTCs only', d.CTCs, CTC, .9, (0, (2, 1.2)))]
    hs = []
    for i, (nm, s_, c1, lw, ls) in enumerate(curves):
        f, t, auc = sx.roc(d.label, s_); a.step(f * 100, t * 100, where='post', color=c1, lw=lw, ls=ls, zorder=5 - i)
        hs.append(Line2D([], [], color=c1, lw=lw, ls=ls, label=f'{nm} ({auc:.2f})'))
    a.plot([0, 100], [0, 100], color=RULE, lw=.5, ls=(0, (2, 2)))
    a.set_xlim(0, 100); a.set_ylim(0, 100); a.set_xticks([0, 50, 100]); a.set_yticks([0, 50, 100]); a.set_aspect('equal')
    a.set_xlabel('False-positive rate (%)'); a.set_ylabel('Sensitivity (%)')
    a.legend(handles=hs, loc='lower right', fontsize=5.4, handlelength=1.3, labelspacing=.3, borderaxespad=.2, title='AUROC', title_fontsize=5.4)

    # ---------- b: SPECTRA score by CK-gated burden ----------
    pg.letter(2.42, .04, 'b')
    c = pg.ax(2.92, Y1, 2.10, H1, panel='b')
    bins = [(0, 0, '0'), (1, 2, '1–2'), (3, 10, '3–10'), (11, 1e9, '>10')]
    for i, (lo, hi, lab) in enumerate(bins):
        sel = d.CK_gate.between(lo, hi)
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
    c.set_xlabel('CK-gated cells per patient'); c.tick_params(axis='x', length=0)
    c.legend(handles=[Patch(fc=SPECTRA, alpha=.8, label='Cancers'), Patch(fc=CONTROL, alpha=.8, label='Controls'), Line2D([], [], color=INK, lw=.9, label='Median')],
             loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=5.6, borderaxespad=0, handlelength=1.2)

    # ---------- c: breast cancer by stage ----------
    pg.letter(5.28, .04, 'c')
    e = pg.ax(5.80, Y1, 1.12, H1, panel='c')
    st = pd.read_csv(sx.TEST / 'clinical_stratum_AUROC.csv'); st = st[(st.endpoint == 'primary') & (st.stratification == 'breast_early_late')].set_index('stratum')
    led = pd.read_csv(sx.REPO / 'analyses/clinical_cohort_annotations_20261110/patient_clinical_ledger.csv', dtype={'donor_id': str})[['donor_id', 'breast_early_late']]
    bd = d.merge(led, on='donor_id', how='left'); ctl = bd[bd.label.eq(0)]
    def mk(s, col):   # per-patient marker count, breast cancers of one stage versus the 44 controls
        cs = bd[bd.label.eq(1) & bd.cancer_types.eq('Breast') & bd.breast_early_late.eq(s)]
        assert len(cs) == st.loc[s, 'cases']
        return roc_auc_score(np.r_[np.ones(len(cs)), np.zeros(len(ctl))], np.r_[cs[col], ctl[col]])
    series = [('SPECTRA', SPECTRA, lambda s: st.loc[s, 'AUROC']), ('CK-gated', CK, lambda s: mk(s, 'CK_gate')), ('CTC-like', CTC, lambda s: mk(s, 'CTC_like'))]
    pd.DataFrame([dict(stage=s, cases=int(st.loc[s, 'cases']), controls=len(ctl), **{nm: fn(s) for nm, _, fn in series}) for s in ['Early', 'Late']]).to_csv(
        sx.ROOT / 'figure4/breast_stage_AUROC_source.csv', index=False)
    for i, s in enumerate(['Early', 'Late']):
        for j, (nm, c1, fn) in enumerate(series):
            e.bar(i + (j - 1) * .26, fn(s) - .5, bottom=.5, width=.24, color=c1)
    e.axhline(.5, color=INK, lw=.6)
    e.set_xticks([0, 1]); e.set_xticklabels([f'Early\n(n={int(st.loc["Early", "cases"])})', f'Late\n(n={int(st.loc["Late", "cases"])})'], linespacing=1.2)
    e.set_ylim(.4, .9); e.set_yticks([.4, .5, .6, .7, .8, .9]); e.set_ylabel('Breast AUROC'); e.tick_params(axis='x', length=0)
    e.legend(handles=[Patch(fc=c1, label=nm) for nm, c1, _ in series], loc='lower left', bbox_to_anchor=(-.32, 1.07), ncol=3, fontsize=5.2, borderaxespad=0,
             handlelength=.8, handletextpad=.25, columnspacing=.4)

    # ---------- d: cell removal on the locked model ----------
    R = pd.read_csv(sx.REPO / 'analyses/figure3_locked6_attribution_20261111/removal_summary.csv').set_index('condition')
    conds = [('CK_gate', 'CK-gated cells'), ('state8', 'CK-bright, CD45-low'), ('CTC_like', 'All CTC-like'), ('top10', 'Top-10 attributed'), ('top50', 'Top-50 attributed')]
    pg.letter(.02, Y2 - .30, 'd')
    f2 = pg.ax(1.02, Y2 + .08, 1.30, 2.22, panel='d')
    for i, (c_, nm) in enumerate(conds):
        f2.barh(i - .19, R.loc[c_].sens90 * 100, height=.36, color=SPECTRA)
        f2.barh(i + .19, R.loc[c_ + '_matched_random'].sens90 * 100, height=.36, color='#C9D1D6')
    f2.axvline(R.loc['none'].sens90 * 100, color=INK, lw=.7, ls=(0, (3, 2)))
    f2.set_yticks(range(len(conds))); f2.set_yticklabels([c[1] for c in conds], fontsize=5.6); f2.set_ylim(len(conds) - .5, -.6)
    f2.tick_params(axis='y', length=0); f2.spines['left'].set_visible(False)
    f2.set_xlim(0, 70); f2.set_xticks([0, 25, 50]); f2.set_xlabel('Sensitivity at 90%\nspecificity (%)', linespacing=1.2)
    f2.legend(handles=[Patch(fc=SPECTRA, label='Remove these cells'), Patch(fc='#C9D1D6', label='Remove random cells'), Line2D([], [], color=INK, lw=.7, ls=(0, (3, 2)), label='Intact')],
              loc='lower left', bbox_to_anchor=(-.72, 1.0), ncol=2, fontsize=5.3, borderaxespad=0, handlelength=1.0, columnspacing=.5)

    # ---------- e: what the cells driving CK-negative detections look like ----------
    pg.letter(2.62, Y2 - .30, 'e')
    A3 = sx.REPO / 'analyses/figure3_locked6_attribution_20261111'
    ex = pd.read_csv(A3 / 'examples_40x_private.csv', dtype={'slide_id': str}); ex = ex[ex.panel == 'fig4_ckneg_40x']
    dd = d.set_index('donor_id')
    order = sorted(ex.example.unique(), key=lambda k: dd.loc[k, 'cancer_types'])     # one detected cancer per row, by cancer type
    tw, gx, gy, X0 = .50, .05, .10, 3.62
    pg.text(X0, Y2 - .22, 'Detected cancers with no CK-gated cell: most-attributed cells (40x, ranked)', fontsize=6.0, va='center')
    for r_, donor in enumerate(order):
        g = ex[ex.example == donor].sort_values('rank').head(6)
        row = dd.loc[donor]
        y0 = Y2 + r_ * (tw + gy)
        pg.text(2.72, y0 + tw / 2 - .07, sx.short(row.cancer_types), fontsize=6.2, fontweight='bold', va='center')
        pg.text(2.72, y0 + tw / 2 + .08, '0 CK-gated cells', fontsize=5.6, va='center')
        for j, c_ in enumerate(g.itertuples()):
            ax = pg.ax(X0 + j * (tw + gx), y0, tw, tw)
            im = np.load(A3 / 'cache_40x' / f'{c_.cell_key}.npz')['image']; h = int(c_.half)
            ax.imshow(rgb(im[c_.cy - h:c_.cy + h + 1, c_.cx - h:c_.cx + h + 1]), interpolation='lanczos'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(SPECTRA); sp.set_linewidth(.9)
            sx.scalebar(ax, 2 * h + 1, sx.UM_PER_PX['40x'], um=5)
    sx.export(pg, 'figure4', 'Figure4', row_groups=[['a', 'b', 'c']],
              exemptions=[dict(panels=['a', 'b', 'c'], checks=['panel-width', 'horizontal-gutter'], reason='different chart types')])
    R.to_csv(sx.ROOT / 'figure4/cell_removal_summary_source.csv')
    d[['label', 'cancer_types', 'score', 'CK_gate', 'CTC_like', 'CTCs']].rename(columns=lambda c: c + '_per_patient' if c in ('CK_gate', 'CTC_like', 'CTCs') else c).to_csv(sx.ROOT / 'figure4/patient_score_vs_burden_deidentified_source.csv', index=False)


if __name__ == '__main__':
    build()
