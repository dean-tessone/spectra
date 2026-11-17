"""Figure 2 - detection on the held-out test cohort: pooled ROC, per-cancer ROCs, and every patient's score."""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import sx
from sx import INK, RULE, SPECTRA
from figS1 import draw_strip

TYPES = ['Lung', 'Pancreatic', 'Breast', 'UTUC', 'Bladder', 'Prostate', 'Myeloma & related']
FULL = {'UTUC': 'Upper Tract Urothelial Carcinoma', 'Myeloma & related': 'Multiple Myeloma'}


def build():
    sx.style()
    pg = sx.Page(7.0, 6.0)
    p = sx.primary_predictions()

    # a: pooled ROC
    pg.letter(.02, .04, 'a')
    a = pg.ax(.55, .25, 1.9, 1.9, panel='a')
    f, t, auc = sx.roc(p.label, p.score)
    a.fill_between(f * 100, t * 100, step='post', color=SPECTRA, alpha=.12, lw=0); a.step(f * 100, t * 100, where='post', color=SPECTRA, lw=1.6)
    a.plot([0, 100], [0, 100], color=RULE, lw=.6, ls=(0, (2, 2)))
    a.set_xlim(0, 100); a.set_ylim(0, 100); a.set_xticks([0, 50, 100]); a.set_yticks([0, 50, 100]); a.set_aspect('equal')
    a.set_xlabel('False-positive rate (%)'); a.set_ylabel('Sensitivity (%)')
    a.text(96, 5, f'All cancers\nAUROC {auc:.2f}', ha='right', va='bottom', fontsize=6.5, linespacing=1.2)

    # b: per-cancer ROCs
    pg.letter(2.95, .04, 'b')
    b = pg.ax(3.45, .25, 1.9, 1.9, panel='b')
    hs = []
    for tname in TYPES:
        key = FULL.get(tname, tname)
        g = p[p.label.eq(0) | p.cancer_types.eq(key)]
        ff, tt, au = sx.roc(g.label, g.score); col = sx.CANCER_COL[key]
        b.step(ff * 100, tt * 100, where='post', color=col, lw=1.1)
        hs.append(Line2D([], [], color=col, lw=1.6, label=f'{tname} ({au:.2f})'))
    b.plot([0, 100], [0, 100], color=RULE, lw=.6, ls=(0, (2, 2)))
    b.set_xlim(0, 100); b.set_ylim(0, 100); b.set_xticks([0, 50, 100]); b.set_yticks([0, 50, 100]); b.set_aspect('equal')
    b.set_xlabel('False-positive rate (%)'); b.set_ylabel('Sensitivity (%)')
    b.legend(handles=hs, loc='center left', bbox_to_anchor=(1.04, .5), fontsize=6, handlelength=1.3, labelspacing=.6, borderaxespad=0, title='AUROC', title_fontsize=6)

    # c: every patient
    draw_strip(pg, 1.10, 2.95, 5.75, 2.65, letter_xy=(.02, 2.55), table=False, letter='c', legend_xy='top')
    sx.export(pg, 'figure2', 'Figure2', row_groups=[['a', 'b']])
    p[['label', 'cancer_types', 'score', 'called90', 'called95']].sort_values(['cancer_types', 'score']).to_csv(sx.ROOT / 'figure2/patient_scores_deidentified_source.csv', index=False)


if __name__ == '__main__':
    build()
