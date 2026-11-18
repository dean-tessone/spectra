"""Supplementary Fig. 8 - UMAP of the locked SPECTRA patient summary (CLS) embedding, all 2,550 primary-analysis patients
(). Data: analyses/revision_20261117/cls_umap (robustness/cls_umap.py).
a, cancer versus non-cancer control; b, cancer type."""
import numpy as np, pandas as pd
from matplotlib.lines import Line2D
import sx
from sx import SPECTRA, SPECTRA_L, CONTROL, INK
CONTROL_L = '#F4C79A'

D = sx.REPO / 'analyses/revision_20261117/cls_umap'
MAJOR = ['Prostate', 'Breast', 'Lung', 'Bladder', 'Upper Tract Urothelial Carcinoma', 'Multiple Myeloma', 'Pancreatic', 'Uveal Melanoma']
OTHER = '#9A9A9A'


def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def build():
    sx.style()
    d = pd.read_csv(D / 'cls_umap_source.csv')
    rng = np.random.default_rng(0)
    pg = sx.Page(7.0, 3.35)
    S = 2.6

    # ---------------- a: cancer vs non-cancer
    a = pg.ax(.40, .55, 2.45, 2.45, panel='a'); pg.letter(.02, .10, 'a')
    ca, co = d[d.label == 1], d[d.label == 0]
    tr, te = d.split == 'train', d.split == 'test'
    groups = [((d.label == 1) & tr, SPECTRA_L, 'none', S, .6, f'Cancer, training ({((d.label == 1) & tr).sum():,})'),
              ((d.label == 0) & tr, CONTROL_L, 'none', S * 2.2, .9, f'Control, training ({((d.label == 0) & tr).sum():,})'),
              ((d.label == 1) & te, SPECTRA, 'none', S * 1.6, .85, f'Cancer, held-out test ({((d.label == 1) & te).sum():,})'),
              ((d.label == 0) & te, CONTROL, INK, S * 4.5, 1., f'Control, held-out test ({((d.label == 0) & te).sum():,})')]
    hs = []
    for z, (m, col, ec, sz, al, lab) in enumerate(groups):
        g = d[m]; a.scatter(g.umap1, g.umap2, s=sz, color=col, edgecolor=ec, lw=.35 if ec != 'none' else 0, alpha=al, rasterized=True, zorder=z + 1)
        hs.append(Line2D([], [], marker='o', ls='', ms=3.5, color=col, mec=ec if ec != 'none' else col, mew=.5, label=lab))
    frame(a)
    a.legend(handles=hs, loc='upper left', bbox_to_anchor=(0, 1.16), ncol=2, frameon=False, fontsize=5.8, handletextpad=.2, columnspacing=.8, borderaxespad=0)

    # ---------------- b: cancer type
    b = pg.ax(3.22, .55, 2.45, 2.45, panel='b'); pg.letter(2.98, .10, 'b')
    oth = ca[~ca.cancer_types.isin(MAJOR)]
    b.scatter(oth.umap1, oth.umap2, s=S, color=OTHER, lw=0, alpha=.8, rasterized=True, zorder=1)
    order = ca[ca.cancer_types.isin(MAJOR)].sample(frac=1, random_state=0)   # shuffled so no type is drawn uniformly on top
    b.scatter(order.umap1, order.umap2, s=S, c=[sx.CANCER_COL[t] for t in order.cancer_types], lw=0, alpha=.7, rasterized=True, zorder=2)
    b.scatter(co.umap1, co.umap2, s=S * 2.2, color=CONTROL, lw=0, alpha=.95, rasterized=True, zorder=3)
    frame(b)
    hs = [Line2D([], [], marker='o', ls='', ms=3.5, color=sx.CANCER_COL[t], label=f'{sx.short(t)} ({(ca.cancer_types == t).sum():,})') for t in MAJOR]
    hs += [Line2D([], [], marker='o', ls='', ms=3.5, color=OTHER, label=f'Other cancers ({len(oth):,})'),
           Line2D([], [], marker='o', ls='', ms=3.5, color=CONTROL, label=f'Non-cancer controls ({len(co):,})')]
    b.legend(handles=hs, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=5.8, handletextpad=.2, labelspacing=.55, borderaxespad=0)

    sx.export(pg, 'supplementary', 'SupplementaryFig8_cls_umap', row_groups=[['a', 'b']])
    return d


if __name__ == '__main__':
    build()
