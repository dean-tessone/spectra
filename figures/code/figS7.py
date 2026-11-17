"""Supplementary Fig. 6 - cell embeddings carry diagnostic information within clusters, beyond cluster abundance
(attribution/atlas/embedding_beyond_composition.py)."""
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import sx
from sx import INK, MUTED, RULE, SPECTRA, SPECTRA_L
from fig3 import PAL, cap, L


def build():
    sx.style()
    w = pd.read_csv(L / 'within_cluster_embedding_auc.csv')
    names = pd.read_csv(L / 'cluster_names.csv', index_col=0).name
    w = w.sort_values('within_cluster_embedding_auc', ascending=False).reset_index(drop=True)
    K = len(w)
    pg = sx.Page(7.0, 3.55)

    # ---------- a: abundance versus within-cluster embedding AUROC, every cluster ----------
    pg.letter(.02, .04, 'a')
    y0, h = .42, 2.75
    nm = pg.ax(.10, y0, 1.95, h); nm.set_xlim(0, 1); nm.set_ylim(K - .5, -.5); nm.axis('off')
    a = pg.ax(2.10, y0, 2.15, h, panel='a')
    for i, r in w.iterrows():
        c = int(r.cluster)
        nm.scatter([.05], [i], s=16, color=PAL[c % len(PAL)], lw=0)
        nm.text(.12, i, f'{int(r.number)}', fontsize=5.6, va='center', fontweight='bold')
        nm.text(.24, i, cap(names[c]), fontsize=5.6, va='center')
        a.plot([r.abundance_auc, r.within_cluster_embedding_auc], [i, i], color=RULE, lw=1.0, zorder=1, solid_capstyle='butt')
        a.scatter([r.abundance_auc], [i], s=13, facecolor='white', edgecolor=MUTED, lw=.8, zorder=2)
        sig = r.permutation_q < .05; nom = r.permutation_p < .05
        a.scatter([r.within_cluster_embedding_auc], [i], s=15, facecolor=SPECTRA if nom else 'white', edgecolor=SPECTRA, lw=.9, zorder=3)
        if sig:
            a.text(r.within_cluster_embedding_auc + .012, i, '*', fontsize=7, va='center', ha='left', color=INK)
    a.axvline(.5, color=INK, lw=.6, zorder=0)
    a.set_ylim(K - .5, -.5); a.set_yticks([]); a.spines['left'].set_visible(False)
    a.set_xlim(.3, .8); a.set_xticks([.3, .4, .5, .6, .7, .8]); a.set_xticklabels(['0.3', '0.4', '0.5', '0.6', '0.7', '0.8'])
    a.set_xlabel('AUROC, cancer versus non-cancer control')
    hs = [Line2D([], [], marker='o', ls='', ms=3.2, mfc='white', mec=MUTED, mew=.8, label='Cluster abundance'),
          Line2D([], [], marker='o', ls='', ms=3.4, mfc=SPECTRA, mec=SPECTRA, label='Within-cluster embedding, P < 0.05'),
          Line2D([], [], marker='o', ls='', ms=3.4, mfc='white', mec=SPECTRA, mew=.9, label='Within-cluster embedding, P ≥ 0.05')]
    pg.fig.legend(handles=hs, loc='upper left', bbox_to_anchor=(.30 / pg.w, 1 - .02 / pg.h), ncol=3, fontsize=5.8, handletextpad=.3, columnspacing=1.2, frameon=False)

    # ---------- b: the same values against each other ----------
    pg.letter(4.40, .04, 'b')
    b = pg.ax(4.80, y0 + .05, 2.10, 2.10)   # different height from a by design; not an alignment group
    b.plot([.3, .8], [.3, .8], color=RULE, lw=.6, ls=(0, (2, 2)), zorder=0)
    b.axvline(.5, color=RULE, lw=.6, zorder=0); b.axhline(.5, color=RULE, lw=.6, zorder=0)
    for _, r in w.iterrows():
        c = int(r.cluster); nom = r.permutation_p < .05
        b.scatter([r.abundance_auc], [r.within_cluster_embedding_auc], s=22, color=PAL[c % len(PAL)], edgecolor=INK if nom else 'none', lw=.5, zorder=3)
    for _, r in w[w.permutation_q < .05].iterrows():
        left = int(r.number) == 4   # cluster 12 sits just to its right
        b.annotate(f'{int(r.number)}', (r.abundance_auc, r.within_cluster_embedding_auc), xytext=(-4 if left else 4, 2), textcoords='offset points',
                   fontsize=5.8, fontweight='bold', ha='right' if left else 'left')
    b.set_xlim(.3, .8); b.set_ylim(.3, .8); b.set_aspect('equal')
    for ax_ in (b,):
        ax_.set_xticks([.3, .5, .7]); ax_.set_yticks([.3, .5, .7]); ax_.set_xticklabels(['0.3', '0.5', '0.7']); ax_.set_yticklabels(['0.3', '0.5', '0.7'])
    b.set_xlabel('Cluster abundance AUROC'); b.set_ylabel('Within-cluster embedding AUROC')
    b.text(.31, .79, 'Embedding better\nthan abundance', ha='left', va='top', fontsize=5.6, color=MUTED, linespacing=1.1)
    sx.export(pg, 'supplementary', 'SupplementaryFig6_within_cluster_embeddings')
    w[['number', 'patients', 'controls', 'abundance_auc', 'within_cluster_embedding_auc', 'permutation_p', 'permutation_q']].sort_values('number').to_csv(
        sx.ROOT / 'supplementary/SupplementaryFig6_source.csv', index=False)


if __name__ == '__main__':
    build()
