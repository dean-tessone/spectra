"""Supplementary Fig. 2 - gallery of attributed cells for every atlas cluster: many small tiles per cluster, 40x re-images first,
then 10x crops (rule-based selection, attribution/atlas/gallery_many.py)."""
import textwrap
import numpy as np
import pandas as pd
import sx
from sx import INK
from fig3 import PAL, cap, L

N = 16   # the smallest number of hand-picked tiles in any cluster, so every row is full and columns line up


def build():
    sx.style()
    k = pd.read_csv(L / 'gallery_many_private.csv'); z = np.load(L / 'gallery_many_private.npz'); field = float(z['field_um'])
    names = pd.read_csv(L / 'cluster_names.csv', index_col=0).name; NUM = sx.cluster_numbers()
    order = sorted(k.cluster.unique(), key=lambda c: NUM[c])
    TW, PX, PY, X0, Y0 = .27, .29, .283, 1.80, .12   # page height <= 170 mm (Nature); one fixed column pitch
    pg = sx.Page(7.0, Y0 + len(order) * PY + .04)
    for i, c in enumerate(order):
        y = Y0 + i * PY; g = k[k.cluster == c].sort_values('tile')
        pg.fig.text(.24 / pg.w, 1 - (y + TW / 2) / pg.h, str(NUM[c]), fontsize=5.8, fontweight='bold', ha='center', va='center', color='white',
                    bbox=dict(boxstyle='circle,pad=.14', fc=PAL[c % len(PAL)], ec='none'))
        pg.text(.40, y + TW / 2, '\n'.join(textwrap.wrap(cap(names[c]), 24)), fontsize=5.4, va='center', linespacing=1.15)
        for j, r in enumerate(g.head(N).itertuples()):
            x = X0 + j * PX
            ax = pg.ax(x, y, TW, TW); im = z[str(r.tile)]; ax.imshow(im, interpolation='lanczos'); ax.set_xticks([]); ax.set_yticks([])
            sx.scalebar(ax, im.shape[0], field / im.shape[0], um=10, lw=.6)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(INK); sp.set_linewidth(.4)
    sx.export(pg, 'supplementary', 'SupplementaryFig2_cluster_gallery')
    k.groupby('cluster').head(N).groupby(['cluster', 'source']).size().unstack(fill_value=0).rename(index=NUM).sort_index().to_csv(sx.ROOT / 'supplementary/SupplementaryFig2_tiles_per_cluster_source.csv')


if __name__ == '__main__':
    build()
