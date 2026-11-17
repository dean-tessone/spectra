"""Figure 3 - which cells drive SPECTRA: atlas with real cells, an aligned atlas table, and composition alone."""
import json
import sys
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch
import sx
from sx import INK, SPECTRA, CONTROL

sys.path.insert(0, str(sx.REPO / 'imaging/cell_images'))
from cell_ext_display import rgb

L = sx.REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
C40 = sx.REPO / 'analyses/figure3_locked6_attribution_20261111/cache_40x'
T10 = sx.REPO / 'analyses/figure3_locked6_attribution_20261111/tenx/image_cache'
PAL = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
       '#2F4B7C', '#A05195', '#D45087', '#F95D6A', '#FF7C43', '#665191', '#8CD17D', '#B6992D', '#499894', '#86BCB6', '#B5BD61', '#79706E', '#17BECF']
FEATURED = [9, 8, 5, 15, 0, 2, 12, 7]   # 0-based clusters shown around the atlas


def cap(s):
    return s[0].upper() + s[1:]


def crops():
    ten = {}
    for f in T10.glob('*.npz'):
        z = np.load(f); ten.update(dict(zip(z['cell_keys'], z['images'])))
    g = pd.read_csv(L / 'gallery_selection.csv'); out = {}
    for r in g.itertuples():
        im = np.load(C40 / f'{r.cell_key}.npz')['image'] if r.source == '40x' else ten[r.cell_key]
        h = int(r.half); out[r.cluster] = (sx.boost(rgb(im[int(r.cy) - h:int(r.cy) + h + 1, int(r.cx) - h:int(r.cx) + h + 1])), r.source, h)
    return out


def build():
    sx.style()
    atlas = pd.read_parquet(L / 'atlas_mass90_private.parquet')
    # display layout: umap-learn 0.6.0 new pipeline (recursive init, Adam), 50 NN, min_dist 0, cosine, seed 17,
    # on the same PCA50; rows aligned with the atlas (checked). Clusters are unchanged (30-NN fuzzy graph + Leiden).
    xy = np.load(sx.ATTR / 'mass90_leiden/umap_compare/new_n50_md0.npy'); assert len(xy) == len(atlas)
    atlas['umap_1'], atlas['umap_2'] = xy[:, 0], xy[:, 1]
    tab = pd.read_csv(L / 'cluster_table.csv', index_col=0)
    names = pd.read_csv(L / 'cluster_names.csv', index_col=0).name
    comp = json.load(open(L / 'composition_reference.json'))
    col = {k: PAL[k % len(PAL)] for k in tab.index}
    NUM = sx.cluster_numbers()
    im = crops()
    pg = sx.Page(7.0, 6.66)   # <= 170 mm tall (Nature)

    # ---------- a: atlas with featured cells ----------
    pg.letter(.02, .04, 'a')
    a = pg.ax(1.45, .14, 4.05, 3.05)
    o = np.random.default_rng(0).permutation(len(atlas))
    a.scatter(atlas.umap_1.to_numpy()[o], atlas.umap_2.to_numpy()[o], s=.2, c=[col[k] for k in atlas.cluster.to_numpy()[o]], lw=0, rasterized=True)
    def peak(g):   # label at the cluster's densest spot (a median can fall between islands)
        h, xe, ye = np.histogram2d(g.umap_1, g.umap_2, bins=40); i, j = np.unravel_index(h.argmax(), h.shape)
        return (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    cen = {k: peak(g) for k, g in atlas.groupby('cluster')}
    for k, (x, y) in cen.items():
        a.text(x, y, str(NUM[k]), fontsize=5.6, fontweight='bold', ha='center', va='center', color='white', bbox=dict(boxstyle='circle,pad=.12', fc=col[k], ec='none'))
    a.set_xticks([]); a.set_yticks([]); a.set_aspect('equal', adjustable='datalim')
    for s in a.spines.values():
        s.set_visible(False)
    pg.fig.canvas.draw()
    gal = pd.read_csv(L / 'gallery_selection.csv'); feat = gal.cluster.tolist()
    feat = sorted(feat, key=lambda k: NUM[k]); left, right = feat[:12], feat[12:]
    tw, vg = .40, .118
    for side, ks in [('L', left), ('R', right)]:
        cols = [.40, .90] if side == 'L' else [5.60, 6.10]
        for i, k in enumerate(ks):
            r_, c_ = divmod(i, 2)
            x0 = cols[c_]
            y0 = .22 + r_ * (tw + vg)
            ax = pg.ax(x0, y0, tw, tw); img, src, h = im[k]
            ax.imshow(img, interpolation='lanczos'); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)   # no coloured frames: tiles are numbered
            um = .18 if src == '40x' else .59   # um per pixel: 40x 0.18, 10x 0.59
            ax.plot([.08 * 2 * h, .08 * 2 * h + 5 / um], [.9 * 2 * h] * 2, color='white', lw=.8, solid_capstyle='butt')
            ax.text(.07, .93, str(NUM[k]), transform=ax.transAxes, fontsize=5.6, fontweight='bold', color='white', ha='left', va='top')
            if src == '10x':
                ax.text(.95, .05, '10x', transform=ax.transAxes, fontsize=5, color='white', ha='right', va='bottom')

    # ---------- b: aligned atlas table ----------
    pg.letter(.02, 3.40, 'b')
    order = tab.sort_values('abundance_auc', ascending=False).index.tolist(); K = len(order)
    y0, h = 3.83, 2.46
    nm = pg.ax(.12, y0, 1.85, h); nm.set_xlim(0, 1); nm.set_ylim(K - .5, -.5); nm.axis('off')
    for i, k in enumerate(order):
        nm.scatter([.05], [i], s=18, color=col[k], lw=0)
        nm.text(.11, i, f'{NUM[k]}', fontsize=5.6, va='center', fontweight='bold')
        nm.text(.22, i, cap(names[k]), fontsize=5.6, va='center')
    mk = pg.ax(2.05, y0, 1.05, h, panel='b_markers')
    for j, (cn, c1) in enumerate([('CK', '#C0392B'), ('CD45_CD31', '#2E9A4A'), ('VIM', '#5F6B72'), ('DAPI', '#2B5BD7')]):
        cmap = LinearSegmentedColormap.from_list(cn, ['#F1F3F4', c1])
        for i, k in enumerate(order):
            v = tab.loc[k, cn]; mk.scatter([j], [i], s=6 + 60 * v / 65535, color=cmap(min(v / 40000, 1.)), lw=0)
    mk.set_xticks(range(4)); mk.set_xticklabels(['CK', 'CD45/\nCD31', 'VIM', 'DAPI'], fontsize=5.6, linespacing=1.25); mk.xaxis.tick_top()
    mk.set_yticks([]); mk.set_ylim(K - .5, -.5); mk.set_xlim(-.6, 3.6); mk.tick_params(length=0)
    for s in mk.spines.values():
        s.set_visible(False)
    ab = pg.ax(3.30, y0, 1.30, h, panel='b_abund')
    for i, k in enumerate(order):
        v = tab.loc[k, 'abundance_auc'] - .5
        ab.barh(i, v, color=SPECTRA if v > 0 else CONTROL, height=.72, lw=0)
        if tab.loc[k, 'abundance_q'] < .05:
            ab.text(v + (.008 if v > 0 else -.008), i, '*', ha='left' if v > 0 else 'right', va='center', fontsize=7)
    ab.axvline(0, color=INK, lw=.6); ab.set_ylim(K - .5, -.5); ab.set_yticks([]); ab.spines['left'].set_visible(False)
    ab.set_xlim(-.18, .22); ab.set_xticks([-.1, 0, .1, .2]); ab.set_xticklabels(['0.4', '0.5', '0.6', '0.7'])
    ab.set_xlabel('Abundance, AUROC of\nper-patient share', linespacing=1.2, fontsize=6)
    ab.set_title('Controls ← → Cancers', fontsize=5.8, pad=3)
    em = pg.ax(4.80, y0, 1.15, h, panel='b_emph')
    for i, k in enumerate(order):
        v = tab.loc[k, 'emphasis_log2_cancer']
        em.barh(i, v, color='#3C5A99' if v > 0 else '#B7BEC4', height=.72, lw=0)
    em.axvline(0, color=INK, lw=.6); em.set_ylim(K - .5, -.5); em.set_yticks([]); em.spines['left'].set_visible(False)
    em.set_xlim(-1.5, 1.1); em.set_xlabel('Model emphasis\n(log2)', linespacing=1.2, fontsize=6)
    em.set_title('Down ← → Up-weighted', fontsize=5.8, pad=3)

    # ---------- c: composition alone ----------
    pg.letter(6.05, 3.40, 'c')
    f = pg.ax(6.42, y0, .52, h * .6, panel='c')
    vals = [(comp['spectra_auc'], 'SPECTRA', SPECTRA), (comp['composition_auc_mean'], 'Cluster\ncomposition', '#5F6B72')]
    for i, (v, lab, c1) in enumerate(vals):
        f.bar(i, v - .5, bottom=.5, color=c1, width=.7)
        f.text(i, v + .008, f'{v:.2f}', ha='center', va='bottom', fontsize=6)
    f.set_xticks([0, 1]); f.set_xticklabels([v[1] for v in vals], fontsize=5.6, rotation=90, linespacing=1.15)
    f.set_ylim(.5, .9); f.set_yticks([.5, .6, .7, .8, .9]); f.set_ylabel('AUROC'); f.tick_params(axis='x', length=0)
    sx.export(pg, 'figure3', 'Figure3', row_groups=[['b_markers', 'b_abund', 'b_emph']],
              exemptions=[dict(panels=['b_markers', 'b_abund', 'b_emph'], checks=['panel-width', 'horizontal-gutter'], reason='aligned table columns of different content widths')])
    tab.assign(name=names.reindex(tab.index)).to_csv(sx.ROOT / 'figure3/cluster_table_source.csv')


if __name__ == '__main__':
    build()
