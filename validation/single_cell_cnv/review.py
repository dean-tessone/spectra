"""Private review sheet of first-pass CNV x attribution results. Panels: copy-number-altered share by
attribution group; by Fig. 3 cluster among high-attention (mass-90) cells; ablation (altered vs flat vs random
deletion, per patient); example embedded genomic profiles (HDF5 /profile_plots) of high-attention cells."""
import os
from pathlib import Path
import io
import numpy as np, pandas as pd, h5py
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
REPO = Path(__file__).resolve().parents[2]; OUT = REPO / 'analyses/cnv_attribution_20261112'
H5 = os.environ.get('SPECTRA_CNV_H5', 'raw/qc_approved_genomic_profiles.h5')
TEAL, GREY, ORANGE = '#0E6E7E', '#9AA5AD', '#E8892B'


def profile_png(names):
    with h5py.File(H5, 'r') as f:
        cn = f['profile_plots/cell_name'].asstr()[:]
        return {n: Image.open(io.BytesIO(f['profile_plots/png'][int(np.flatnonzero(cn == n)[0])].tobytes())) for n in names}


def main():
    T = pd.read_csv(OUT / 'test_sequenced_cells_private.csv', dtype={'slide_id': str})
    L = pd.read_csv(OUT / 'ablation_leave_one_out_private.csv'); G = pd.read_csv(OUT / 'ablation_group_private.csv')
    fig = plt.figure(figsize=(13, 14))
    a = fig.add_axes([.07, .74, .26, .22]); order = ['not a model input', 'other model input', 'other mass-90', 'top-50 attributed']
    g = T.groupby('group').altered.agg(['mean', 'size']).reindex(order)
    a.barh(range(4), 100 * g['mean'], color=[GREY, GREY, TEAL, TEAL])
    for i, (m, n) in enumerate(zip(g['mean'], g['size'])):
        a.text(100 * m + 1, i, f'{100 * m:.0f}% (n={n})', va='center', fontsize=9)
    a.set_yticks(range(4)); a.set_yticklabels(order); a.set_xlim(0, 100); a.set_xlabel('Sequenced cells copy-number altered (%)')
    a.set_title('a  Test cancer patients (45), by attribution', loc='left', fontsize=10)
    b = fig.add_axes([.45, .74, .22, .22]); h = T[T.mass90 == True].groupby('display_cluster').altered.agg(['mean', 'size'])
    h = h[h['size'] >= 5].sort_values('mean')
    b.barh(range(len(h)), 100 * h['mean'], color=TEAL)
    for i, (m, n) in enumerate(zip(h['mean'], h['size'])):
        b.text(100 * m + 1, i, f'{100 * m:.0f}% (n={n})', va='center', fontsize=9)
    b.set_yticks(range(len(h))); b.set_yticklabels([f'Cluster {int(c)}' for c in h.index]); b.set_xlim(0, 110)
    b.set_xlabel('High-attention cells altered (%)'); b.set_title('b  By Fig. 3 cluster (>=5 cells)', loc='left', fontsize=10)
    c = fig.add_axes([.77, .74, .2, .22]); w = G.pivot(index='donor_id', columns='removed', values='delta')
    rnd = G.groupby('donor_id').random_delta_mean.mean()
    for d in w.index:
        c.plot([0, 1], [w.loc[d, 'flat'], w.loc[d, 'altered']], color=GREY, lw=.8, marker='o', ms=3)
    c.scatter(np.full(len(rnd), 2), rnd, color=ORANGE, s=10)
    c.set_xticks([0, 1, 2]); c.set_xticklabels(['Delete\nflat', 'Delete\naltered', 'Delete\nrandom'])
    c.set_yscale('symlog', linthresh=.01); c.set_ylabel('Score drop (log-odds)')
    c.set_title(f'c  Ablation, {len(w)} patients', loc='left', fontsize=10)
    ex = []
    for clus, alt in [(20, True), (21, True), (10, False), (16, False)]:
        s = T[(T.display_cluster == clus) & (T.altered == alt) & (T.mass90 == True)].sort_values('positive_rank').head(2)
        ex += [(r.cell_name, f'Cluster {clus}, rank {int(r.positive_rank)}, altered genome {100 * r.altered_fraction:.0f}%') for r in s.itertuples()]
    imgs = profile_png([n for n, _ in ex])
    for k, (n, lab) in enumerate(ex):
        ax = fig.add_axes([.04 + (k % 2) * .48, .52 - (k // 2) * .165, .46, .15]); ax.imshow(imgs[n]); ax.axis('off')
        ax.set_title(lab, fontsize=9, loc='left')
    fig.text(.04, .69, 'd  Example profiles of high-attention cells (embedded review plots; log ratio, grey bins, black segments)', fontsize=10)
    fig.savefig(OUT / 'cnv_attribution_review_private.png', dpi=110); print('saved')


if __name__ == '__main__':
    main()
