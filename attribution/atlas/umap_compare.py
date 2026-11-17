"""Display-only comparison of 2-D UMAP layouts for the Figure 3 atlas.

Inputs are the saved PCA50 of the 204,131 mass90 cells; Leiden clusters are NOT recomputed (they come from the 30-NN fuzzy graph and
are independent of the 2-D layout). Requires umap-learn 0.6.0.

    python umap_compare.py fit <name>     # one layout -> runs/.../mass90_leiden/umap_compare/<name>.npy
    python umap_compare.py score          # separation metrics + contact sheet
"""
from pathlib import Path
import json, sys, time
import numpy as np

W = Path(__file__).resolve().parents[2] / 'runs/figure3_locked6_attribution_20261111/mass90_leiden'
OUT = W / 'umap_compare'; OUT.mkdir(exist_ok=True)
CONFIGS = {
    'v06_n30_md0.1': dict(n_neighbors=30, min_dist=.1),     # current settings, new version (recursive init)
    'v06_n30_md0': dict(n_neighbors=30, min_dist=0.),
    'v06_n50_md0': dict(n_neighbors=50, min_dist=0.),
    'v06_n15_md0': dict(n_neighbors=15, min_dist=0.),
    # the four above run 0.6.0's legacy (0.5.x) pipeline by default; these use the new recursive init + Adam optimizer
    'new_n30_md0.1': dict(n_neighbors=30, min_dist=.1, compatibility_layout=False),
    'new_n30_md0': dict(n_neighbors=30, min_dist=0., compatibility_layout=False),
    'new_n50_md0': dict(n_neighbors=50, min_dist=0., compatibility_layout=False),
}


def fit(name):
    import umap
    Z = np.load(W / 'mass90_pca50.npy'); t = time.time()
    xy = umap.UMAP(metric='cosine', random_state=17, **CONFIGS[name]).fit_transform(Z)
    np.save(OUT / f'{name}.npy', xy.astype(np.float32))
    json.dump(dict(name=name, umap=umap.__version__, seconds=time.time() - t, **CONFIGS[name]), open(OUT / f'{name}.json', 'w'))


def score():
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    k = np.load(W / 'cluster_primary.npy'); rng = np.random.default_rng(0); s = rng.choice(len(k), 20000, replace=False)
    lays = {'current (0.5.12, n30, md 0.1)': np.load(W / 'umap2d.npy')} | {n: np.load(OUT / f'{n}.npy') for n in CONFIGS if (OUT / f'{n}.npy').exists()}
    res = {}
    for n, xy in lays.items():
        nn = NearestNeighbors(n_neighbors=16).fit(xy); _, ix = nn.kneighbors(xy[s])
        res[n] = dict(knn15_same_cluster=float((k[ix[:, 1:]] == k[s, None]).mean()), silhouette_2d=float(silhouette_score(xy[s], k[s])))
    json.dump(res, open(OUT / 'separation_metrics.json', 'w'), indent=1); print(json.dumps(res, indent=1))
    pal = plt.get_cmap('tab20').colors + plt.get_cmap('Dark2').colors
    fig, axs = plt.subplots(1, len(lays), figsize=(5 * len(lays), 5.3))
    o = rng.permutation(len(k))
    for a, (n, xy) in zip(np.atleast_1d(axs), lays.items()):
        a.scatter(xy[o, 0], xy[o, 1], s=.15, c=[pal[i % len(pal)] for i in k[o]], lw=0, rasterized=True)
        a.set_title(f"{n}\nkNN purity {res[n]['knn15_same_cluster']:.2f}, silhouette {res[n]['silhouette_2d']:.2f}", fontsize=10); a.axis('off')
    fig.savefig(OUT / 'umap_compare.png', dpi=110, bbox_inches='tight')


if __name__ == '__main__':
    fit(sys.argv[2]) if sys.argv[1] == 'fit' else score()
