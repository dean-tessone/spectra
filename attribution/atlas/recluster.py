"""atlas over the positive-relevance-mass-90 cells of every patient,
Leiden clustering of ALL atlas cells (no noise label), as in the original Figure 3 (resolution 1).

stage graph  (project venv): PCA50 refit on mass90 cells, 30-NN cosine UMAP fuzzy graph (seed 17),
                              2-D UMAP for display (30 NN, min_dist 0.1, cosine, seed 17).
stage leiden (any venv with leidenalg 0.11.0 / igraph 1.0.0): RBConfiguration, resolution 1,
                              seeds 17 (primary), 29, 43 (stability, ARI).
stage assign (project venv):  every retained model-input cell -> nearest Leiden cluster (15-NN in PCA50).
"""
from pathlib import Path
import json, sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
R = REPO / 'runs/figure3_locked6_attribution_20261111'
W = R / 'mass90_leiden'; W.mkdir(exist_ok=True)


def graph():
    import pandas as pd
    from scipy import sparse
    from sklearn.decomposition import PCA
    from sklearn.utils import check_random_state
    from umap.umap_ import fuzzy_simplicial_set
    from umap import UMAP
    sys.path.insert(0, str(REPO / 'model/pipeline'))
    from common import read
    cells = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id', 'slide_id', 'source_row', 'mass90'])
    idx = np.flatnonzero(cells.mass90.to_numpy())
    receipt = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
    lock = read(REPO / 'runs/reference_control_full_refit_20261110/model_lock.json')
    with np.load(lock['normalizer']) as z:
        mu, sd = z['mean'], z['scale']
    X = np.empty((len(idx), 512), np.float32); sub = cells.iloc[idx]
    pos = {v: i for i, v in enumerate(idx)}
    for sid, g in sub.groupby('slide_id'):
        raw = np.load(receipt[sid]['raw_cache'], mmap_mode='r'); X[[pos[v] for v in g.index]] = raw[g.source_row.to_numpy()]
    X = (X - mu) / sd
    pca = PCA(50, svd_solver='randomized', random_state=17).fit(X); Z = pca.transform(X).astype(np.float32); del X
    np.save(W / 'mass90_pca50.npy', Z); np.save(W / 'mass90_index.npy', idx)
    np.savez(W / 'pca_transform.npz', mean=pca.mean_, components=pca.components_, evr=pca.explained_variance_ratio_)
    G, _, _ = fuzzy_simplicial_set(Z, 30, check_random_state(17), 'cosine', verbose=False)
    G = sparse.triu(G.tocoo(), k=1).tocoo(); sparse.save_npz(W / 'fuzzy_graph_upper.npz', G)
    xy = UMAP(n_neighbors=30, min_dist=.1, metric='cosine', random_state=17, n_jobs=1, low_memory=True).fit_transform(Z)
    np.save(W / 'umap2d.npy', xy.astype(np.float32))
    print('graph', len(idx), 'cells', G.nnz, 'edges', flush=True)


def leiden():
    import leidenalg, igraph
    from scipy import sparse
    G = sparse.load_npz(W / 'fuzzy_graph_upper.npz').tocoo(); n = G.shape[0]
    g = igraph.Graph(n=n, edges=np.column_stack([G.row, G.col]).tolist(), directed=False)
    out = {}
    for seed in [17, 29, 43]:
        p = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights=G.data.tolist(), resolution_parameter=1., n_iterations=-1, seed=seed)
        lab = np.asarray(p.membership); np.save(W / f'leiden_r1_seed{seed}.npy', lab); out[seed] = dict(clusters=int(lab.max() + 1), quality=float(p.quality()))
        print(seed, out[seed], flush=True)
    json.dump(dict(leidenalg=leidenalg.version, igraph=igraph.__version__, resolution=1, runs=out), open(W / 'leiden_receipt.json', 'w'), indent=1)


def assign():
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score
    from sklearn.neighbors import KNeighborsClassifier
    sys.path.insert(0, str(REPO / 'model/pipeline'))
    from common import read
    lab = {s: np.load(W / f'leiden_r1_seed{s}.npy') for s in [17, 29, 43] if (W / f'leiden_r1_seed{s}.npy').exists()}
    # relabel primary by size
    vals, counts = np.unique(lab[17], return_counts=True); order = vals[np.argsort(-counts, kind='stable')]; m = {v: i for i, v in enumerate(order)}
    primary = np.array([m[v] for v in lab[17]]); np.save(W / 'cluster_primary.npy', primary)
    ari = {s: float(adjusted_rand_score(lab[17], lab[s])) for s in lab if s != 17}
    idx = np.load(W / 'mass90_index.npy'); Z = np.load(W / 'mass90_pca50.npy')
    cells = pd.read_parquet(R / 'all_cells.parquet', columns=['slide_id', 'source_row'])
    receipt = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
    lock = read(REPO / 'runs/reference_control_full_refit_20261110/model_lock.json')
    with np.load(lock['normalizer']) as z:
        mu, sd = z['mean'], z['scale']
    with np.load(W / 'pca_transform.npz') as z:
        pm, pc = z['mean'], z['components']
    knn = KNeighborsClassifier(15).fit(Z, primary); pool = np.empty(len(cells), int)
    for sid, g in cells.groupby('slide_id'):
        raw = np.load(receipt[sid]['raw_cache'], mmap_mode='r')
        x = ((np.asarray(raw[g.source_row.to_numpy()], np.float32) - mu) / sd - pm) @ pc.T
        pool[g.index.to_numpy()] = knn.predict(x)
    np.save(W / 'pool_cluster_all_cells.npy', pool)
    # sanity: atlas cells keep their own label in the pool assignment
    agree = float((pool[idx] == primary).mean())
    json.dump(dict(clusters=int(primary.max() + 1), ari_vs_seed17=ari, atlas_cells=int(len(idx)), atlas_self_assignment_agreement=agree),
              open(W / 'assign_receipt.json', 'w'), indent=1)
    print(open(W / 'assign_receipt.json').read())


if __name__ == '__main__':
    globals()[sys.argv[1]]()
