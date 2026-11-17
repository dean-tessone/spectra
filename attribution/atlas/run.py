"""Locked six-layer SPECTRA AttnLRP attribution on the primary test endpoint (inference only).

Stages: attribute (GPU) -> annotate (CPU) -> embed (CPU).
"""
from pathlib import Path
import argparse, importlib.util, json, os, sys, time, traceback, hashlib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import ROOT as MAIN, read, write, sha, utc  # noqa: E402
from runtime import Cache, model_for  # noqa: E402

RUN = REPO / 'runs/figure3_locked6_attribution_20261111'
FULL = REPO / 'runs/reference_control_full_refit_20261110'
TEST = REPO / 'runs/reference_control_test_20261110_v1a'
TESTA = REPO / 'analyses/reference_control_test_20261110_v1a'
BLUE = REPO / 'runs/bluev2_ctc_comparison_20261110_v1b/slides'
METHOD_PATH = REPO / 'attribution/methods/attnlrp.py'
POLICY_PATH = REPO / 'attribution/methods/policy.py'
SEEDS = [17, 29, 43]
CLASSES = ['Large Nuclei', 'imCTCs', 'CTCs', 'pcCTCs', 'CECs', 'Megakaryocytes', 'Fibroblasts', 'Lymph', 'Gran', 'Mono']


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


policy = load('fig3_policy', POLICY_PATH)


def status(phase, **kw):
    write(RUN / 'status.json', dict(phase=phase, pid=os.getpid(), updated_utc=utc(), **kw))


def donors_table():
    pred = pd.read_csv(TEST / 'private_primary_donor_predictions.csv', dtype={'donor_id': str}, float_precision='round_trip')
    ledger = pd.read_csv(REPO / 'cohort/reference_policy/population_patient_ledger.csv', dtype={'donor_id': str})
    pred = pred.merge(ledger[['donor_id', 'primary_cancer_vs_reference_eligible']], on='donor_id', validate='one_to_one')
    pred = pred[pred.primary_cancer_vs_reference_eligible].drop(columns='primary_cancer_vs_reference_eligible')
    assert len(pred) == 757 and pred.label.eq(1).sum() == 713
    ops = pd.read_csv(TESTA / 'operating_points.csv')
    thr = float(ops[(ops.endpoint == 'primary') & (ops.cancer_type == 'Overall') & (ops.target_specificity == .9) & (ops.comparison == 'Cancer vs primary reference')].threshold.iloc[0])
    pred['called_at_90spec'] = pred.probability.gt(thr)
    return pred.sort_values('donor_id').reset_index(drop=True), thr


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = read(FULL / 'model_lock.json')
    files = [Path(__file__), METHOD_PATH, POLICY_PATH, FULL / 'model_lock.json', Path(lock['normalizer']),
             TEST / 'complete.json', TEST / 'private_primary_donor_predictions.csv', TESTA / 'operating_points.csv',
             REPO / 'cohort/reference_policy/population_patient_ledger.csv'] + [TEST / f'private_seed_{s}_group_predictions.csv' for s in SEEDS]
    assert sha(lock['normalizer']) == lock['normalizer_sha256']
    for s in SEEDS:
        m = lock['models'][str(s)]; assert sha(m['checkpoint']) == m['checkpoint_sha256']; files.append(Path(m['checkpoint']))
    d, thr = donors_table()
    d.to_csv(RUN / 'donors.csv', index=False)
    reg = dict(created_utc=utc(), inputs_sha256={str(p): sha(p) for p in files}, donors=len(d), cases=int(d.label.sum()),
               controls=int((d.label == 0).sum()), threshold_90spec=thr, endpoint='primary group', inference_only=True,
               new_fitting=False, method='AttnLRP (benchmark-selected), FP32, sigmoid chain rule, equal seed mean')
    write(RUN / 'registration.json', reg); status('prepared'); return reg


def attribute():
    reg = read(RUN / 'registration.json')
    for p, h in reg['inputs_sha256'].items():
        assert sha(p) == h, p
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.20)
    methods = load('fig3_methods', METHOD_PATH)
    lock = read(FULL / 'model_lock.json'); models = {}; saved = {}
    for s in SEEDS:
        state = torch.load(lock['models'][str(s)]['checkpoint'], map_location='cpu', weights_only=False)
        m = model_for('transformer', 'd6_do04_lr2e4').cuda().eval(); m.load_state_dict(state['model'], strict=True)
        for p in m.parameters():
            p.requires_grad_(False)
        models[s] = m
        saved[s] = pd.read_csv(TEST / f'private_seed_{s}_group_predictions.csv', float_precision='round_trip').set_index('group_id')
    with np.load(lock['normalizer']) as z:
        mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
    donors = pd.read_csv(RUN / 'donors.csv', dtype={'donor_id': str}, float_precision='round_trip')
    cache = Cache(set(donors.donor_id))
    primary = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}
    assert len(primary) == 757
    out = RUN / 'donors'; out.mkdir(exist_ok=True); stats = []
    for k, row in enumerate(donors.itertuples(), 1):
        dest = out / f'{row.donor_id}.parquet'
        if dest.exists():
            continue
        gid = primary[row.donor_id]; b = cache.sample(gid)
        x = ((b['features'].cuda().float()[None] - mean) / scale).half().float()
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device='cuda')
        table = pd.DataFrame(dict(slide_id=np.repeat(b['slide_ids'], [len(i) for i in b['slide_row_indices']]),
                                  source_row=np.concatenate([i.numpy() for i in b['slide_row_indices']])))
        probs = []; diag = []
        for s, m in models.items():
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                actual = float(m(x, mask)['logits'][0])
            err = abs(actual - float(saved[s].loc[gid, 'logit']))
            if err > 1e-4:
                raise ValueError(f'BF16 replay mismatch {row.donor_id} seed{s}: {err}')
            vals, info = methods.gradient_scores(m, x, mask, relevance=True)
            table[f'logit_relevance_{s}'] = vals
            table[f'relevance_{s}'] = policy.probability_relevance(vals, info['output'])
            probs.append(float(saved[s].loc[gid, 'probability']))
            diag.append(dict(seed=s, replay_error=err, fp32_logit=info['output'], cell_sum=info['cell_sum']))
        assert abs(np.mean(probs) - row.probability) < 1e-9
        table['relevance'] = table[[f'relevance_{s}' for s in SEEDS]].mean(axis=1)
        table['logit_relevance'] = table[[f'logit_relevance_{s}' for s in SEEDS]].mean(axis=1)
        table['donor_id'] = row.donor_id
        table, st = policy.select_cells(table)
        st.update(donor_id=row.donor_id, label=int(row.label), cancer_type=row.cancer_types, probability=float(row.probability),
                  max_replay_error=max(q['replay_error'] for q in diag))
        table.to_parquet(dest, index=False); stats.append(st)
        status('attributing', completed=k, total=757)
        if k % 50 == 0:
            print('attributed', k, flush=True)
    status('attribution_complete')


def annotate():
    """Join per-cell BLUEV2 class and raw marker means (CPU)."""
    donors = pd.read_csv(RUN / 'donors.csv', dtype={'donor_id': str})
    receipt = read(MAIN / 'data/cache_receipt.json')['slides']
    frames = []
    for i, d in enumerate(donors.donor_id, 1):
        t = pd.read_parquet(RUN / 'donors' / f'{d}.parquet')
        parts = []
        for sid, g in t.groupby('slide_id', sort=False):
            rows = g.source_row.to_numpy()
            src = pd.read_parquet(receipt[sid]['source'], columns=['frame_id', 'cell_id', 'x', 'y', 'area', 'eccentricity', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean'])
            src = src.iloc[rows].reset_index(drop=True)
            with np.load(BLUE / sid / 'classes.npz') as z:
                cls = z['class_id'][rows]
            assert (cls >= 0).all(), sid
            g = g.reset_index(drop=True); g = pd.concat([g, src], axis=1); g['bluev2_class'] = cls
            parts.append(g)
        frames.append(pd.concat(parts, ignore_index=True))
        if i % 100 == 0:
            print('annotated', i, flush=True)
    cells = pd.concat(frames, ignore_index=True)
    meta = donors[['donor_id', 'label', 'cancer_types', 'study', 'probability', 'called_at_90spec']]
    cells = cells.merge(meta, on='donor_id', validate='many_to_one')
    cells['CK_gate'] = (cells.TRITC_mean > 40000) & (cells.CY5_mean < 10000)
    cells.to_parquet(RUN / 'all_cells.parquet', index=False)
    write(RUN / 'annotate_complete.json', dict(cells=len(cells), completed_utc=utc()))
    status('annotate_complete', cells=len(cells))


def embed():
    from sklearn.decomposition import PCA
    from sklearn.cluster import HDBSCAN
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import adjusted_rand_score
    from umap import UMAP
    cells = pd.read_parquet(RUN / 'all_cells.parquet')
    receipt = read(MAIN / 'data/cache_receipt.json')['slides']
    lock = read(FULL / 'model_lock.json')
    with np.load(lock['normalizer']) as z:
        mu, sd = z['mean'], z['scale']
    top = cells.index[cells.top50].to_numpy()
    # PCA fitted on top-50 cells only; then streamed transform of every cell (no full Z512 copy kept).
    X = np.empty((len(top), 512), np.float32)
    sub = cells.loc[top]
    for sid, g in sub.groupby('slide_id'):
        raw = np.load(receipt[sid]['raw_cache'], mmap_mode='r'); X[np.searchsorted(top, g.index.to_numpy())] = raw[g.source_row.to_numpy()]
    X = (X - mu) / sd
    pca = PCA(50, svd_solver='randomized', random_state=17).fit(X)
    Z = np.empty((len(cells), 50), np.float32)
    for sid, g in cells.groupby('slide_id'):
        raw = np.load(receipt[sid]['raw_cache'], mmap_mode='r')
        Z[g.index.to_numpy()] = pca.transform((np.asarray(raw[g.source_row.to_numpy()], np.float32) - mu) / sd)
    np.save(RUN / 'all_cells_pca50.npy', Z)
    np.savez(RUN / 'pca_transform.npz', mean=pca.mean_, components=pca.components_, evr=pca.explained_variance_ratio_)
    status('embedding')
    atlas = cells.loc[top].copy()
    labels = {}
    for seed in [17, 29, 43]:
        coords = UMAP(n_neighbors=30, min_dist=.1, metric='cosine', random_state=seed, n_jobs=1, low_memory=True).fit_transform(Z[top])
        lab = HDBSCAN(min_cluster_size=100, min_samples=15, cluster_selection_method='eom').fit(coords).labels_
        labels[seed] = policy.relabel_by_size(lab)
        if seed == 17:
            atlas['umap_1'], atlas['umap_2'] = coords[:, 0], coords[:, 1]; atlas['cluster'] = labels[17]
    stab = {s: float(adjusted_rand_score(labels[17], labels[s])) for s in [29, 43]}
    atlas.to_parquet(RUN / 'atlas_top50.parquet')
    keep = atlas.cluster.ge(0).to_numpy()
    knn = KNeighborsClassifier(15).fit(Z[top][keep], atlas.cluster.to_numpy()[keep])
    pool = np.empty(len(cells), int)
    for a in range(0, len(cells), 200000):
        pool[a:a + 200000] = knn.predict(Z[a:a + 200000])
    cells['pool_cluster'] = pool
    cells[['donor_id', 'slide_id', 'source_row', 'pool_cluster']].to_parquet(RUN / 'pool_cluster_assignment.parquet')
    write(RUN / 'embed_complete.json', dict(atlas_cells=int(len(atlas)), clusters=int(atlas.cluster.max() + 1),
                                            noise_fraction=float((atlas.cluster < 0).mean()), ari_vs_seed17=stab,
                                            pca_evr50=float(pca.explained_variance_ratio_.sum()), completed_utc=utc()))
    status('embed_complete')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare', 'attribute', 'annotate', 'embed', 'all']); a = p.parse_args()
    try:
        for st in (['prepare', 'attribute', 'annotate', 'embed'] if a.stage == 'all' else [a.stage]):
            globals()[st]()
    except Exception:
        status('failed', error=traceback.format_exc()); raise
