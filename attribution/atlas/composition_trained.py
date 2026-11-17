"""What do the cell embeddings add to patient-level AUROC beyond cluster composition? (exploratory)

Composition-only comparator trained the same way as SPECTRA: fitted on the 1,793 primary training patients
(1,689 cancers, 104 reference controls; primary group, all three cell filters) and scored once on the same 757 held-out
test patients. SPECTRA test predictions are the saved locked-model probabilities (no inference here).

  stage assign (CPU): every model-input cell of each training patient's primary group -> one of the 23 atlas clusters,
                      exactly as for the test cells (locked normalizer, mass-90 PCA50, 15-NN on the Leiden atlas).
                      Checked by re-assigning a sample of test patients and matching the saved test assignment.
  stage compare:      composition = 23 centred log-ratio cluster shares (as in the within-test composition reference);
                      logistic regression C=0.1, class-balanced. Secondary: composition + patient-mean embedding (PCA50).
                      Training-cohort 3-fold CV (inherited donor folds) for reference; test AUROC with a stratified
                      patient bootstrap (B=4000, seed 20261111); paired differences use the same replicates.

The atlas clusters were defined on test-cohort cells (unsupervised; no labels enter the clustering); the composition
classifier itself never sees a test label.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'model/pipeline'))
R = REPO / 'runs/figure3_locked6_attribution_20261111'; W = R / 'mass90_leiden'
OUT = REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
LEDGER = REPO / 'cohort/reference_policy/population_patient_ledger.csv'
LOCK = REPO / 'runs/reference_control_full_refit_20261110/model_lock.json'
K, B, SEED = 23, 4000, 20261111


def assigner():
    from sklearn.neighbors import KNeighborsClassifier
    from common import read
    lock = read(LOCK)
    with np.load(lock['normalizer']) as z:
        mu, sd = z['mean'], z['scale']
    with np.load(W / 'pca_transform.npz') as z:
        pm, pc = z['mean'], z['components']
    knn = KNeighborsClassifier(15, n_jobs=32).fit(np.load(W / 'mass90_pca50.npy'), np.load(W / 'cluster_primary.npy'))
    return lambda raw: (lambda x: (knn.predict(x), x))(((np.asarray(raw, np.float32) - mu) / sd - pm) @ pc.T)


def assign():
    from runtime import Cache
    led = pd.read_csv(LEDGER, dtype={'donor_id': str})
    tr = led[led.split.eq('train') & led.primary_cancer_vs_reference_eligible.astype(str).eq('True')]
    assert len(tr) == 1793 and tr.label.eq(1).sum() == 1689
    f = assigner()
    # check: re-assign some test patients and compare with the saved test assignment
    cells = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id', 'slide_id', 'source_row'])
    saved = np.load(W / 'pool_cluster_all_cells.npy')
    from common import read
    receipt = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
    chk = sorted(cells.donor_id.unique())[::38]; agree = []
    for d in chk:
        g = cells[cells.donor_id.eq(d)]
        for sid, h in g.groupby('slide_id'):
            lab, _ = f(np.load(receipt[sid]['raw_cache'], mmap_mode='r')[h.source_row.to_numpy()]); agree.append(lab == saved[h.index.to_numpy()])
    agree = float(np.concatenate(agree).mean()); print('test re-assignment agreement', agree, flush=True)
    assert agree > .999
    cache = Cache(set(tr.donor_id))
    primary = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}
    assert len(primary) == 1793
    counts = np.zeros((len(tr), K), np.int64); emb = np.zeros((len(tr), 50))
    for i, d in enumerate(tr.donor_id):
        lab, x = f(cache.sample(primary[d])['features'].numpy())
        counts[i] = np.bincount(lab, minlength=K); emb[i] = x.mean(0)
        if (i + 1) % 200 == 0:
            print('assigned', i + 1, flush=True)
    np.savez(R / 'train_composition.npz', donor_id=tr.donor_id.to_numpy(), label=tr.label.to_numpy(), fold=tr.fold.astype(int).to_numpy(),
             counts=counts, mean_pca50=emb)
    json.dump(dict(train_patients=len(tr), cells=int(counts.sum()), test_reassignment_check_patients=len(chk), test_reassignment_agreement=agree),
              open(R / 'train_composition_receipt.json', 'w'), indent=1)


def clr(counts):
    p = counts / counts.sum(1, keepdims=True); c = np.log(p + 1e-3); return c - c.mean(1, keepdims=True)


def model(n_comp, n_all):
    lr = LogisticRegression(C=.1, class_weight='balanced', max_iter=4000)
    if n_comp == n_all:
        return lr
    return make_pipeline(ColumnTransformer([('comp', 'passthrough', list(range(n_comp))), ('emb', StandardScaler(), list(range(n_comp, n_all)))]), lr)


def compare():
    t = np.load(R / 'train_composition.npz', allow_pickle=True)
    ytr, fold = t['label'], t['fold']
    Xtr = {'composition': clr(t['counts']), 'composition_plus_mean_embedding': np.hstack([clr(t['counts']), t['mean_pca50']])}
    # test patients: saved assignment of every model-input cell
    cells = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id'])
    kcl = np.load(W / 'pool_cluster_all_cells.npy'); Z = np.load(R / 'all_cells_pca50.npy', mmap_mode='r')
    donors = pd.read_csv(R / 'donors.csv', dtype={'donor_id': str}, float_precision='round_trip').set_index('donor_id')
    ids = donors.index.to_numpy(); codes = pd.Categorical(cells.donor_id, categories=ids).codes
    assert (codes >= 0).all() and len(ids) == 757
    ct = np.zeros((len(ids), K), np.int64); np.add.at(ct, (codes, kcl), 1)
    # test mean embedding must be in the same (mass-90) PCA space as the training patients
    from common import read
    receipt = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
    full = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id', 'slide_id', 'source_row'])
    f = assigner(); emb = np.zeros((len(ids), 50)); n = np.zeros(len(ids))
    for sid, h in full.groupby('slide_id'):
        _, x = f(np.load(receipt[sid]['raw_cache'], mmap_mode='r')[h.source_row.to_numpy()])
        c = pd.Categorical(h.donor_id, categories=ids).codes; np.add.at(emb, c, x); np.add.at(n, c, 1)
    emb /= n[:, None]
    yte = donors.label.to_numpy(); spectra = donors.probability.to_numpy()
    Xte = {'composition': clr(ct), 'composition_plus_mean_embedding': np.hstack([clr(ct), emb])}
    res = dict(train_patients=len(ytr), train_controls=int((ytr == 0).sum()), test_patients=len(yte), test_controls=int((yte == 0).sum()))
    scores = {'SPECTRA': spectra}
    for name in Xtr:
        m = model(K, Xtr[name].shape[1])
        cv = [roc_auc_score(ytr[fold == k], m.fit(Xtr[name][fold != k], ytr[fold != k]).decision_function(Xtr[name][fold == k])) for k in range(3)]
        scores[name] = m.fit(Xtr[name], ytr).decision_function(Xte[name])
        res[f'{name}_train_cv_mean_fold_auc'] = float(np.mean(cv)); res[f'{name}_train_cv_folds'] = [float(v) for v in cv]
    rng = np.random.default_rng(SEED); pos, neg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    I = np.hstack([pos[rng.integers(0, len(pos), (B, len(pos)))], neg[rng.integers(0, len(neg), (B, len(neg)))]]); yb = yte[I[0]]
    boot = {k: np.array([roc_auc_score(yb, v[i]) for i in I]) for k, v in scores.items()}
    res['test_auc'] = {k: dict(auc=float(roc_auc_score(yte, v)), ci95=[float(q) for q in np.quantile(boot[k], [.025, .975])]) for k, v in scores.items()}
    for a, b in [('SPECTRA', 'composition'), ('SPECTRA', 'composition_plus_mean_embedding'), ('composition_plus_mean_embedding', 'composition')]:
        d = boot[a] - boot[b]
        res[f'diff_{a}_minus_{b}'] = dict(diff=float(roc_auc_score(yte, scores[a]) - roc_auc_score(yte, scores[b])),
                                          ci95=[float(q) for q in np.quantile(d, [.025, .975])], p_two_sided=float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
    res.update(bootstrap=dict(B=B, seed=SEED, kind='stratified patient, percentile'),
               note='Exploratory; composition model trained on the training cohort only and scored once on the test cohort; atlas clusters defined on test cells (unsupervised)')
    json.dump(res, open(OUT / 'composition_trained_vs_spectra.json', 'w'), indent=1); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    globals()[sys.argv[1]]()
