"""Revision step 1.7: how much study / collection-site information the locked CLS embedding carries (2026-11-17).

Features: CLS embeddings of the three sealed models concatenated (3 x 256). Linear probes (standardised, L2 logistic C=0.1,
class-balanced) are fitted on training patients and evaluated on test patients. Chance is estimated by refitting with shuffled
training labels (200 permutations). Probes:
  A. reference-control source (NBD, EPIC, INTC17-C, PANC-Ctrl) among reference controls;
  B. collection site among breast cancers (sites with >=10 training patients);
  C. collection site among prostate cancers (MD Anderson vs other sites);
  D. reference: cancer vs control (the model's own task) with the same probe, for scale.
"""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent)); from sites import patient_sites  # noqa: E402
OUT = REPO / 'analyses/revision_20261117/origin_probe'; OUT.mkdir(parents=True, exist_ok=True)
Z = [np.load(REPO / f'runs/revision_20261117/cls/seed_{s}.npz') for s in (17, 29, 43)]
assert all((z['donor_id'] == Z[0]['donor_id']).all() for z in Z)
X = np.concatenate([z['cls'] for z in Z], axis=1); ids = Z[0]['donor_id']
led = patient_sites().set_index('donor_id').loc[ids].reset_index()
rng = np.random.default_rng(20261117); NPERM = 200


def probe(mask, target, name, nperm=NPERM):
    d = led[mask].copy(); x = X[mask.to_numpy()]; yv = target[mask].to_numpy()
    tr, te = (d.split == "train").to_numpy().copy(), (d.split == "test").to_numpy().copy()
    classes = sorted(set(yv[tr]) & set(yv[te])); keep = np.isin(yv, classes); tr &= keep; te &= keep
    sc = StandardScaler().fit(x[tr])
    def fit_eval(ytr):
        clf = LogisticRegression(C=.1, class_weight='balanced', max_iter=5000).fit(sc.transform(x[tr]), ytr)
        return balanced_accuracy_score(yv[te], clf.predict(sc.transform(x[te])))
    obs = fit_eval(yv[tr]); null = np.array([fit_eval(rng.permutation(yv[tr])) for _ in range(nperm)]) if nperm else np.array([np.nan])
    return dict(probe=name, classes=classes, train_counts=pd.Series(yv[tr]).value_counts().to_dict(), test_counts=pd.Series(yv[te]).value_counts().to_dict(),
                balanced_accuracy=obs, chance=1 / len(classes), permutation_mean=float(null.mean()), permutation_95=float(np.percentile(null, 95)),
                p_perm=float((1 + (null >= obs).sum()) / (1 + nperm)) if nperm else None)


res = []
ref = led.primary_reference_control.fillna(False).astype(bool)
res.append(probe(ref, led.study, 'A. Reference-control source among reference controls'))
br = (led.label == 1) & (led.cancer_types == 'Breast') & (led.site != 'SWOG S1416 sites')  # S1416 (triple-negative) excluded; the other breast cohorts are HR+
bsite = led.site.where(led.site.map(led[br & (led.split == 'train')].site.value_counts()).fillna(0) >= 10)
res.append(probe(br & bsite.notna(), bsite, 'B. Collection site among HR+ breast cancers (S1416 excluded; sites with >=10 training patients)'))
# C. prostate omitted: cohorts of mixed, unrecorded disease state
prim = led.primary_cancer_vs_reference_eligible.fillna(False).astype(bool)
res.append(probe(prim, led.label.astype(str), 'D. Cancer vs reference control (scale reference)', nperm=0))
json.dump(res, open(OUT / 'summary.json', 'w'), indent=1, default=str)
for r in res:
    print(r['probe'], '| bal acc', round(r['balanced_accuracy'], 3), '| chance', round(r['chance'], 3), '| perm mean', round(r['permutation_mean'], 3), '95%', round(r['permutation_95'], 3), '| p', r['p_perm'])
    print('   train', r['train_counts'], 'test', r['test_counts'])
