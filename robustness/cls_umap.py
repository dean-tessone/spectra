"""UMAP of the locked SPECTRA patient summary (CLS) embedding for all primary-analysis patients ().

Inference-free: reads the CLS embeddings saved by extract_cls.py (three sealed full-refit models, primary sample group, replay
verified against the sealed test report). Each seed's 256-d CLS is z-scored per dimension across patients and the three are
concatenated (768-d), reduced to 50 principal components, then embedded with UMAP (30 neighbours, min_dist 0.3, seed 0).
Population: the 2,550 primary-analysis patients (1,793 training, 757 test). Training patients were used to fit the model, so their
embeddings are in-sample.
"""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
import umap

REPO = Path(__file__).resolve().parents[2]
CLS = REPO / 'runs/revision_20261117/cls'
LEDGER = REPO / 'cohort/reference_policy/population_patient_ledger.csv'
OUT = REPO / 'analyses/revision_20261117/cls_umap'; OUT.mkdir(parents=True, exist_ok=True)

led = pd.read_csv(LEDGER, dtype={'donor_id': str})
led = led[led.primary_cancer_vs_reference_eligible.astype(bool)]
z = [np.load(CLS / f'seed_{s}.npz') for s in (17, 29, 43)]
ids = z[0]['donor_id']
assert all((q['donor_id'] == ids).all() for q in z)
keep = np.isin(ids, led.donor_id.to_numpy())
X = np.concatenate([(q['cls'] - q['cls'][keep].mean(0)) / (q['cls'][keep].std(0) + 1e-8) for q in z], axis=1)[keep]
prob = np.mean([q['probability'] for q in z], axis=0)[keep]
pcs = PCA(50, random_state=0).fit_transform(X)
emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(pcs)

d = pd.DataFrame(dict(donor_id=ids[keep], umap1=emb[:, 0], umap2=emb[:, 1], ensemble_probability=prob)) \
    .merge(led[['donor_id', 'label', 'cancer_types', 'split']], on='donor_id', how='left', validate='one_to_one')
assert len(d) == 2550 and d.label.notna().all()
d.to_csv(OUT / 'cls_umap_private.csv', index=False)
d.drop(columns='donor_id').to_csv(OUT / 'cls_umap_source.csv', index=False)
summ = dict(patients=len(d), cancers=int(d.label.sum()), controls=int((d.label == 0).sum()),
            training=int((d.split == 'train').sum()), test=int((d.split == 'test').sum()),
            representation='3 seeds x 256-d CLS, z-scored, concatenated; PCA 50; UMAP n_neighbors 30, min_dist 0.3, random_state 0')
json.dump(summ, open(OUT / 'summary.json', 'w'), indent=1); print(summ)
