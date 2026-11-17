"""In-sample scores of the locked SPECTRA ensemble on its own 1,793 training patients (inference only).

Answers a group-meeting question ("what is the performance on the training set?"). The honest generalisation estimate remains
training cross-validation (held-out folds) and the sealed test; in-sample AUROC only shows how closely the final fit matches its
own training data. Same primary group per donor, normaliser, BF16 autocast and three-seed probability mean as the test evaluation.
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read, utc  # noqa: E402
from runtime import Cache, model_for, score  # noqa: E402

FULL = REPO / 'runs/reference_control_full_refit_20261110'
OUT_RUN = REPO / 'runs/population_analyses'
OUT = REPO / 'analyses/population_analyses'
SEEDS = [17, 29, 43]


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.20)
    counts = pd.read_csv(REPO / 'analyses/bluev2_ctc_comparison_20261110_v1b/private_primary_patient_counts.csv', dtype={'donor_id': str})
    donors = counts[(counts.split == 'train') & counts.primary_cancer_vs_reference_eligible.astype(bool)].donor_id
    assert len(donors) == 1793
    cache = Cache(set(donors))
    lock = read(FULL / 'model_lock.json')
    with np.load(lock['normalizer']) as z:
        mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
    frames = []
    for s in SEEDS:
        m = model_for('transformer', 'd6_do04_lr2e4').cuda().eval()
        m.load_state_dict(torch.load(lock['models'][str(s)]['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True)
        d = score(m, 'transformer', cache, mean, scale, primary=True).assign(seed=s)
        frames.append(d); print('seed', s, len(d), utc(), flush=True)
        del m; torch.cuda.empty_cache()
    d = pd.concat(frames)
    d.to_parquet(OUT_RUN / 'training_set_scores_private.parquet')
    e = d.groupby(['donor_id', 'label', 'cancer_types'], as_index=False).probability.mean()
    res = dict(completed_utc=utc(), patients=int(len(e)), cancers=int(e.label.sum()), controls=int((e.label == 0).sum()),
               ensemble_auroc=float(roc_auc_score(e.label, e.probability)),
               seed_auroc={str(s): float(roc_auc_score(g.label, g.probability)) for s, g in d.groupby('seed')},
               note='in-sample (training patients the final models were fitted on); not a generalisation estimate')
    json.dump(res, open(OUT / 'training_set_auroc.json', 'w'), indent=1)
    print(res)


if __name__ == '__main__':
    main()
