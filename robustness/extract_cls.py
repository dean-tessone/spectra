"""Revision step 1.7: frozen CLS embeddings of the locked SPECTRA ensemble (inference only, 2026-11-17).

Runs the three sealed full-refit models (seeds 17/29/43, sealed normalizer) on the primary sample group of every patient in the
population ledger (training, test and reserved controls) and saves the CLS embedding and probability. No training, no threshold or
endpoint change. Replay check: test probabilities must match the sealed test report.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from runtime import Cache, model_for, loader  # noqa: E402

FULL = REPO / 'runs/reference_control_full_refit_20261110'
LEDGER = REPO / 'cohort/reference_policy/population_patient_ledger.csv'
OUT = REPO / 'runs/revision_20261117/cls'; OUT.mkdir(parents=True, exist_ok=True)
torch.cuda.set_per_process_memory_fraction(0.2)


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


lock = json.load(open(FULL / 'model_lock.json'))
assert sha(lock['normalizer']) == lock['normalizer_sha256']
with np.load(lock['normalizer']) as z:
    mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
led = pd.read_csv(LEDGER, dtype={'donor_id': str})
cache = Cache(set(led.donor_id))
prim = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}
donors = sorted(prim)
print('donors with a primary group:', len(donors), 'of', len(led), flush=True)

for seed, m in lock['models'].items():
    dest = OUT / f'seed_{seed}.npz'
    if dest.exists():
        continue
    assert sha(m['checkpoint']) == m['checkpoint_sha256']
    model = model_for('transformer', 'd6_do04_lr2e4')
    model.load_state_dict(torch.load(m['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True)
    model = model.cuda().eval(); ids, probs, cls = [], [], []
    with torch.no_grad():
        for b in loader(cache, ids=[prim[d] for d in donors]):
            mask = b['attention_mask'].cuda(non_blocking=True)
            x = ((b['features'].cuda(non_blocking=True).float() - mean) / scale).half().float().masked_fill(~mask.unsqueeze(-1), 0.)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(x, mask)
            ids.append(cache.entries[b['group_id'][0]]['donor_id']); probs.append(float(out['logits'].float().sigmoid()[0]))
            cls.append(out['cls_embedding'][0].float().cpu().numpy())
    np.savez(dest, donor_id=np.array(ids), probability=np.array(probs), cls=np.stack(cls))
    print('seed', seed, 'done', len(ids), flush=True)

# replay check against the sealed test report
pr = pd.read_csv(REPO / 'runs/reference_control_test_20261110_v1a/private_primary_donor_predictions.csv', dtype={'donor_id': str}) \
    if (REPO / 'runs/reference_control_test_20261110_v1a/private_primary_donor_predictions.csv').exists() else None
z = [np.load(OUT / f'seed_{s}.npz') for s in lock['models']]
ens = pd.DataFrame(dict(donor_id=z[0]['donor_id'], probability=np.mean([q['probability'] for q in z], axis=0)))
rep = {}
if pr is not None:
    mm = pr.merge(ens, on='donor_id', suffixes=('_sealed', '_replay'))
    rep = dict(matched=len(mm), max_abs_diff=float((mm.probability_sealed - mm.probability_replay).abs().max()))
json.dump(dict(donors=len(ens), replay=rep), open(OUT / 'receipt.json', 'w'), indent=1)
print(rep)
