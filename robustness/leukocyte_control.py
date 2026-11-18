"""Revision step 1.8: leukocyte negative control (locked-model inference only, 2026-11-17).

For each primary test patient, the locked SPECTRA ensemble scores a bag built only from candidate cells that pass technical QC
but are classified as leukocytes (wbcclass >= 0.9), i.e. the cells the third filter removes from the model input. Bags are
size-matched to the patient's real model-input bag (random subset, fixed seed); patients with fewer leukocyte candidates use all
of them. Pre-stated interpretation: if leukocyte-only bags separate cancers from controls about as well as real bags, the signal
is largely systemic (host response) or pre-analytical; if they separate poorly, the signal sits mainly in the non-leukocyte
candidate cells the model is given. Neither outcome alone distinguishes host biology from pre-analytical differences.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from runtime import Cache, model_for, loader  # noqa: E402
from common import local  # noqa: E402

FULL = REPO / 'runs/reference_control_full_refit_20261110'
LEDGER = REPO / 'cohort/reference_policy/population_patient_ledger.csv'
RECEIPT = REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json'
OUT = REPO / 'runs/revision_20261117/leukocyte_control'; OUT.mkdir(parents=True, exist_ok=True)
QC_T = 0.46509784932900855
torch.cuda.set_per_process_memory_fraction(0.2)

lock = json.load(open(FULL / 'model_lock.json'))
with np.load(lock['normalizer']) as z:
    mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
led = pd.read_csv(LEDGER, dtype={'donor_id': str})
led = led[(led.split == 'test') & (led.primary_cancer_vs_reference_eligible)] if 'primary_cancer_vs_reference_eligible' in led else led[led.split == 'test']
cache = Cache(set(led.donor_id))
prim = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}
rec = json.load(open(RECEIPT))['slides']
rng = np.random.default_rng(20261117)
audit = []
for d, gid in sorted(prim.items()):
    e = cache.entries[gid]; n_input = sum(e['lengths']); per = []
    for s in e['slide_ids']:
        r = rec[s]
        qc = np.load(local(r['qc_cache']), allow_pickle=False)
        f = pd.read_parquet(local(r['source']), columns=['wbcclass'])
        assert len(qc) == len(f)
        per.append(np.where((qc <= QC_T) & (f.wbcclass.to_numpy() >= .9))[0].astype(np.int64))
    tot = sum(len(p) for p in per)
    if tot > n_input:  # size-match: subsample across slides in proportion
        allidx = np.concatenate([np.c_[np.full(len(p), i), p] for i, p in enumerate(per)])
        keep = allidx[np.sort(rng.choice(len(allidx), n_input, replace=False))]
        per = [keep[keep[:, 0] == i, 1] for i in range(len(per))]
    cache.indices[gid] = per; e['lengths'] = [len(p) for p in per]
    audit.append(dict(donor_id=d, label=e['label'], model_input_cells=n_input, leukocyte_candidates=tot, leukocyte_bag=sum(e['lengths'])))
audit = pd.DataFrame(audit)
usable = audit[audit.leukocyte_bag > 0]
print('patients', len(audit), 'with leukocyte candidates', len(usable), flush=True)

probs = {}
for seed, m in lock['models'].items():
    model = model_for('transformer', 'd6_do04_lr2e4')
    model.load_state_dict(torch.load(m['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True)
    model = model.cuda().eval(); out_p = {}
    with torch.no_grad():
        for b in loader(cache, ids=[prim[d] for d in usable.donor_id]):
            mask = b['attention_mask'].cuda(non_blocking=True)
            x = ((b['features'].cuda(non_blocking=True).float() - mean) / scale).half().float().masked_fill(~mask.unsqueeze(-1), 0.)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = model(x, mask)
            out_p[cache.entries[b['group_id'][0]]['donor_id']] = float(o['logits'].float().sigmoid()[0])
    probs[seed] = out_p; print('seed', seed, 'done', flush=True)
usable = usable.copy()
usable['probability'] = [np.mean([probs[s][d] for s in probs]) for d in usable.donor_id]
usable.to_csv(OUT / 'leukocyte_bag_predictions_private.csv', index=False)
audit.to_csv(OUT / 'leukocyte_bag_audit_private.csv', index=False)
print('done')
