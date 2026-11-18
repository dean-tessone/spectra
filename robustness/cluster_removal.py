"""Revision steps 2.3/3: remove whole Leiden clusters from every test bag (locked-model inference only, 2026-11-17).

Same design as the registered Figure 4d removal (attribution/atlas/removal.py): all 757 primary
test patients, three sealed seeds averaged, baseline replay checked, matched random removal (same number of the patient's other
cells, five fixed-seed draws). Conditions: every model-input cell assigned (15-NN, pool_cluster_all_cells.npy) to cluster 10,
cluster 20 (CK+, CD45/CD31-) or cluster 21, using the paper's display numbers.
"""
from pathlib import Path
import json, sys
import numpy as np, pandas as pd, torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read  # noqa: E402
from runtime import Cache, model_for  # noqa: E402

RUN = REPO / 'runs/figure3_locked6_attribution_20261111'
L = REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
FULL = REPO / 'runs/reference_control_full_refit_20261110'
OUT = REPO / 'runs/revision_20261117/cluster_removal'; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [17, 29, 43]; DRAWS = 5
disp = pd.read_csv(L / 'cluster_display_numbers.csv').set_index('number').cluster
CONDS = {f'cluster{n}': int(disp[n]) for n in (10, 20, 21)}

torch.backends.cuda.matmul.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(.20)
lock = read(FULL / 'model_lock.json'); models = []
for s in SEEDS:
    m = model_for('transformer', 'd6_do04_lr2e4').cuda().eval()
    m.load_state_dict(torch.load(lock['models'][str(s)]['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True); models.append(m)
with np.load(lock['normalizer']) as z:
    mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
donors = pd.read_csv(RUN / 'donors.csv', dtype={'donor_id': str}, float_precision='round_trip')
cells = pd.read_parquet(RUN / 'all_cells.parquet', columns=['donor_id', 'slide_id', 'source_row', 'relevance'])
cells["cl"] = np.load(RUN / "mass90_leiden/pool_cluster_all_cells.npy")
for c, k in CONDS.items():
    cells[c] = cells.cl == k
cache = Cache(set(donors.donor_id))
primary = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}


@torch.no_grad()
def prob(x):
    out = []
    for m in models:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(float(m(x, mask)['logits'].float().sigmoid()[0]))
    return float(np.mean(out))


rows = []
for k, d in enumerate(donors.itertuples(), 1):
    b = cache.sample(primary[d.donor_id]); x = ((b['features'].cuda().float()[None] - mean) / scale).half().float()
    key = pd.DataFrame(dict(slide_id=np.repeat(b['slide_ids'], [len(i) for i in b['slide_row_indices']]),
                            source_row=np.concatenate([i.numpy() for i in b['slide_row_indices']])))
    c = key.merge(cells[cells.donor_id == d.donor_id], on=['slide_id', 'source_row'], how='left', validate='one_to_one')
    assert c.relevance.notna().all() and len(c) == x.shape[1]
    base = prob(x); assert abs(base - d.probability) <= 1e-6
    row = dict(donor_id=d.donor_id, label=int(d.label), cancer_types=d.cancer_types, n_cells=len(c), base=base); n = len(c)
    for cond in CONDS:
        tgt = np.flatnonzero(c[cond].to_numpy()); row[f'{cond}_n'] = len(tgt)
        if len(tgt) == 0:
            row[f'{cond}_removed'] = base; row[f'{cond}_random'] = base; continue
        if len(tgt) >= n:
            tgt = tgt[np.argsort(c.relevance.to_numpy()[tgt])][1:]; row[f'{cond}_emptied_flag'] = True
        keep = np.setdiff1d(np.arange(n), tgt); row[f'{cond}_removed'] = prob(x[:, torch.as_tensor(keep, device='cuda')])
        non = keep; m_ = min(len(tgt), len(non) - 1) if len(non) > 1 else 0; rs = []
        for draw in range(DRAWS):
            if m_ <= 0:
                rs.append(base); continue
            drop = np.random.default_rng([20261117 + draw, k]).choice(non, size=m_, replace=False)
            rs.append(prob(x[:, torch.as_tensor(np.setdiff1d(np.arange(n), drop), device='cuda')]))
        row[f'{cond}_random'] = float(np.mean(rs)); row[f'{cond}_random_matched_n'] = m_
    rows.append(row)
    if k % 100 == 0:
        print('removal', k, flush=True)
pd.DataFrame(rows).to_csv(OUT / 'cluster_removal_per_patient_private.csv', index=False)
json.dump(dict(patients=len(rows), conditions=CONDS, draws=DRAWS, baseline_replay_max_error_le=1e-6), open(OUT / 'receipt.json', 'w'), indent=1)
print('done')
