"""Locked-model AttnLRP attribution for TRAINING patients with IMC data.
Identical method to Figure 3 (methods.gradient_scores relevance=True, FP32, probability relevance, equal mean over
seeds 17/29/43, policy.select_cells top-50 / mass-90), primary sample group only; cells assigned to Figure 3
Leiden clusters (15-NN in atlas PCA space, analyze_attribution.assign). Inference only.
Arguments: donor ids (primary sample group) or 'group:<cache group id>' for a non-primary tube of a training
control donor whose slides have IMC."""
from pathlib import Path
import importlib.util, sys
import numpy as np, pandas as pd, torch
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read, sha  # noqa: E402
from runtime import Cache, model_for  # noqa: E402
load = lambda n, p: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(n, p))
methods = load('m', REPO / 'attribution/methods/attnlrp.py')
policy = load('p', REPO / 'attribution/methods/policy.py')
AA = load('aa', REPO / 'adaptation/disease_specific/analyze_attribution.py')
FULL = REPO / 'runs/reference_control_full_refit_20261110'; SEEDS = [17, 29, 43]
OUT = REPO / 'runs/imc_linkage_20261111/train_attribution'; OUT.mkdir(parents=True, exist_ok=True)


def main(donors):
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.20)
    lock = read(FULL / 'model_lock.json'); assert sha(lock['normalizer']) == lock['normalizer_sha256']
    models = {}
    for s in SEEDS:
        assert sha(lock['models'][str(s)]['checkpoint']) == lock['models'][str(s)]['checkpoint_sha256']
        m = model_for('transformer', 'd6_do04_lr2e4').cuda().eval()
        m.load_state_dict(torch.load(lock['models'][str(s)]['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True)
        for p in m.parameters():
            p.requires_grad_(False)
        models[s] = m
    with np.load(lock['normalizer']) as z:
        mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
    groups = [d.split(':', 1)[1] for d in donors if d.startswith('group:')]; donors = [d for d in donors if not d.startswith('group:')]
    cache = Cache(set(donors) | {g.split('__')[1] for g in groups}); primary = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}
    jobs = [(d, primary[d], d) for d in donors] + [(cache.entries[g]['donor_id'], g, g) for g in groups]
    rec = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
    for d, grp, name in jobs:
        b = cache.sample(grp)
        x = ((b['features'].cuda().float()[None] - mean) / scale).half().float(); mask = torch.ones(x.shape[:2], dtype=torch.bool, device='cuda')
        t = pd.DataFrame(dict(slide_id=np.repeat(b['slide_ids'], [len(i) for i in b['slide_row_indices']]), source_row=np.concatenate([i.numpy() for i in b['slide_row_indices']])))
        probs = []
        for s, m in models.items():
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                probs.append(float(torch.sigmoid(m(x, mask)['logits'][0])))
            v, info = methods.gradient_scores(m, x, mask, relevance=True)
            t[f'logit_relevance_{s}'] = v; t[f'relevance_{s}'] = policy.probability_relevance(v, info['output'])
        t['relevance'] = t[[f'relevance_{s}' for s in SEEDS]].mean(axis=1); t['donor_id'] = d; t['probability'] = float(np.mean(probs))
        t, st = policy.select_cells(t)
        parts = []
        for sid, g in t.groupby('slide_id', sort=False):
            src = pd.read_parquet(rec[sid]['source'], columns=['frame_id', 'cell_id', 'x', 'y', 'area', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']).iloc[g.source_row.to_numpy()]
            parts.append(pd.concat([g.reset_index(drop=True), src.reset_index(drop=True)], axis=1))
        t = pd.concat(parts, ignore_index=True); t['cluster'] = AA.assign(t)
        t['group'] = grp; t.to_parquet(OUT / f'{name}.parquet', index=False); print(d, 'cells', len(t), 'p', round(t.probability.iloc[0], 3), 'mass90', int(t.mass90.sum()), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
