"""AttnLRP model-randomization sanity check (Adebayo et al. 2018, cascading randomization).

Inference only, no fitting. For a fixed subset of primary test patients (all 44 reference controls + 76 cancers,
random seed 20261111), recompute AttnLRP cell relevance (same method as Figure 3: methods.gradient_scores,
relevance=True, FP32, equal mean over seeds 17/29/43) for the trained locked models and for cascading
randomizations from the top: classifier; + final norm; + block 6; ... + block 1; + input projection and CLS token.
Randomization re-initializes the randomized modules with PyTorch default initializers (fixed seed per model seed
and stage), as in the original test. A first version that redrew each tensor around its own trained mean kept
the readout direction and is preserved as analyses/attribution_sanity_20261111_meanmatched (superseded). Per patient, agreement with the trained-model
map: Spearman correlation of signed logit relevance and of |relevance|, and the overlap of the top-50 positive
cells (|A and B| / 50; random expectation 50 / cells). A faithful attribution method should lose agreement as
learned weights are destroyed. The trained-stage recomputation is checked against the saved Figure 3 values."""
from pathlib import Path
import importlib.util, json, sys
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read, sha, utc  # noqa: E402
from runtime import Cache, model_for  # noqa: E402
FULL = REPO / 'runs/reference_control_full_refit_20261110'
F3 = REPO / 'runs/figure3_locked6_attribution_20261111'
OUT = REPO / 'analyses/attribution_sanity_20261111'; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [17, 29, 43]
spec = importlib.util.spec_from_file_location('m', REPO / 'attribution/methods/attnlrp.py'); methods = importlib.util.module_from_spec(spec); spec.loader.exec_module(methods)
STAGES = [('trained', []), ('classifier', ['classifier']), ('+ final norm', ['norm'])] + \
         [(f'+ block {b + 1}', [f'blocks.{b}']) for b in range(5, -1, -1)] + [('+ input projection, CLS', ['projection', 'cls'])]


def randomize(model, prefixes, gen):
    """Re-initialize the named modules with PyTorch default initializers (zero-mean Kaiming-uniform Linear,
    LayerNorm weight 1 / bias 0); the CLS parameter is redrawn as N(0, its trained std)."""
    seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen))
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, mod in model.named_modules():
            if n and any(n == q or n.startswith(q + '.') for q in prefixes) and hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()
        if 'cls' in prefixes:
            model.cls.copy_(torch.randn(model.cls.shape, device=model.cls.device) * model.cls.std())


def main():
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.20)
    lock = read(FULL / 'model_lock.json')
    assert sha(lock['normalizer']) == lock['normalizer_sha256']
    with np.load(lock['normalizer']) as z:
        mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
    states = {}
    for s in SEEDS:
        assert sha(lock['models'][str(s)]['checkpoint']) == lock['models'][str(s)]['checkpoint_sha256']
        states[s] = torch.load(lock['models'][str(s)]['checkpoint'], map_location='cpu', weights_only=False)['model']
    donors = pd.read_csv(F3 / 'donors.csv', dtype={'donor_id': str})
    rng = np.random.default_rng(20261111)
    sub = pd.concat([donors[donors.label == 0], donors[donors.label == 1].sample(76, random_state=rng.integers(1 << 31))]).reset_index(drop=True)
    cache = Cache(set(sub.donor_id)); primary = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}
    rows = []; check = []
    for k, d in enumerate(sub.itertuples(), 1):
        b = cache.sample(primary[d.donor_id])
        x = ((b['features'].cuda().float()[None] - mean) / scale).half().float(); mask = torch.ones(x.shape[:2], dtype=torch.bool, device='cuda')
        rel = {}
        for si, (stage, _) in enumerate(STAGES):
            vals = []
            for s in SEEDS:
                m = model_for('transformer', 'd6_do04_lr2e4').cuda().eval(); m.load_state_dict(states[s], strict=True)
                for i in range(1, si + 1):
                    g = torch.Generator().manual_seed(1000 * s + i)
                    randomize(m, STAGES[i][1], g)
                for p in m.parameters():
                    p.requires_grad_(False)
                v, _ = methods.gradient_scores(m, x, mask, relevance=True); vals.append(np.asarray(v, float))
            rel[stage] = np.mean(vals, axis=0)
        keys = pd.DataFrame(dict(slide_id=np.repeat(b['slide_ids'], [len(i) for i in b['slide_row_indices']]),
                                 source_row=np.concatenate([i.numpy() for i in b['slide_row_indices']])))
        sv = pd.read_parquet(F3 / 'donors' / f'{d.donor_id}.parquet', columns=['slide_id', 'source_row', 'logit_relevance'])
        saved = keys.merge(sv, on=['slide_id', 'source_row'], how='left', validate='one_to_one').logit_relevance.to_numpy()
        check.append(float(np.max(np.abs(saved - rel['trained']))))
        o = rel['trained']; top = set(np.argsort(-o)[:50]); n = len(o)
        for stage, _ in STAGES:
            r = rel[stage]
            rows.append(dict(donor_id=d.donor_id, label=int(d.label), cells=n, stage=stage,
                             spearman_signed=spearmanr(o, r).statistic, spearman_abs=spearmanr(np.abs(o), np.abs(r)).statistic,
                             top50_overlap=len(top & set(np.argsort(-r)[:50])) / 50, top50_random_expectation=min(50, n) / n))
        if k % 10 == 0:
            print(k, 'patients', 'max replay diff', max(check), flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / 'per_patient_private.csv', index=False)
    order = [s for s, _ in STAGES]
    summ = df.groupby('stage', sort=False)[['spearman_signed', 'spearman_abs', 'top50_overlap', 'top50_random_expectation']].median().reindex(order)
    summ.to_csv(OUT / 'summary_median.csv')
    json.dump(dict(completed_utc=utc(), patients=len(sub), controls=int((sub.label == 0).sum()), cancers=int(sub.label.sum()),
                   max_abs_diff_trained_vs_saved_figure3=max(check), stages=order, method=__doc__), open(OUT / 'receipt.json', 'w'), indent=1)
    print(summ.round(3).to_string())


if __name__ == '__main__':
    main()
