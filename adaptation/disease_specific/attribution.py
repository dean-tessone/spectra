"""How cancer-specific adaptation changes which cells drive the score.

For each task (lung vs lung-nodule work-up, breast vs BRCA carriers) and each locked seed (17/29/43):
  * source head: the locked model, unchanged;
  * probe head: the frozen-feature probe refitted exactly as in run.py (StandardScaler + L2 logistic,
    C=0.1, balanced, on training-task primary-group CLS embeddings), folded into the classifier layer
    (w' = w/scale, b' = b - sum(w*mean/scale)); held predictions must reproduce the saved probe;
  * final-block fine-tune: refitted with run.py's fixed recipe and seed, weights saved this time;
    agreement with the earlier (unsaved) fit is reported, not assumed.
AttnLRP (benchmark-selected methods.py, FP32) on every held-out TEST task patient for each head.
Per cell: seed-mean logit relevance per head. Inference/description only; no selection on test labels.
"""
from pathlib import Path
import importlib.util, json, sys, copy
import numpy as np, pandas as pd, torch

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location('adapt_run', HERE / 'run.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
spec = importlib.util.spec_from_file_location('lrp', REPO / 'attribution/methods/attnlrp.py'); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
OUT = REPO / 'runs/specific_adaptation_20261111/attribution'; OUT.mkdir(parents=True, exist_ok=True)


def probe_head(model, cache, fit, mean, scale, held_saved):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    fp, fz = R.predict(model, cache, sorted(fit.donor_id), mean, scale)
    sc = StandardScaler().fit(fz); clf = LogisticRegression(C=.1, class_weight='balanced', max_iter=5000).fit(sc.transform(fz), fp.label)
    w = clf.coef_[0] / sc.scale_; b = clf.intercept_[0] - float((clf.coef_[0] * sc.mean_ / sc.scale_).sum())
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.classifier.weight.copy_(torch.tensor(w[None], dtype=m.classifier.weight.dtype)); m.classifier.bias.copy_(torch.tensor([b], dtype=m.classifier.bias.dtype))
    hp, _ = R.predict(m, cache, sorted(held_saved.donor_id), mean, scale)
    err = float(np.abs(hp.set_index('donor_id').probability - held_saved.set_index('donor_id').probability).max())
    return m, err


def refit_final_block(task, seed):
    """Re-run run.py's fixed recipe for the test arm and keep the weights."""
    ck = REPO / f'runs/specific_adaptation_20261111/{task}/test/seed_{seed}/adapt_refit.pt'
    fit, held, src, norm = R.context(task, 'test', seed)
    if ck.exists():
        m = R.make('source', src); m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)); return m
    import random, torch.nn.functional as F
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    with np.load(norm) as z:
        mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
    cache = R.Cache(set(fit.donor_id) | set(held.donor_id)); model = R.make('adapt', src)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=.05); sch = torch.optim.lr_scheduler.LambdaLR(opt, R.lr)
    for batches in R.schedule(cache, fit, seed):
        R.lora.training_mode(model, 'adapt'); opt.zero_grad(set_to_none=True)
        for i, b in enumerate(R.loader(cache, batches=batches)):
            out = R.forward(model, b, mean, scale)
            (F.binary_cross_entropy_with_logits(out['logits'].float(), b['label'].cuda(), reduction='none') * b['sample_weight'].cuda()).sum().backward()
            if (i + 1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    torch.save(model.state_dict(), ck)
    return model


def attribute(model, cache, donors, mean, scale, tag):
    prim = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}
    rows = []
    for d in donors:
        b = cache.sample(prim[d]); x = ((b['features'].cuda().float()[None] - mean) / scale).half().float()
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device='cuda')
        v, info = M.gradient_scores(model, x, mask, relevance=True)
        rows.append(pd.DataFrame(dict(donor_id=d, slide_id=np.repeat(b['slide_ids'], [len(i) for i in b['slide_row_indices']]),
                                      source_row=np.concatenate([i.numpy() for i in b['slide_row_indices']]), **{tag: v})))
    return pd.concat(rows, ignore_index=True)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.set_per_process_memory_fraction(.25)
    receipt = {}
    for task in ['lung', 'breast']:
        per_seed = []
        for seed in [17, 29, 43]:
            fit, held, src, norm = R.context(task, 'test', seed)
            with np.load(norm) as z:
                mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
            cache = R.Cache(set(fit.donor_id) | set(held.donor_id))
            base = pd.read_csv(REPO / f'runs/specific_adaptation_20261111/{task}/test/seed_{seed}/baseline_predictions_private.csv', dtype={'donor_id': str})
            src_m = R.make('source', src).eval()
            pr_m, perr = probe_head(src_m, cache, fit, mean, scale, base[base.arm == 'probe'])
            ad_m = refit_final_block(task, seed).eval()
            adp, _ = R.predict(ad_m, cache, sorted(held.donor_id), mean, scale)
            old = pd.read_csv(REPO / f'runs/specific_adaptation_20261111/{task}/test/seed_{seed}/adapt/held_predictions_private.csv', dtype={'donor_id': str})
            from sklearn.metrics import roc_auc_score
            receipt[f'{task}_{seed}'] = dict(probe_replay_max_abs_error=perr, adapt_refit_auroc=float(roc_auc_score(adp.label, adp.probability)),
                                              adapt_original_auroc=float(roc_auc_score(old.label, old.probability)),
                                              adapt_refit_vs_original_spearman=float(adp.set_index('donor_id').probability.corr(old.set_index('donor_id').probability, method='spearman')))
            donors = sorted(held.donor_id); parts = []
            for tag, m in [('source', src_m), ('probe', pr_m), ('adapt', ad_m)]:
                parts.append(attribute(m.eval(), cache, donors, mean, scale, tag).set_index(['donor_id', 'slide_id', 'source_row']))
            t = pd.concat(parts, axis=1).reset_index(); t['seed'] = seed; per_seed.append(t)
            print(task, seed, receipt[f'{task}_{seed}'], flush=True)
        a = pd.concat(per_seed).groupby(['donor_id', 'slide_id', 'source_row'])[['source', 'probe', 'adapt']].mean().reset_index()
        lab = R.cohort(task).set_index('donor_id').label; a['label'] = a.donor_id.map(lab)
        a.to_parquet(OUT / f'{task}_cells_private.parquet')
    json.dump(receipt, open(OUT / 'receipt.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
