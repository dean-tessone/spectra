"""Cancer-specific adaptation of SPECTRA to reserved diagnostic-control populations."""
from pathlib import Path
import argparse, importlib.util, json, math, os, random, sys, time, traceback
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read, write, sha, utc  # noqa: E402
from runtime import Cache, model_for, loader, Queue  # noqa: E402

RUN = REPO / 'runs/specific_adaptation_20261111'
CVSRC = REPO / 'runs/reference_control_refit_20261110'
FULL = REPO / 'runs/reference_control_full_refit_20261110'
LEDGER = REPO / 'cohort/reference_policy/population_patient_ledger.csv'
spec = importlib.util.spec_from_file_location('qv_lora', REPO / 'model/architectures/lora.py')
lora = importlib.util.module_from_spec(spec); spec.loader.exec_module(lora)
TASKS = {'lung': dict(cancer='Lung', controls=['SU', 'JN2', 'JN2-R']), 'breast': dict(cancer='Breast', controls=['INTC09'])}
ARMS = ['adapt', 'lora']


def cohort(task):
    l = pd.read_csv(LEDGER, dtype={'donor_id': str}); t = TASKS[task]
    d = l[((l.label == 1) & (l.cancer_types == t['cancer'])) | ((l.label == 0) & l.study.isin(t['controls']))].copy()
    assert not d.donor_id.duplicated().any()
    return d


def jobs():
    out = []
    for task in TASKS:
        for fold in [0, 1, 2]:
            for seed in [17, 29]:
                for arm in ARMS:
                    out.append((task, f'cv{fold}', seed, arm))
        for seed in [17, 29, 43]:
            for arm in ARMS:
                out.append((task, 'test', seed, arm))
    return out


def context(task, part, seed):
    d = cohort(task)
    if part == 'test':
        fit, held = d[d.split == 'train'], d[d.split == 'test']
        ck = read(FULL / 'model_lock.json')['models'][str(seed)]['checkpoint']; norm = read(FULL / 'model_lock.json')['normalizer']
    else:
        f = int(part[2:]); tr = d[d.split == 'train']
        fit, held = tr[tr.fold != f], tr[tr.fold == f]
        ck = CVSRC / f'reference_cv/transformer/d6_do04_lr2e4/fold_{f}/seed_{seed}/best.pt'; norm = CVSRC / f'data/contexts/{f}/normalization.npz'
        prov = read(CVSRC / f'data/contexts/{f}/provenance.json')
        assert not set(held.donor_id) & set(prov['fit_donors']), 'held donor in source fit'
    assert not set(fit.donor_id) & set(held.donor_id)
    return fit, held, Path(ck), Path(norm)


def lr(step):
    if step < 32:
        return (step + 1) / 32
    return .05 + .95 * .5 * (1 + math.cos(math.pi * min((step - 32) / (320 - 32), 1)))


def make(arm, ck):
    m = model_for('transformer', 'd6_do04_lr2e4')
    m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'], strict=True)
    if arm == 'adapt':
        for p in m.parameters():
            p.requires_grad_(False)
        for mod in (m.blocks[-1], m.norm, m.classifier):
            for p in mod.parameters():
                p.requires_grad_(True)
    elif arm == 'lora':
        lora.install(m, rank=4, alpha=8)
    return m.cuda()


def schedule(cache, fit, seed):
    meta = cache.meta[cache.meta.donor_id.isin(fit.donor_id)]
    cases = sorted(fit.donor_id[fit.label == 1]); ctrls = sorted(fit.donor_id[fit.label == 0])
    pos = Queue(cases, np.random.default_rng([seed, 1001])); neg = Queue(ctrls, np.random.default_rng([seed, 1002]))
    groups = {p: Queue(g.index, np.random.default_rng([seed, 2000 + i])) for i, (p, g) in enumerate(meta.groupby('donor_id', sort=True))}
    rng = np.random.default_rng([seed, 3000]); epochs = []
    for _ in range(5):
        batches = []
        for _ in range(64):
            ps = [pos.draw(1)[0] for _ in range(32)]; ns = [neg.draw(1)[0] for _ in range(32)]
            for i in range(0, 32, 4):
                ids = ps[i:i + 4] + ns[i:i + 4]; rng.shuffle(ids)
                batches.append([(groups[p].draw(1)[0], 1 / 64) for p in ids])
        epochs.append(batches)
    return epochs


def forward(model, batch, mean, scale):
    mask = batch['attention_mask'].cuda(non_blocking=True)
    x = ((batch['features'].cuda(non_blocking=True).float() - mean) / scale).half().float().masked_fill(~mask.unsqueeze(-1), 0.)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(x, mask)


@torch.no_grad()
def predict(model, cache, donors, mean, scale):
    model.eval(); prim = {e['donor_id']: g for g, e in cache.entries.items() if e['primary_evaluation']}
    ids = [prim[d] for d in donors]; rows = []; cls = []
    for b in loader(cache, ids=ids):
        out = forward(model, b, mean, scale); gid = b['group_id'][0]
        rows.append(dict(donor_id=cache.entries[gid]['donor_id'], label=cache.entries[gid]['label'], logit=float(out['logits'][0]),
                         probability=float(out['logits'].float().sigmoid()[0])))
        cls.append(out['cls_embedding'][0].float().cpu().numpy())
    return pd.DataFrame(rows), np.stack(cls)


def run(task, part, seed, arm):
    dest = RUN / task / part / f'seed_{seed}' / arm; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    fit, held, ck, norm = context(task, part, seed)
    with np.load(norm) as z:
        mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
    cache = Cache(set(fit.donor_id) | set(held.donor_id))
    model = make(arm, ck); params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=.05); sch = torch.optim.lr_scheduler.LambdaLR(opt, lr)
    hist = []
    for ep, batches in enumerate(schedule(cache, fit, seed), 1):
        t0 = time.monotonic(); lora.training_mode(model, arm); opt.zero_grad(set_to_none=True); tot = 0.; ups = 0
        for i, b in enumerate(loader(cache, batches=batches)):
            out = forward(model, b, mean, scale)
            loss = (F.binary_cross_entropy_with_logits(out['logits'].float(), b['label'].cuda(), reduction='none') * b['sample_weight'].cuda()).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward(); tot += float(loss)
            if (i + 1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); ups += 1
        assert ups == 64
        hist.append(dict(epoch=ep, loss=tot / 64, seconds=time.monotonic() - t0)); print(task, part, seed, arm, ep, round(tot / 64, 4), flush=True)
    pred, _ = predict(model, cache, sorted(held.donor_id), mean, scale); pred['arm'] = arm; pred['seed'] = seed
    pred.to_csv(dest / 'held_predictions_private.csv', index=False)
    write(dest / 'complete.json', dict(task=task, part=part, seed=seed, arm=arm, history=hist, fit_cases=int(fit.label.sum()), fit_controls=int((fit.label == 0).sum()),
                                       held_cases=int(held.label.sum()), held_controls=int((held.label == 0).sum()), source_checkpoint_sha256=sha(ck),
                                       trainable_parameters=sum(p.numel() for p in params), completed_utc=utc()))


def baselines(task, part, seed):
    """Source score and frozen-CLS probe for one source model (no training of the transformer)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    dest = RUN / task / part / f'seed_{seed}'; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'baselines_complete.json').exists():
        return
    fit, held, ck, norm = context(task, part, seed)
    with np.load(norm) as z:
        mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
    cache = Cache(set(fit.donor_id) | set(held.donor_id)); model = make('source', ck)
    hp, hz = predict(model, cache, sorted(held.donor_id), mean, scale)
    fp, fz = predict(model, cache, sorted(fit.donor_id), mean, scale)
    sc = StandardScaler().fit(fz); clf = LogisticRegression(C=.1, class_weight='balanced', max_iter=5000).fit(sc.transform(fz), fp.label)
    probe = hp.copy(); probe['probability'] = clf.predict_proba(sc.transform(hz))[:, 1]; probe['logit'] = clf.decision_function(sc.transform(hz))
    hp['arm'] = 'source'; probe['arm'] = 'probe'
    for d in (hp, probe):
        d['seed'] = seed
    pd.concat([hp, probe]).to_csv(dest / 'baseline_predictions_private.csv', index=False)
    write(dest / 'baselines_complete.json', dict(completed_utc=utc(), held=len(hp), fit=len(fp)))


def summarize():
    rows = []; allp = []
    for task in TASKS:
        for part in ['cv0', 'cv1', 'cv2', 'test']:
            seeds = [17, 29, 43] if part == 'test' else [17, 29]
            ps = []
            for s in seeds:
                p = RUN / task / part / f'seed_{s}'
                ps.append(pd.read_csv(p / 'baseline_predictions_private.csv', dtype={'donor_id': str}))
                for arm in ARMS:
                    ps.append(pd.read_csv(p / arm / 'held_predictions_private.csv', dtype={'donor_id': str}))
            d = pd.concat(ps).groupby(['donor_id', 'label', 'arm'], as_index=False).probability.mean(); d['task'] = task; d['part'] = part; allp.append(d)
            for arm, g in d.groupby('arm'):
                rows.append(dict(task=task, part=part, arm=arm, cases=int(g.label.sum()), controls=int((g.label == 0).sum()), auroc=roc_auc_score(g.label, g.probability)))
    r = pd.DataFrame(rows); out = REPO / 'analyses/specific_adaptation_20261111'; out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / 'auroc_by_part.csv', index=False)
    cv = r[r.part.str.startswith('cv')].groupby(['task', 'arm']).auroc.agg(['mean', 'min', 'max']).reset_index()
    cv.to_csv(out / 'cv_mean_fold_auroc.csv', index=False)
    pd.concat(allp).to_csv(out / 'predictions_private.csv', index=False)
    print(r.to_string()); print(cv.to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1); a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(.25); torch.backends.cuda.matmul.allow_tf32 = False
    if a.cmd == 'baselines':
        for task in TASKS:
            for part in ['cv0', 'cv1', 'cv2', 'test']:
                for s in ([17, 29, 43] if part == 'test' else [17, 29]):
                    baselines(task, part, s)
    elif a.cmd == 'fit':
        for i, j in enumerate(jobs()):
            if i % a.workers == a.worker:
                try:
                    run(*j)
                except Exception:
                    write(RUN / f'failed_{"_".join(map(str, j))}.json', dict(error=traceback.format_exc())); raise
    elif a.cmd == 'summarize':
        summarize()
