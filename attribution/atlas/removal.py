"""Locked-model cell-removal test. Inference only."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read, write, sha, utc  # noqa: E402
from runtime import Cache, model_for  # noqa: E402

RUN = REPO / 'runs/figure3_locked6_attribution_20261111'
OUT = REPO / 'analyses/figure3_locked6_attribution_20261111'
FULL = REPO / 'runs/reference_control_full_refit_20261110'
TESTA = REPO / 'analyses/reference_control_test_20261110_v1a'
SEEDS = [17, 29, 43]; DRAWS = 5
CONDITIONS = ['top10', 'top50', 'mass90', 'CK_gate', 'CTC_like', 'state8']


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.20)
    lock = read(FULL / 'model_lock.json'); models = []
    for s in SEEDS:
        m = model_for('transformer', 'd6_do04_lr2e4').cuda().eval()
        m.load_state_dict(torch.load(lock['models'][str(s)]['checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True)
        models.append(m)
    with np.load(lock['normalizer']) as z:
        mean, scale = [torch.tensor(z[k], device='cuda') for k in ['mean', 'scale']]
    donors = pd.read_csv(RUN / 'donors.csv', dtype={'donor_id': str}, float_precision='round_trip')
    cells = pd.read_parquet(RUN / 'all_cells.parquet', columns=['donor_id', 'slide_id', 'source_row', 'relevance', 'positive_rank', 'top50', 'mass90', 'CK_gate', 'bluev2_class'])
    pool = pd.read_parquet(RUN / 'pool_cluster_assignment.parquet')
    assert (pool.donor_id.to_numpy() == cells.donor_id.to_numpy()).all() and (pool.source_row.to_numpy() == cells.source_row.to_numpy()).all()
    c2m = pd.read_csv(OUT / 'cluster_to_meta.csv').set_index('cluster').meta
    cells['state8'] = pool.pool_cluster.map(c2m).to_numpy() == 7
    cells['top10'] = cells.positive_rank.between(1, 10)
    cells['CTC_like'] = cells.bluev2_class.isin([1, 2, 3])
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
        b = cache.sample(primary[d.donor_id])
        x = ((b['features'].cuda().float()[None] - mean) / scale).half().float()
        key = pd.DataFrame(dict(slide_id=np.repeat(b['slide_ids'], [len(i) for i in b['slide_row_indices']]),
                                source_row=np.concatenate([i.numpy() for i in b['slide_row_indices']])))
        c = cells[cells.donor_id == d.donor_id]
        c = key.merge(c, on=['slide_id', 'source_row'], how='left', validate='one_to_one')
        assert c.relevance.notna().all() and len(c) == x.shape[1]
        base = prob(x); assert abs(base - d.probability) <= 1e-6, (d.donor_id, base, d.probability)
        row = dict(donor_id=d.donor_id, label=int(d.label), cancer_types=d.cancer_types, n_cells=len(c), base=base)
        n = len(c)
        for cond in CONDITIONS:
            tgt = np.flatnonzero(c[cond].to_numpy()); row[f'{cond}_n'] = len(tgt)
            if len(tgt) == 0:
                row[f'{cond}_removed'] = base; row[f'{cond}_random'] = base; continue
            flagged = len(tgt) >= n
            if flagged:  # keep the lowest-relevance target
                tgt = tgt[np.argsort(c.relevance.to_numpy()[tgt])][1:]
            keep = np.setdiff1d(np.arange(n), tgt)
            row[f'{cond}_removed'] = prob(x[:, torch.as_tensor(keep, device='cuda')]); row[f'{cond}_emptied_flag'] = flagged
            non = np.setdiff1d(np.arange(n), tgt); m_ = min(len(tgt), len(non) - 1) if len(non) > 1 else 0
            rs = []
            for draw in range(DRAWS):
                if m_ <= 0:
                    rs.append(base); continue
                drop = np.random.default_rng([20261111 + draw, k]).choice(non, size=m_, replace=False)
                keep_r = np.setdiff1d(np.arange(n), drop); rs.append(prob(x[:, torch.as_tensor(keep_r, device='cuda')]))
            row[f'{cond}_random'] = float(np.mean(rs)); row[f'{cond}_random_matched_n'] = m_
        rows.append(row)
        if k % 50 == 0:
            print('removal', k, flush=True)
    r = pd.DataFrame(rows); r.to_csv(OUT / 'removal_per_patient_private.csv', index=False)
    summarize()


def summarize():
    r = pd.read_csv(OUT / 'removal_per_patient_private.csv')
    ops = pd.read_csv(TESTA / 'operating_points.csv')
    ops = ops[(ops.endpoint == 'primary') & (ops.cancer_type == 'Overall') & (ops.comparison == 'Cancer vs primary reference')].set_index('target_specificity').threshold
    def summ(col, name):
        p = r[col]; y = r.label
        return dict(condition=name, auroc=roc_auc_score(y, p),
                    sens90=float((p[y == 1] >= ops[.9]).mean()), spec90=float((p[y == 0] < ops[.9]).mean()),
                    sens95=float((p[y == 1] >= ops[.95]).mean()), spec95=float((p[y == 0] < ops[.95]).mean()))
    out = [summ('base', 'none')]
    for cond in CONDITIONS:
        out.append(summ(f'{cond}_removed', cond)); out.append(summ(f'{cond}_random', f'{cond}_matched_random'))
    s = pd.DataFrame(out)
    tgt = {cond: dict(cancers_with_any=float((r.loc[r.label == 1, f'{cond}_n'] > 0).mean()), controls_with_any=float((r.loc[r.label == 0, f'{cond}_n'] > 0).mean()),
                      median_removed_cases=float(r.loc[r.label == 1, f'{cond}_n'].median()), emptied=int(r.get(f'{cond}_emptied_flag', pd.Series(False)).fillna(False).sum()))
           for cond in CONDITIONS}
    s.to_csv(OUT / 'removal_summary.csv', index=False)
    write(OUT / 'removal_receipt.json', dict(completed_utc=utc(), patients=len(r), conditions=CONDITIONS, random_draws=DRAWS, targets=tgt,
                                             thresholds={str(k): float(v) for k, v in ops.items()}, baseline_replay_max_error_le=1e-6,
                                             outputs_sha256={str(p): sha(p) for p in [OUT / 'removal_summary.csv', OUT / 'removal_per_patient_private.csv']},
                                             new_fitting=False))
    print(s.round(4).to_string()); print(json.dumps(tgt, indent=1))


if __name__ == '__main__':
    summarize() if sys.argv[1:] == ['summarize'] else main()
