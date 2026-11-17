"""Breast cancer by stage: does SPECTRA's performance in late-stage breast cancer come from CK+ / CTC-like events?
Post hoc, descriptive, test cohort, 44 primary reference controls; uses saved predictions, attributions and the saved
locked-model removal scores (no new model calls)."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
R = REPO / 'runs/figure3_locked6_attribution_20261111'
A = REPO / 'analyses/figure3_locked6_attribution_20261111'
OUT = REPO / 'analyses/population_analyses'
CKPOS = [5, 6, 8, 12, 13, 15]  # internal Leiden ids of CK+ clusters (display 21, 19, 20, 18, 17, 16)


def boot_auc(y, s, B=4000, seed=17):
    y = np.asarray(y); s = np.asarray(s); i1, i0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0); rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        a, b = rng.choice(i1, len(i1)), rng.choice(i0, len(i0)); v.append(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[s[a], s[b]]))
    return np.percentile(v, [2.5, 97.5])


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7); return np.log(p / (1 - p))


def main():
    led = pd.read_csv(REPO / 'analyses/clinical_cohort_annotations_20261110/patient_clinical_ledger.csv', dtype=str).set_index('donor_id')
    don = pd.read_csv(R / 'donors.csv', dtype={'donor_id': str}).set_index('donor_id')
    cnt = pd.read_csv(REPO / 'analyses/bluev2_ctc_comparison_20261110_v1b/private_primary_patient_counts.csv', dtype={'donor_id': str})
    cnt = cnt[cnt['pool'] == 'model_input'].set_index('donor_id')
    rem = pd.read_csv(A / 'removal_per_patient_private.csv', dtype={'donor_id': str}).set_index('donor_id')
    c = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id', 'relevance', 'mass90', 'CK_gate', 'bluev2_class'])
    c['k'] = np.load(R / 'mass90_leiden/pool_cluster_all_cells.npy'); c['ckpos'] = c.k.isin(CKPOS); c['ctc'] = c.bluev2_class.isin([1, 2, 3])
    c['pos'] = c.relevance.clip(lower=0)
    g = c.groupby('donor_id')
    att = pd.DataFrame({f'{m}_share_of_positive_evidence': g.apply(lambda d, m=m: d.pos[d[m]].sum() / d.pos.sum()) for m in ['ckpos', 'CK_gate', 'ctc']})
    att = att.join(pd.DataFrame({f'{m}_fraction_of_cells': g[m].mean() for m in ['ckpos', 'CK_gate', 'ctc']}))
    d = don.join(att).join(cnt[['CTC_like_per_slide', 'CK_gate_per_slide']])
    d['stage'] = led.reindex(d.index).breast_early_late; d['cohort_stage'] = led.reindex(d.index).cohort_breast_early_late
    d['study'] = led.reindex(d.index).study; d['logit'] = logit(d.probability)
    for cond in ['CK_gate', 'CTC_like', 'top50']:
        d[f'drop_logit_{cond}'] = logit(rem.reindex(d.index).base) - logit(rem.reindex(d.index)[f'{cond}_removed'])
        d[f'score_without_{cond}'] = rem.reindex(d.index)[f'{cond}_removed']
    ctrl = d[d.label == 0]; br = d[(d.label == 1) & (d.cancer_types == 'Breast')]
    rows = []
    for st, s in [('All breast', br)] + [(k, br[br.stage == k]) for k in ['Early', 'Late', 'Unknown']]:
        if len(s) < 5:
            continue
        y = np.r_[np.ones(len(s)), np.zeros(len(ctrl))]
        r = dict(stratum=st, cases=len(s), controls=len(ctrl), studies='; '.join(f'{k} {v}' for k, v in s.study.value_counts().items()))
        for name, col in [('SPECTRA', 'probability'), ('SPECTRA without CK-gate cells', 'score_without_CK_gate'), ('SPECTRA without CTC-like cells', 'score_without_CTC_like'),
                          ('CTC-like count', 'CTC_like_per_slide'), ('CK-gate count', 'CK_gate_per_slide'),
                          ('Evidence share on CK+ clusters', 'ckpos_share_of_positive_evidence'), ('Evidence share on CTC-like cells', 'ctc_share_of_positive_evidence')]:
            sc = np.r_[s[col], ctrl[col]]; r[f'AUROC {name}'] = roc_auc_score(y, sc)
            if name == 'SPECTRA':
                lo, hi = boot_auc(y, sc); r['SPECTRA 95% CI'] = f'{lo:.2f} to {hi:.2f}'
        for col in ['ckpos_share_of_positive_evidence', 'ctc_share_of_positive_evidence', 'ckpos_fraction_of_cells', 'ctc_fraction_of_cells', 'drop_logit_CK_gate', 'drop_logit_CTC_like']:
            r[f'median {col}'] = s[col].median()
        rows.append(r)
    r = dict(stratum='Reference controls', cases=0, controls=len(ctrl))
    for col in ['ckpos_share_of_positive_evidence', 'ctc_share_of_positive_evidence', 'ckpos_fraction_of_cells', 'ctc_fraction_of_cells', 'drop_logit_CK_gate', 'drop_logit_CTC_like']:
        r[f'median {col}'] = ctrl[col].median()
    rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'breast_stage_attribution.csv', index=False)
    d.loc[br.index.append(ctrl.index)].to_csv(OUT / 'breast_stage_per_patient_private.csv')
    print(t.T.to_string())


if __name__ == '__main__':
    main()
