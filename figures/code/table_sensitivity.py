"""Supplementary Table 1 - performance by cancer type on the held-out test set (44 non-cancer controls).

AUROC with stratified patient bootstrap 95% CIs (analyses/bootstrap_ci_20261111; not computed for types with fewer than
10 patients, which are descriptive). Sensitivity at the common 90% and 95% specificity cut-offs, calls as in the sealed
test report (probability >= cut-off), with Wilson 95% intervals."""
import numpy as np
import pandas as pd
import sx

CI = sx.REPO / 'analyses/bootstrap_ci_20261111/confidence_intervals.csv'


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def build():
    p = sx.primary_predictions(); c = p[p.label.eq(1)]
    ci = pd.read_csv(CI).set_index('metric')
    rows = []
    order = c.cancer_types.value_counts().index.tolist()
    for name, g in [('All cancers', c)] + [(t, c[c.cancer_types.eq(t)]) for t in order]:
        key = 'Primary pooled AUROC (713 cancers vs 44 controls)' if name == 'All cancers' else f'AUROC {name}'
        a = ci.loc[key]
        r = dict(cancer_type=name if name == 'All cancers' else sx.short(name), n=len(g), auroc=a.estimate, auroc_ci_low=a.ci_low, auroc_ci_high=a.ci_high)
        for t, col in [(90, 'called90'), (95, 'called95')]:
            k = int(g[col].sum()); lo, hi = wilson(k, len(g))
            r[f'sens{t}_k'] = k; r[f'sens{t}'] = k / len(g); r[f'sens{t}_ci_low'] = lo; r[f'sens{t}_ci_high'] = hi
        rows.append(r)
    t = pd.DataFrame(rows)
    assert t.loc[0, 'sens90_k'] == 403 and t.loc[0, 'sens95_k'] == 283          # sealed test report
    out = sx.ROOT / 'final_figures'
    t.to_csv(out / 'SupplementaryTable1_performance_by_cancer_type.csv', index=False)
    pct = lambda r, s: f"{100 * r[f'sens{s}']:.1f}% ({r[f'sens{s}_k']}/{r.n}; {100 * r[f'sens{s}_ci_low']:.1f}–{100 * r[f'sens{s}_ci_high']:.1f})"
    auc = lambda r: f'{r.auroc:.2f} ({r.auroc_ci_low:.2f}–{r.auroc_ci_high:.2f})' if pd.notna(r.auroc_ci_low) else f'{r.auroc:.2f}'
    md = ['**Supplementary Table 1 | SPECTRA performance by cancer type on the held-out test set.**', '',
          '| Cancer type | n | AUROC (95% CI) | Sensitivity at 90% specificity (95% CI) | Sensitivity at 95% specificity (95% CI) |',
          '|---|---|---|---|---|']
    md += [f'| {r.cancer_type} | {r.n} | {auc(r)} | {pct(r, 90)} | {pct(r, 95)} |' for _, r in t.iterrows()]
    k = p[p.label.eq(0)]
    md += ['', f'Each cancer type versus the same 44 non-cancer controls. AUROC 95% CIs from a stratified patient bootstrap (4,000 replicates); '
               f'not given for types with fewer than 10 patients, which are descriptive. Sensitivity at the common cut-offs set on the pooled ROC '
               f'at 90% and 95% specificity (achieved {int((~k.called90).sum())}/44 and {int((~k.called95).sum())}/44), with Wilson 95% intervals.']
    (out / 'SupplementaryTable1_performance_by_cancer_type.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    build()
