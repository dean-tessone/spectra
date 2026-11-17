"""Sensitivity by cancer type at the common 90% and 95% specificity cut-offs (held-out test set, 44 non-cancer controls).
Calls follow the sealed test report: probability >= cut-off. Wilson 95% intervals for each proportion."""
import numpy as np
import pandas as pd
import sx


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def build():
    p = sx.primary_predictions(); c = p[p.label.eq(1)]
    rows = []
    order = c.cancer_types.value_counts().index.tolist()
    for name, g in [('All cancers', c)] + [(t, c[c.cancer_types.eq(t)]) for t in order]:
        r = dict(cancer_type=name if name == 'All cancers' else sx.short(name), n=len(g))
        for t, col in [(90, 'called90'), (95, 'called95')]:
            k = int(g[col].sum()); lo, hi = wilson(k, len(g))
            r[f'sens{t}_k'] = k; r[f'sens{t}'] = k / len(g); r[f'sens{t}_ci_low'] = lo; r[f'sens{t}_ci_high'] = hi
        rows.append(r)
    t = pd.DataFrame(rows)
    assert t.loc[0, 'sens90_k'] == 403 and t.loc[0, 'sens95_k'] == 283          # sealed test report
    out = sx.ROOT / 'final_figures'
    t.to_csv(out / 'Table_sensitivity_by_cancer_type.csv', index=False)
    fmt = lambda r, s: f"{100 * r[f'sens{s}']:.1f}% ({r[f'sens{s}_k']}/{r.n})"
    md = ['| Cancer type | n | Sensitivity at 90% specificity | Sensitivity at 95% specificity |', '|---|---|---|---|']
    md += [f'| {r.cancer_type} | {r.n} | {fmt(r, 90)} | {fmt(r, 95)} |' for _, r in t.iterrows()]
    k = p[p.label.eq(0)]
    md += ['', f'Held-out test set; 44 non-cancer reference controls. Common cut-offs set on the pooled ROC at 90% and 95% specificity '
               f'(achieved {int((~k.called90).sum())}/44 and {int((~k.called95).sum())}/44). Wilson 95% intervals are in the CSV.']
    (out / 'Table_sensitivity_by_cancer_type.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    build()
