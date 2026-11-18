"""Supplementary Fig. 3 - study, collection-site and acquisition checks (revision 2026-11-17).
Data: analyses/revision_20261117/{confounding,leukocyte_control}. Prostate omitted (cohorts of mixed, unrecorded disease state); breast S1416 omitted (triple-negative; the others are HR+). """
import json
import numpy as np, pandas as pd
import sx
from sx import INK, MUTED, RULE, SPECTRA, CONTROL

R = sx.REPO / 'analyses/revision_20261117'
C = R / 'confounding'
SRC_COL = {'EPIC': '#E8892B', 'NBD': '#B5651D', 'INTC17-C': '#D4A017', 'PANC-Ctrl': '#8C5A3C'}
SRC_NAME = {'EPIC': 'Healthy,\nEpic Sci.', 'NBD': 'Healthy,\nScripps', 'INTC17-C': 'Mammo-\ngram neg.', 'PANC-Ctrl': 'Benign\npancreatic'}
SRC_LEG = {'EPIC': 'Epic Sci.', 'NBD': 'Scripps', 'INTC17-C': 'Mammo. neg.', 'PANC-Ctrl': 'Benign panc.'}


def strip(ax, x, v, col, w=.32, s=5, alpha=.85, seed=0):
    j = np.random.default_rng(seed).uniform(-w, w, len(v))
    ax.scatter(x + j, v, s=s, color=col, lw=0, alpha=alpha, zorder=2)
    ax.plot([x - w - .06, x + w + .06], [np.median(v)] * 2, color=INK, lw=.9, zorder=3, solid_capstyle='butt')


def forest(ax, rows, labels):
    for i, r in enumerate(rows):
        for dy, key, fill in ((-.15, 'test', True), (.15, 'oof', False)):
            if r.get(key) is None:
                continue
            a, lo, hi = r[key]
            ax.plot([lo, hi], [i + dy] * 2, color=SPECTRA, lw=.9, zorder=1)
            ax.scatter([a], [i + dy], s=14, facecolor=SPECTRA if fill else 'white', edgecolor=SPECTRA, lw=.8, zorder=2)
    ax.set_ylim(len(rows) - .5, -.6); ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=5.6)
    ax.axvline(.5, color=RULE, lw=.6, zorder=0); ax.set_xlim(0, 1.02); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0', '.25', '.5', '.75', '1'])
    ax.tick_params(axis='y', length=0)


def build():
    sx.style()
    t = pd.read_csv(C / 'test_scores_with_site_private.csv', dtype={'donor_id': str})
    T90 = json.load(open(C / 'summary.json'))['locked_cutoffs_logodds']['spec90']
    src = pd.read_csv(C / 'per_control_source.csv').set_index('control_source')
    same = pd.read_csv(C / 'same_site_comparisons.csv')
    pair = pd.read_csv(C / 'site_pairwise_AUROC.csv'); kw = pd.read_csv(C / 'site_kruskal.csv')
    leuk = json.load(open(R / 'leukocyte_control/summary.json'))
    pg = sx.Page(7.0, 6.55)
    src_rows = []

    # ---------------- a: scores by reference-control source
    a = pg.ax(.42, .30, 2.55, 1.55, panel='a'); pg.letter(.02, .10, 'a')
    ca = t[t.label == 1].score.to_numpy(); order = ['EPIC', 'NBD', 'INTC17-C', 'PANC-Ctrl']
    parts = a.violinplot([ca], positions=[0], widths=.7, showextrema=False)
    for b in parts['bodies']:
        b.set_facecolor(SPECTRA); b.set_alpha(.35); b.set_edgecolor('none')
    a.plot([-.4, .4], [np.median(ca)] * 2, color=INK, lw=.9)
    for i, s in enumerate(order, 1):
        v = t[(t.label == 0) & (t.study == s)].score.to_numpy(); strip(a, i, v, SRC_COL[s], seed=i)
        r = src.loc[s]; a.text(i, 8.6, f'{r.AUROC:.2f}', ha='center', va='bottom', fontsize=5.6)
        src_rows.append(dict(control_source=s, n=len(v), AUROC_vs_713_cancers=r.AUROC, ci_low=r.ci_low, ci_high=r.ci_high, median_score=np.median(v),
                             below_locked_90_cutoff=r.spec90_cutoff, pooled_AUROC_without_source=r.AUROC_without_source))
    a.text(-.55, 8.6, 'AUROC', ha='left', va='bottom', fontsize=5.6)
    a.axhline(T90, color=MUTED, lw=.6, ls=(0, (3, 2)), zorder=0)
    a.set_xticks(range(5)); a.set_xticklabels([f'Cancers\n\n{len(ca)}'] + [f'{SRC_NAME[s]}\n{int(src.loc[s].controls)}' for s in order], fontsize=5.4, linespacing=1.25)
    a.set_xlim(-.6, 4.6); a.set_ylim(-9, 8.5); a.set_ylabel('SPECTRA score (log-odds)')
    pd.DataFrame(src_rows).to_csv(sx.ROOT / 'supplementary/SupplementaryFig3_a_source.csv', index=False)

    # ---------------- b: same-site / same-study comparisons
    b = pg.ax(4.95, .30, 1.85, 1.55, panel='b'); pg.letter(3.30, .10, 'b')
    def get(name, setting):
        r = same[(same.comparison == name) & (same.setting == setting)].iloc[0]; return (r.AUROC, r.ci_low, r.ci_high), int(r.cancers), int(r.controls)
    spec = [('Pooled primary comparison (all sites)', 'All sites, reference\ncontrols'),
            ('Cedars-Sinai pancreatic: PANCEUS vs PANC-Ctrl (one study)', 'Pancreatic vs benign\npancreatic, one study'),
            ('Cedars-Sinai pancreatic: PANCEUS + PANC vs PANC-Ctrl', 'Pancreatic vs benign\npancreatic, one site')]
    rows, labels, out = [], [], []
    for name, lab in spec:
        te, n1, n0 = get(name, 'test'); oo, m1, m0 = get(name, 'training CV out-of-fold')
        rows.append(dict(test=te, oof=oo)); labels.append(f'{lab}  {n1}/{n0}')
        out.append(dict(comparison=name, test_AUROC=te[0], test_ci=(te[1], te[2]), test_cancers=n1, test_controls=n0,
                        oof_AUROC=oo[0], oof_ci=(oo[1], oo[2]), oof_cancers=m1, oof_controls=m0))
    forest(b, rows, labels); b.set_xlabel('AUROC (95% CI)')
    pd.DataFrame(out).to_csv(sx.ROOT / 'supplementary/SupplementaryFig3_b_source.csv', index=False)

    # ---------------- c: cancer scores by site, same cancer type
    groups = [('Breast', ['Billings Clinic', 'Duke', 'Walter Reed', 'Anne Arundel', 'University of Colorado', 'NRG-BR002 sites', 'Unknown (S1222)'],
               ['Billings', 'Duke', 'Walter\nReed', 'Anne\nArundel', 'Colorado', 'NRG\nBR002', 'S1222']),
              ('Lung', ['Stanford', 'USC', 'USC (JN)', 'Manchester', 'MSKCC'], ['Stanford', 'USC', 'USC\n(JN)', 'Man-\nchester', 'MSKCC'])]
    xs, ws = [.42, 4.75], [3.75, 2.10]; pg.letter(.02, 2.45, 'c'); out = []
    for k, ((ct, sites, short), x0, w0) in enumerate(zip(groups, xs, ws)):
        c = pg.ax(x0, 2.62, w0, 1.30, panel=f'c{k}')
        for i, s in enumerate(sites):
            v = t[(t.label == 1) & (t.cancer_types == ct) & (t.site == s)].score.to_numpy()
            strip(c, i, v, sx.CANCER_COL.get(ct, SPECTRA), w=.28, s=3.5, seed=10 * k + i)
            c.text(i, 8.6, f'{len(v)}', ha='center', va='bottom', fontsize=5.2)
            out.append(dict(cancer_type=ct, site=s, n=len(v), median_score=float(np.median(v)) if len(v) else np.nan))
        c.axhline(T90, color=MUTED, lw=.6, ls=(0, (3, 2)), zorder=0)
        c.set_xticks(range(len(sites))); c.set_xticklabels(short, fontsize=5.2, linespacing=1.25); c.set_xlim(-.6, len(sites) - .4); c.set_ylim(-9, 8.5)
        kq = kw[(kw.setting == 'test') & (kw.cancer_type == ct)].iloc[0]
        nn = sum(int(((t.label == 1) & (t.cancer_types == ct) & (t.site == s_)).sum()) for s_ in sites); eps = max((kq.H - len(sites) + 1) / (nn - len(sites)), 0)
        c.set_title(f'{ct}  (Kruskal–Wallis P = {kq.p:.2f}; ε² = {eps:.2f})', fontsize=6, pad=9)
        if k == 0:
            c.set_ylabel('SPECTRA score (log-odds)'); c.text(-.55, 8.6, 'n', ha='left', va='bottom', fontsize=5.2)
        else:
            c.set_yticklabels([])
    pd.DataFrame(out).to_csv(sx.ROOT / 'supplementary/SupplementaryFig3_c_source.csv', index=False)

    # ---------------- d: AUROC between sites, same cancer type
    d = pg.ax(1.62, 4.62, 2.30, 1.55, panel='d'); pg.letter(.02, 4.42, 'd')
    spec = [('Breast', 'Billings Clinic', 'Duke', 'Breast: Billings vs Duke'), ('Breast', 'Walter Reed', 'Anne Arundel', 'Breast: Walter Reed vs\nAnne Arundel'),
            ('Lung', 'Stanford', 'USC', 'Lung: Stanford vs USC'), ('Lung', 'Stanford', 'Manchester', 'Lung: Stanford vs\nManchester')]
    rows, labels = [], []
    for ct, s1, s2, lab in spec:
        r = {}
        for key, setting in (('test', 'test'), ('oof', 'training CV out-of-fold')):
            q = pair[(pair.setting == setting) & (pair.cancer_type == ct) & (pair.site_a == s1) & (pair.site_b == s2)]
            r[key] = tuple(q.iloc[0][['AUROC_site_a_vs_b', 'ci_low', 'ci_high']]) if len(q) else None
        rows.append(r); labels.append(lab)
    forest(d, rows, labels); d.set_xlabel('AUROC between sites')
    pd.DataFrame([dict(comparison=l.replace('\n', ' '), **{f'{k}_{n}': (v[i] if v else np.nan) for k, v in r.items() for i, n in enumerate(['AUROC', 'ci_low', 'ci_high'])})
                  for r, l in zip(rows, labels)]).to_csv(sx.ROOT / 'supplementary/SupplementaryFig3_d_source.csv', index=False)

    # ---------------- e: leukocyte-only bags
    f = pg.ax(5.30, 4.62, 1.55, 1.55, panel='e'); pg.letter(4.85, 4.42, 'e')
    items = [('Cells SPECTRA\nreads', leuk['real_AUROC'], *leuk['real_ci'], SPECTRA), ('Leukocyte-\nonly bags', leuk['leukocyte_AUROC'], *leuk['leukocyte_ci'], MUTED)]
    for i, (lab, v, lo, hi, col) in enumerate(items):
        f.plot([i, i], [lo, hi], color=col, lw=1.0); f.scatter([i], [v], s=16, color=col, zorder=2); f.text(i + .12, v, f'{v:.2f}', fontsize=5.6, va='center')
    f.axhline(.5, color=RULE, lw=.6, zorder=0); f.set_xlim(-.5, 1.7); f.set_ylim(.4, 1.0); f.set_xticks([0, 1]); f.set_xticklabels([x[0] for x in items], fontsize=5.4)
    f.set_ylabel('Test AUROC (95% CI)')
    pd.DataFrame([dict(bags=x[0].replace('\n', ' '), AUROC=x[1], ci_low=x[2], ci_high=x[3]) for x in items]).to_csv(
        sx.ROOT / 'supplementary/SupplementaryFig3_e_source.csv', index=False)

    sx.export(pg, 'supplementary', 'SupplementaryFig3_study_site_checks', row_groups=[['a', 'b'], ['c0', 'c1'], ['d', 'e']],
              exemptions=[dict(panels=['a', 'b', 'c0', 'c1', 'd', 'e'], checks=['panel-width', 'horizontal-gutter'], reason='strip panels sized to number of sites; forest panels need label space')])


if __name__ == '__main__':
    build()
