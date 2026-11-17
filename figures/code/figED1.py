"""Supplementary Fig. 2 - training-CV benchmark, score diagnostics and acquisition batch checks."""
import numpy as np
import pandas as pd
from scipy.special import logit as _logit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import sx
from sx import INK, MUTED, RULE, SPECTRA, SPECTRA_D, SPECTRA_L, CONTROL

NAMES = {'transformer': 'SPECTRA', 'abmil': 'ABMIL', 'mean': 'Mean pooling', 'clam_mb': 'CLAM-MB', 'additive': 'Additive MIL', 'clam_sb': 'CLAM-SB', 'count': 'Cell count'}


def build():
    sx.style()
    pg = sx.Page(7.0, 4.45)
    f = pd.read_csv(sx.MILCV / 'fold_metrics.csv'); f = f[(f.endpoint == 'primary') & f.fold.isin(['0', '1', '2'])].copy(); f['fold'] = f.fold.astype(int)
    cnt = pd.read_csv(sx.REPO / 'analyses/reference_count_cv_20261111/fold_metrics.csv'); cnt = cnt[cnt.fold.astype(str).isin(['0', '1', '2'])].copy()
    cnt['family'] = 'count'; cnt['fold'] = cnt.fold.astype(int)
    f = pd.concat([f[['family', 'fold', 'pooled_AUROC']], cnt[['family', 'fold', 'pooled_AUROC']]])
    order = f.groupby('family').pooled_AUROC.mean().sort_values(ascending=False).index.tolist()

    # a: fold-level CV
    pg.letter(.02, .08, 'a')
    a = pg.ax(.95, .30, 1.75, 1.75, panel='a')
    for i, fam in enumerate(order):
        g = f[f.family == fam].sort_values('fold'); col = SPECTRA if fam == 'transformer' else MUTED
        a.scatter(g.pooled_AUROC, [i] * 3, s=9, color=col, alpha=.55, lw=0)
        mu = g.pooled_AUROC.mean(); a.plot([mu, mu], [i - .3, i + .3], color=col, lw=1.4)
        a.text(.86, i, f'{mu:.3f}', va='center', fontsize=5.8, color=col)
    a.set_yticks(range(len(order))); a.set_yticklabels([NAMES[o] for o in order]); a.set_ylim(len(order) - .4, -.6)
    a.tick_params(axis='y', length=0); a.spines['left'].set_visible(False)
    a.set_xlim(.55, .85); a.set_xticks([.55, .65, .75, .85]); a.set_xlabel('Validation-fold AUROC\n(3 folds x 2 seeds, training cohort)', linespacing=1.2)

    # b: paired gains over folds
    pg.letter(3.0, .08, 'b')
    b = pg.ax(3.55, .30, 1.30, 1.75, panel='b')
    sp = f[f.family == 'transformer'].set_index('fold').pooled_AUROC
    others = [o for o in order if o != 'transformer']
    for i, fam in enumerate(others):
        d = sp - f[f.family == fam].set_index('fold').pooled_AUROC
        b.scatter(d, [i] * 3, s=9, color=SPECTRA, alpha=.6, lw=0); b.plot([d.mean()] * 2, [i - .3, i + .3], color=SPECTRA_D, lw=1.3)
    b.axvline(0, color=MUTED, lw=.6)
    b.set_yticks(range(len(others))); b.set_yticklabels([NAMES[o] for o in others]); b.set_ylim(len(others) - .4, -.6)
    b.tick_params(axis='y', length=0); b.spines['left'].set_visible(False)
    b.set_xlim(-.05, .3); b.set_xticks([0, .1, .2, .3]); b.set_xlabel('SPECTRA minus baseline\n(same fold)', linespacing=1.2)

    # d: seed agreement / saturation on test
    pg.letter(5.05, .08, 'c')
    d = pg.ax(5.55, .30, 1.35, 1.75, panel='c')
    p = sx.primary_predictions(); gids = []
    seeds = {}
    for s in [17, 29, 43]:
        g = pd.read_csv(sx.TESTRUN / f'private_seed_{s}_group_predictions.csv'); g = g[g.primary_evaluation & g.donor_id.isin(p.donor_id)]
        seeds[s] = g.set_index('donor_id').probability
    S = pd.DataFrame(seeds).loc[p.donor_id]
    votes = (S > .5).sum(axis=1)
    sat = float(((S < .02) | (S > .98)).to_numpy().mean())
    for lab, col, off in [(1, SPECTRA, -.18), (0, CONTROL, .18)]:
        v = votes[p.label.values == lab]; frac = v.value_counts(normalize=True).reindex(range(4), fill_value=0)
        d.bar(np.arange(4) + off, frac * 100, width=.34, color=col)
    d.set_xticks(range(4)); d.set_xticklabels(['0/3', '1/3', '2/3', '3/3']); d.set_xlabel('Seeds with probability >0.5')
    d.set_ylabel('Patients (%)'); d.set_ylim(0, 100)
    (sx.ROOT / 'supplementary/SupplementaryFig2_seed_saturation.txt').write_text(f'{sat:.4f}\n')
    from matplotlib.patches import Patch
    d.legend(handles=[Patch(fc=SPECTRA, label='Cancers'), Patch(fc=CONTROL, label='Controls')], loc='upper left', bbox_to_anchor=(0, .80), fontsize=5.6, handlelength=1)

    # e: score vs retained cells
    pg.letter(.02, 2.55, 'd')
    e = pg.ax(.95, 2.80, 1.75, 1.30, panel='d')
    e.scatter(np.log10(p.retained_cells), p.score, s=3, c=np.where(p.label == 1, SPECTRA, CONTROL), lw=0, alpha=.7)
    r = spearmanr(p.retained_cells, p.score)[0]
    e.set_xticks(np.log10([30, 100, 300, 1000, 3000])); e.set_xticklabels(['30', '100', '300', '1k', '3k'])
    e.set_xlabel('Model-input cells per patient'); e.set_ylabel('SPECTRA score (log-odds)')
    e.text(.02, .02, f'Spearman ρ = {r:.2f}', transform=e.transAxes, fontsize=5.6, color=MUTED)

    # f: endpoint sensitivity
    pg.letter(3.0, 2.55, 'e')
    g = pg.ax(3.55, 2.80, 1.30, 1.30, panel='e')
    es = pd.read_csv(sx.TEST / 'endpoint_summary.csv'); es = es[es.comparison == 'Cancer vs primary reference']
    lab = {'primary': 'Primary group\n(reported)', 'strict_same_draw_tube_supplemental': 'Same draw\nand tube', 'all_eligible_groups_supplemental': 'All eligible\ngroups'}
    for i, (_, r) in enumerate(es.iterrows()):
        g.scatter([i], [r.pooled_auroc], s=18, color=SPECTRA if r.endpoint == 'primary' else SPECTRA_L, zorder=3)
        g.scatter([i], [r.macro_auroc], s=14, marker='D', facecolor='white', edgecolor=SPECTRA_D, lw=.7, zorder=3)
    g.set_xticks(range(len(es))); g.set_xticklabels([lab[x] for x in es.endpoint], fontsize=5.3, linespacing=1.2)
    g.set_xlim(-.5, len(es) - .5); g.set_ylim(.80, .83); g.set_yticks([.80, .81, .82, .83]); g.set_ylabel('Test AUROC'); g.tick_params(axis='x', length=0)
    from matplotlib.lines import Line2D
    g.legend(handles=[Line2D([], [], marker='o', ls='', ms=4, mfc=SPECTRA, mec=SPECTRA, label='Pooled'), Line2D([], [], marker='D', ls='', ms=3.5, mfc='white', mec=SPECTRA_D, label='Macro (8 types)')], loc='upper right', fontsize=5.6, handletextpad=.2)

    # f: acquisition batch checks (test cohort, no refits)
    pg.letter(5.05, 2.55, 'f')
    q = pg.ax(5.55, 2.80, 1.35, 1.30, panel='f')
    bs = __import__('json').load(open(sx.REPO / 'analyses/batch_effects_20261111/summary.json'))
    rows = [('All pairs', bs['overall_auroc']), ('Same scanner', bs['scanner_stratified_auroc']), ('Same year', bs['year_stratified_auroc']),
            ('≤180 days apart', bs['within180d_auroc']), ('Same DAPI\nexposure', bs['DAPI_quartile_stratified_auroc'])]
    for i, (lab, v) in enumerate(rows):
        q.plot([.5, v], [i, i], color=RULE, lw=.8, zorder=1); q.scatter([v], [i], s=16, color=SPECTRA if i else INK, zorder=3)
        q.text(v + .025, i, f'{v:.2f}', va='center', fontsize=5.6)
    q.axvline(bs['overall_auroc'], color=MUTED, lw=.6, ls=(0, (2, 2)))
    q.set_yticks(range(len(rows))); q.set_yticklabels([r[0] for r in rows], fontsize=5.6, linespacing=1.15); q.set_ylim(len(rows) - .4, -.6)
    q.tick_params(axis='y', length=0); q.spines['left'].set_visible(False)
    q.set_xlim(.5, 1.0); q.set_xticks([.5, .75, 1.0]); q.set_xlabel('Test AUROC')
    pd.DataFrame([dict(comparison=l.replace('\n', ' '), auroc=v) for l, v in rows]).to_csv(sx.ROOT / 'supplementary/SupplementaryFig2_batch_check_source.csv', index=False)
    sx.export(pg, 'supplementary', 'SupplementaryFig2_benchmark_and_checks', row_groups=[['a', 'b', 'c'], ['d', 'e', 'f']], column_groups=[['a', 'd'], ['b', 'e'], ['c', 'f']],
              exemptions=[dict(panels=['a', 'b', 'c', 'd', 'e', 'f'], checks=['panel-width', 'horizontal-gutter'], reason='category-label widths differ by panel; tops/heights aligned per row')])
    f.to_csv(sx.ROOT / 'supplementary/SupplementaryFig2_fold_AUROC_source.csv', index=False)


if __name__ == '__main__':
    build()
