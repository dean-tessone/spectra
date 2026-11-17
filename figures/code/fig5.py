"""Figure 5 - adapting SPECTRA to harder diagnostic questions, and how adaptation changes which cells it uses
(lung cancer vs lung-nodule work-up; breast cancer vs BRCA carriers)."""
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import sx
from sx import INK, RULE, SPECTRA, CONTROL

sys.path.insert(0, str(sx.REPO / 'imaging/cell_images'))
from cell_ext_display import rgb

AD = sx.REPO / 'analyses/specific_adaptation_20261111'
L = sx.REPO / 'analyses/figure3_locked6_attribution_20261111/leiden'
C40 = sx.REPO / 'analyses/figure3_locked6_attribution_20261111/cache_40x'
PAL = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC',
       '#2F4B7C', '#A05195', '#D45087', '#F95D6A', '#FF7C43', '#665191', '#8CD17D', '#B6992D', '#499894', '#86BCB6', '#B5BD61', '#79706E', '#17BECF']
ARMS = [('source', 'Pan-cancer model, unchanged', '#9AA5AD'), ('probe', 'Adapted: decision layer refitted', SPECTRA)]   # other methods: Supplementary Fig. 7
TASKS = [('lung', 'Lung cancer vs\nbenign lung nodules', None), ('breast', 'Breast cancer vs\nBRCA carriers without cancer', None)]


def cap(s):
    return s[0].upper() + s[1:]


def build():
    sx.style()
    pr = pd.read_csv(AD / 'predictions_private.csv', dtype={'donor_id': str}); pr = pr[pr.part == 'test']
    auc = pd.read_csv(AD / 'auroc_by_part.csv'); test = auc[auc.part == 'test'].set_index(['task', 'arm'])
    names = pd.read_csv(L / 'cluster_names.csv', index_col=0).name
    NUM = sx.cluster_numbers()
    pg = sx.Page(7.0, 5.62)

    # ---------- a: AUROC gain ----------
    pg.letter(.02, .04, 'a')
    a = pg.ax(.50, .40, 2.55, 1.55, panel='a')
    w = .30
    for ti, (task, tname, _) in enumerate(TASKS):
        base = test.loc[(task, 'source')].auroc
        for ai, (arm, an, c1) in enumerate(ARMS):
            v = test.loc[(task, arm)].auroc
            a.bar(ti + (ai - .5) * w, v - .4, bottom=.4, width=w * .92, color=c1)
            a.text(ti + (ai - .5) * w, v + .01, f'{v:.2f}', ha='center', va='bottom', fontsize=5.6)
        new = test.loc[(task, 'probe')].auroc
        a.text(ti, new + .07, f'+{new - base:.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    a.set_xticks([0, 1]); a.set_xticklabels([t[1] for t in TASKS], linespacing=1.2); a.tick_params(axis='x', length=0)
    a.set_ylim(.4, 1.05); a.set_yticks([.4, .6, .8, 1.0]); a.set_ylabel('Test AUROC')
    a.legend(handles=[Patch(fc=c1, label=an) for _, an, c1 in ARMS], loc='lower left', bbox_to_anchor=(0, 1.02), ncol=1, fontsize=5.6,
             borderaxespad=0, handlelength=1.0)

    # ---------- b: average score rank, cancers vs controls ----------
    pg.letter(3.45, .04, 'b')
    for ti, (task, tname, _) in enumerate(TASKS):
        ax = pg.ax(3.95 + ti * 1.55, .55, 1.25, 1.40, panel=f'b{ti}')
        d = pr[(pr.task == task) & pr.arm.isin(['source', 'probe'])].copy()
        d['pct'] = d.groupby('arm').probability.rank(pct=True) * 100
        vals = {lab: [d[(d.arm == arm) & (d.label == lab)].pct.mean() for arm in ['source', 'probe']] for lab in (1, 0)}
        for lab, col, mk in [(1, SPECTRA, 'o'), (0, CONTROL, 'D')]:
            v = list(vals[lab])
            if abs(vals[1][0] - vals[0][0]) < 6:
                v = [v[0] + (3 if lab == 1 else -3), v[1]]
                vals_draw = None
            ax.plot([0, 1], vals[lab], color=col, lw=1.8, marker=mk, ms=4.5, zorder=3)
            ax.text(-.08, v[0], f'{vals[lab][0]:.0f}', ha='right', va='center', fontsize=5.6)
            ax.text(1.08, v[1], f'{v[1]:.0f}', ha='left', va='center', fontsize=5.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Pan-cancer', 'Adapted'], fontsize=5.8); ax.tick_params(axis='x', length=0); ax.set_xlim(-.4, 1.4)
        ax.set_ylim(0, 75); ax.set_yticks([0, 25, 50, 75])
        if ti == 0:
            ax.set_ylabel('Mean score percentile\nin test cohort (%)', linespacing=1.2)
        else:
            ax.set_yticklabels([])
        ax.set_title(tname.replace('\n', ' ').replace('Lung cancer vs ', 'Lung vs ').replace('Breast cancer vs ', 'Breast vs '), fontsize=5.8, pad=3)
    pg.fig.legend(handles=[Line2D([], [], color=SPECTRA, marker='o', ms=4, lw=1.6, label='Cancers'), Line2D([], [], color=CONTROL, marker='D', ms=4, lw=1.6, label='Diagnostic controls')], loc='upper left',
                  bbox_to_anchor=(3.95 / pg.w, 1 - .08 / pg.h), ncol=2, fontsize=5.6, handlelength=1.8)

    # ---------- c, d: clusters whose share of the evidence changes most ----------
    import textwrap
    N_SHOW = 6   # shown: the six clusters with the largest equal-weight mean change that moves the same way in cancers and controls
    ckc = pd.read_csv(sx.REPO / 'analyses/population_analyses/adaptation_CK_share_change.csv')
    Y0 = 2.62
    for ti, (task, tname, _) in enumerate(TASKS):
        x0 = .02 + ti * 3.5
        pg.letter(x0, 2.32, 'cd'[ti])
        sh = pd.read_csv(AD / f'attribution/{task}_cluster_share_shift.csv')
        pv = sh.pivot_table(index='cluster', columns='label', values='delta_probe').fillna(0) * 100
        both = pv.mean(axis=1); same = both[np.sign(pv[0]) == np.sign(pv[1])]; keep = same.reindex(same.abs().sort_values(ascending=False).index[:N_SHOW]).sort_values(ascending=False)
        ck = ckc[(ckc.task == task) & (ckc.clusters == 'all CK+ clusters') & (ckc['head'] == 'probe')].change_pp.mean()
        n = len(keep) + 1; RH = .19
        ax = pg.ax(x0 + 1.55, Y0, 1.55, RH * n, panel=f'c{ti}')
        for k_, (k, v) in enumerate(keep.items()):
            ax.barh(k_, v, height=.62, color=SPECTRA if v > 0 else '#9AA5AD', lw=0)
            ax.text(v + (.25 if v > 0 else -.25), k_, f'{v:+.1f}', ha='left' if v > 0 else 'right', va='center', fontsize=5.4)
            pg.text(x0 + .05, Y0 + k_ * RH + RH / 2, f'{NUM[k]}  {cap(names[k])}', fontsize=5.4, va='center')
        ax.barh(n - 1, ck, height=.62, color='none', edgecolor=SPECTRA if ck > 0 else '#7D878E', hatch='//////', lw=.6)
        ax.text(ck + (.25 if ck > 0 else -.25), n - 1, f'{ck:+.1f}', ha='left' if ck > 0 else 'right', va='center', fontsize=5.4)
        pg.text(x0 + .05, Y0 + (n - 1) * RH + RH / 2, 'All CK+ clusters combined', fontsize=5.4, va='center')
        ax.axhline(n - 1.5, color=RULE, lw=.4)
        ax.plot([0, 0], [n - .5, -.5], color=INK, lw=.6); ax.set_ylim(n - .5, -.5); ax.set_yticks([]); ax.spines['left'].set_visible(False)
        lo, hi = min(keep.min(), ck, -1) * 1.35, max(keep.max(), ck, 1) * 1.35; ax.set_xlim(lo, hi)
        step = 5 if hi - lo > 12 else 2; ax.set_xticks([t for t in np.arange(-20, 21, step) if lo <= t <= hi]); ax.tick_params(axis='x', labelsize=5.4)
        ax.set_xlabel('Change in share of the evidence (percentage points)', fontsize=5.6)
        pg.text(x0 + 1.75, 2.36, tname.replace('\n', ' '), fontsize=6, ha='center', va='top')

    # ---------- e, f: individual 40x cells that gain or lose the most weight ----------
    FC = sx.REPO / 'analyses/population_analyses/fig5_cells'
    z = np.load(FC / 'updown_fixed_field_private.npz'); UMPX = float(z['um_per_px_display']); ud = pd.read_csv(FC / 'updown_private.csv')
    TW, TG, E0 = .385, .03, 4.30
    for ti, (task, tname, _) in enumerate(TASKS):
        x0 = .02 + ti * 3.5
        pg.letter(x0, E0 - .06, 'ef'[ti])
        pg.text(x0 + 1.75, E0 - .02, tname.replace('\n', ' '), fontsize=6, ha='center', va='top')
        for ri, (dirn, lab) in enumerate([('up', 'More weight\nafter adaptation'), ('down', 'Less weight\nafter adaptation')]):
            y = E0 + .32 + ri * (TW + .20)
            pg.text(x0 + .05, y - .03, lab.replace('\n', ' '), fontsize=5.6, va='bottom')
            cells = ud[(ud.task == task) & (ud.direction == dirn)].reset_index(drop=True)
            for j, r in cells.iterrows():
                th = pg.ax(x0 + .05 + j * (TW + TG), y, TW, TW); th.set_xticks([]); th.set_yticks([])
                im_ = z[str(int(r.tile))]; th.imshow(im_, interpolation='lanczos'); sx.scalebar(th, im_.shape[0], UMPX, um=10, lw=.8)
                for sp in th.spines.values():
                    sp.set_visible(False)   # no frames
    sx.export(pg, 'figure5', 'Figure5', row_groups=[['b0', 'b1'], ['c0', 'c1']],
              exemptions=[dict(panels=['c0', 'c1'], checks=['row'], reason='bar panels sized to their number of clusters')])
    auc.to_csv(sx.ROOT / 'figure5/adaptation_AUROC_source.csv', index=False)
    for task, _, _ in TASKS:
        sh = pd.read_csv(AD / f'attribution/{task}_cluster_share_shift.csv'); sh['display_number'] = sh.cluster.map(NUM)
        sh.to_csv(sx.ROOT / f'figure5/{task}_cluster_share_shift_by_group_source.csv', index=False)
    ud[['task', 'direction', 'label', 'source_share', 'probe_share', 'delta_pp']].to_csv(sx.ROOT / 'figure5/example_cells_evidence_change_source.csv', index=False)


if __name__ == '__main__':
    build()
