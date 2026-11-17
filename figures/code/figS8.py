"""Supplementary Fig. 6 - adaptation methods compared (lung cancer vs lung-nodule work-up; breast cancer vs BRCA carriers).
Data: analyses/specific_adaptation_20261111 (auroc_by_part.csv) and analyses/bootstrap_ci_20261111 (test AUROC and paired-gain CIs)."""
import pandas as pd
import sx
from sx import INK, MUTED, RULE, SPECTRA

AD = sx.REPO / 'analyses/specific_adaptation_20261111'
CI = sx.REPO / 'analyses/bootstrap_ci_20261111/confidence_intervals.csv'
ARMS = [('source', 'Pan-cancer model, unchanged', '#9AA5AD'),
        ('probe', 'Decision layer refitted (Fig. 6)', SPECTRA),
        ('adapt', 'Last block and decision layer retrained', '#2C7FB8'),
        ('lora', 'LoRA and decision layer retrained', '#8E6BB0')]
TASKS = [('lung', 'Lung cancer vs\nbenign lung nodules'), ('breast', 'Breast cancer vs\nBRCA carriers without cancer')]


def build():
    sx.style()
    auc = pd.read_csv(AD / 'auroc_by_part.csv')
    ci = pd.read_csv(CI).set_index('metric')
    pg = sx.Page(7.0, 2.33)
    K = len(ARMS); top, h, w = .50, 1.30, 1.15
    xa, xb = [1.62, 2.90], [4.45, 5.85]

    lab = pg.ax(.10, top, 1.42, h); lab.set_xlim(0, 1); lab.set_ylim(K - .5, -.5); lab.axis('off')
    for i, (_, name, col) in enumerate(ARMS):
        lab.text(1, i, name, ha='right', va='center', fontsize=5.8)

    rows = []
    pg.letter(.02, .04, 'a'); pg.letter(4.20, .04, 'b')
    for ti, (task, tname) in enumerate(TASKS):
        # ---------- a: held-out test AUROC with 95% CI ----------
        a = pg.ax(xa[ti], top, w, h)
        cv = auc[(auc.task == task) & auc.part.str.startswith('cv')]
        b = pg.ax(xb[ti], top, 1.05, h)
        for i, (arm, name, col) in enumerate(ARMS):
            r = ci.loc[f'Adaptation {task} AUROC {arm}']
            a.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=1.0, solid_capstyle='butt', zorder=1)
            a.scatter([r.estimate], [i], s=18, color=col, zorder=2)
            a.text(r.ci_high + .02, i, f'{r.estimate:.2f}', fontsize=5.6, va='center', ha='left')
            f = cv[cv.arm == arm].sort_values('part')
            b.scatter(f.auroc, [i] * len(f), s=9, facecolor='white', edgecolor=col, lw=.8, zorder=2)
            b.plot([f.auroc.mean()] * 2, [i - .28, i + .28], color=col, lw=1.4, solid_capstyle='butt', zorder=3)
            row = dict(task=task, method=name, test_auroc=r.estimate, test_ci_low=r.ci_low, test_ci_high=r.ci_high,
                       cases=int(r.cases), controls=int(r.controls), cv_mean_fold_auroc=f.auroc.mean())
            row.update({f'cv_fold{k}_auroc': v for k, v in enumerate(f.auroc)})
            if arm != 'source':
                g = ci.loc[f'Adaptation {task} AUROC gain {arm} minus source (paired)']
                row.update(test_gain=g.estimate, test_gain_ci_low=g.ci_low, test_gain_ci_high=g.ci_high)
            rows.append(row)
        for ax_, lim, ticks in [(a, (.2, 1.05), [.2, .4, .6, .8, 1.0]), (b, (.4, 1.0), [.4, .6, .8, 1.0])]:
            ax_.set_ylim(K - .5, -.5); ax_.set_yticks([]); ax_.spines['left'].set_visible(False)
            ax_.set_xlim(*lim); ax_.set_xticks(ticks); ax_.set_xticklabels([f'{t:.1f}' for t in ticks])
            ax_.axvline(.5, color=RULE, lw=.6, zorder=0)
            ax_.set_title(tname, fontsize=5.8, pad=4, linespacing=1.15)
        a.set_xlabel('Test AUROC (95% CI)')
        b.set_xlabel('Training CV AUROC')
    pg.text(xb[0], top + h + .36, 'Open circles, the three training folds; line, mean', fontsize=5.6, color=MUTED)
    sx.export(pg, 'supplementary', 'SupplementaryFig6_adaptation_methods')
    pd.DataFrame(rows).to_csv(sx.ROOT / 'supplementary/SupplementaryFig6_source.csv', index=False)


if __name__ == '__main__':
    build()
