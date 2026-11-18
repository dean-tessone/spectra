"""Supplementary Tables 2 and 3 (revision 2026-11-17).

Table 2 - compact summary by cancer type and control source: studies, collection sites, patients (revised 2026-11-18, owner: the
          per-study table was too large for the paper). The full per-study listing (49 studies; site, scan years, scanners, tube)
          is Supplementary Data 1 (analyses/revision_20261117/confounding/study_acquisition_table.csv).
Table 3 - in silico cell-removal experiments on the locked model (analyses/revision_20261117/ablation/*, ck_gate_grid/*).
Writes CSV and Markdown to final_figures/ and a LaTeX tabular body to final_figures/tex/.
"""
import sys
import numpy as np
import pandas as pd
import sx
sys.path.insert(0, str(sx.REPO / 'robustness'))
from sites import patient_sites  # noqa: E402

R = sx.REPO / 'analyses/revision_20261117'
OUT = sx.ROOT / 'final_figures'; TEX = OUT / 'tex'; TEX.mkdir(parents=True, exist_ok=True)
ROLE = {'cancer': 'Cancer', 'reference control': 'Reference', 'reserved control': 'Reserved',
        'cancers + reserved controls': 'Cancer + reserved'}
SITE = {'USC (JN)': 'USC', 'Unknown (S1222)': 'NR'}


def tex_escape(s):
    return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('–', '--')


def table2():
    t = pd.read_csv(R / 'confounding/study_acquisition_table.csv')
    t['cancer_types'] = t.cancer_types.str.replace('Control, ', '').str.replace(', Control', '').str.replace('Control', '–')
    t['cancer_types'] = t.cancer_types.replace({'Upper Tract Urothelial Carcinoma': 'UTUC', 'HeadNeck, Lung': 'Head and neck, lung',
                                                'Leptomeningeal Disease (LMD)': 'Leptomeningeal disease'})
    t['scanners'] = t.scanners.fillna('').map(lambda s: ', '.join(f'BZ{int(float(x))}' for x in s.split(', ') if x) or 'NR')
    t['scan_years'] = t.scan_years.fillna('NR').str.replace(r'^(\d{4})-\1$', r'\1', regex=True).str.replace('-', '–')
    t['tube_type'] = t.tube_type.fillna('NR')
    out = pd.DataFrame({'Study': t.study, 'Role': t.role.map(ROLE), 'Collection site': t.site.replace(SITE), 'Cancer type': t.cancer_types,
                        'Cancers (train / test)': t.cancers_train.astype(str) + ' / ' + t.cancers_test.astype(str),
                        'Controls (train / test)': t.controls_train.astype(str) + ' / ' + t.controls_test.astype(str),
                        'Scan years': t.scan_years, 'Scanners': t.scanners, 'Tube': t.tube_type})
    order = {'Cancer': 0, 'Cancer + reserved': 1, 'Reference': 2, 'Reserved': 3}
    out = out.assign(_o=out.Role.map(order)).sort_values(['_o', 'Study']).drop(columns='_o')
    out.to_csv(OUT / 'SupplementaryData1_study_site_acquisition.csv', index=False)
    cap = ('**Supplementary Data 1 | Studies, collection sites and acquisition.** Patients per study in the training and held-out test '
           'partitions. Reference controls form the primary comparison; reserved controls (lung-nodule work-up, BRCA-variant carriers) '
           'were used only for the adaptation analyses (Fig. 6). Collection site from laboratory records; multi-site trials are listed as '
           'trial sites; JN-family patients were collected through USC and Billings Clinic without a per-patient record. Scan years and '
           'scanners from slide acquisition metadata. Tube, blood collection tube type (EPIC controls: type not recorded, retained by a '
           'predefined exception). Role: Cancer, cancer patients; Reference, reference controls; Reserved, reserved controls. NR, not recorded.')
    md = [cap, '', '| ' + ' | '.join(out.columns) + ' |', '|' + '---|' * len(out.columns)]
    md += ['| ' + ' | '.join(map(str, r)) + ' |' for r in out.itertuples(index=False)]
    (OUT / 'SupplementaryData1_study_site_acquisition.md').write_text('\n'.join(md) + '\n')
    return out


CT_ORDER = ['Prostate', 'Breast', 'Lung', 'Bladder', 'Upper Tract Urothelial Carcinoma', 'Multiple Myeloma', 'Pancreatic', 'Uveal Melanoma',
            'Bone', 'HeadNeck', 'Colon', 'Ovarian', 'Leptomeningeal Disease (LMD)']
CT_NAME = {'Upper Tract Urothelial Carcinoma': 'UTUC', 'Multiple Myeloma': 'Myeloma and related states', 'Uveal Melanoma': 'Uveal melanoma',
           'Bone': 'Bone tumors', 'HeadNeck': 'Head and neck', 'Leptomeningeal Disease (LMD)': 'Leptomeningeal disease'}
TRIAL = {'SWOG S1416 sites': 'SWOG S1416', 'NRG-BR002 sites': 'NRG-BR002', 'Gilead sites': 'Gilead'}


def sites_text(g):
    v = g.site.replace({'USC (JN)': 'USC'})
    named = [x for x in v.value_counts().index if x not in TRIAL and not x.startswith('Unknown') and x != 'BP007 unknown']
    out = ', '.join(named)
    trials = [TRIAL[x] for x in v.value_counts().index if x in TRIAL]
    if trials: out += ('; ' if out else '') + ' and '.join(trials) + ' trial sites'
    if v.str.startswith('Unknown').any() or (v == 'BP007 unknown').any(): out += '; one study NR'
    return out


def table2_compact(full):
    led = patient_sites()
    led = led[led.study.isin(set(full.Study))]
    groups = [(CT_NAME.get(ct, ct), led[(led.label == 1) & (led.cancer_types == ct)]) for ct in CT_ORDER]
    ref = led[(led.label == 0) & led.primary_reference_control]
    res = led[(led.label == 0) & ~led.primary_reference_control]
    groups += [('Reference controls', ref)]
    groups += [('Lung-nodule work-up (reserved)', res[res.study.isin(['SU', 'JN2', 'JN2-R'])]),
               ('BRCA-variant carriers (reserved)', res[res.study == 'INTC09'])]
    rows = [dict(Group=n, Studies=g.study.nunique(), **{'Collection sites': sites_text(g)},
                 **{'Patients (train / test)': f"{(g.split == 'train').sum()} / {(g.split == 'test').sum()}"}) for n, g in groups]
    out = pd.DataFrame(rows)
    assert out['Patients (train / test)'].str.split(' / ', expand=True).astype(int).to_numpy().sum() == len(led) == 2653
    out.to_csv(OUT / 'SupplementaryTable2_study_site_summary.csv', index=False)
    cap = ('**Supplementary Table 2 | Studies and collection sites.** Patients by cancer type and control source in the training and '
           'held-out test partitions, with the number of contributing studies and their collection sites (from laboratory records; '
           'multi-site trials are listed as trial sites; sites ordered by number of patients). Reference controls (Scripps normal donors, '
           'Epic Sciences, Cedars-Sinai INTC17-C and benign pancreatic work-up) form the primary comparison; reserved controls were used '
           'only for the adaptation analyses (Fig. 6). JN-family lung patients were collected through USC and Billings Clinic without a '
           'per-patient record and are listed under USC. Per-study counts, scan years, scanners and tube types are in Supplementary Data 1. '
           'NR, not recorded.')
    md = [cap, '', '| ' + ' | '.join(out.columns) + ' |', '|' + '---|' * len(out.columns)]
    md += ['| ' + ' | '.join(map(str, r)) + ' |' for r in out.itertuples(index=False)]
    (OUT / 'SupplementaryTable2_study_site_summary.md').write_text('\n'.join(md) + '\n')
    (TEX / 'SupplementaryTable2_body.tex').write_text('\n'.join(' & '.join(tex_escape(x) for x in r) + r' \\' for r in out.itertuples(index=False)) + '\n')
    return out


def table3():
    a = pd.read_csv(R / 'ablation/removal_full_reporting.csv')
    c = pd.read_csv(R / 'ablation/cluster_removal_reporting.csv')
    g = pd.read_csv(R / 'ck_gate_grid/ck_gate_grid_per_patient.csv')
    rows = []
    name = {'No removal': 'None (intact model)', 'CK-gated cells': 'CK-gated cells', 'All CTC-like cells': 'All CTC-like cells',
            'Top-10 attributed cells': 'Top-10 attributed cells', 'Top-50 attributed cells': 'Top-50 attributed cells',
            'Cells carrying 90% of positive attribution': 'Mass-90 attributed cells'}
    for cond in name:
        r = a[(a.condition == cond) & (a.arm == 'removed')].iloc[0]
        rr = a[(a.condition == cond) & (a.arm == 'matched random')]
        rr = rr.iloc[0] if len(rr) else None
        ci = r.diff_vs_random_ci if isinstance(r.diff_vs_random_ci, str) else ''
        lo, hi = (eval(ci.replace('np.float64', '')) if ci else (np.nan, np.nan))
        rows.append(dict(removed=name[cond], median_cells=int(r.median_targets_cancers), auroc=r.AUROC, auroc_lo=r.AUROC_ci_low, auroc_hi=r.AUROC_ci_high,
                         random_auroc=rr.AUROC if rr is not None else np.nan, diff=r.get('diff_vs_random', np.nan), diff_lo=lo, diff_hi=hi,
                         sens90=r.sens90_locked, spec90=r.spec90_locked, sens90_re=r.sens90_recomputed))
    for cl in ['Cluster 10', 'Cluster 20', 'Cluster 21']:
        r = c[(c.condition == cl) & (c.arm == 'removed')].iloc[0]; rr = c[(c.condition == cl) & (c.arm == 'matched random')].iloc[0]
        rows.append(dict(removed=f'All {cl.lower()} cells', median_cells=int(r.median_removed_cancers), auroc=r.AUROC, auroc_lo=r.ci_low, auroc_hi=r.ci_high,
                         random_auroc=rr.AUROC, diff=r.diff_vs_random, diff_lo=r.diff_ci_low, diff_hi=r.diff_ci_high,
                         sens90=r.sens90_locked, spec90=r.spec90_locked, sens90_re=r.sens90_recomputed))
    d = pd.DataFrame(rows)
    f2 = lambda x: '–' if pd.isna(x) else f'{x:.2f}'
    pct = lambda x: f'{100 * x:.1f}%'
    out = pd.DataFrame({'Cells removed': d.removed, 'Median per cancer patient': d.median_cells.replace({0: '–'}),
                        'AUROC (95% CI)': [f'{f2(x)} ({f2(l)}–{f2(h)})' for x, l, h in zip(d.auroc, d.auroc_lo, d.auroc_hi)],
                        'AUROC, matched random removal': d.random_auroc.map(f2),
                        'Difference vs random (95% CI)': ['–' if pd.isna(x) else f'{x:+.3f} ({l:+.3f} to {h:+.3f})' for x, l, h in zip(d['diff'], d.diff_lo, d.diff_hi)],
                        'Sensitivity at original 90% cut-off (controls below)': [f'{pct(s)} ({sp})' for s, sp in zip(d.sens90, d.spec90)],
                        'Sensitivity at re-estimated 90% cut-off': d.sens90_re.map(pct)})
    out.to_csv(OUT / 'SupplementaryTable3_cell_removal.csv', index=False)
    gg = pd.DataFrame({'CK gate (CK > / CD45/CD31 <)': [f'{int(a_/1000)}k / {int(b_/1000)}k' + (' (primary)' if p_ else '') for a_, b_, p_ in zip(g.CK_gt, g.CD45CD31_lt, g.primary)],
                       'AUROC (95% CI)': [f'{x:.2f} ({l:.2f}–{h:.2f})' for x, l, h in zip(g.AUROC, g.ci_low, g.ci_high)],
                       'Controls with ≥1 gated cell (of 44)': g.controls_with_any})
    gg.to_csv(OUT / 'SupplementaryTable3b_ck_gate_grid.csv', index=False)
    cap = ('**Supplementary Table 3 | In silico cell removal on the locked model, and CK-gate thresholds.** **a**, Each cell set was removed from '
           'every held-out test patient (713 cancers, 44 controls; cancers and controls alike) and the locked three-model ensemble re-scored; '
           'matched random removal deletes the same number of the patient\'s other cells (mean of five draws). AUROC 95% CIs and differences from '
           'stratified patient bootstrap (4,000 replicates). Sensitivity is given at the original cut-off (with the number of controls below it) '
           'and at a cut-off re-estimated on the altered control scores to keep 40 of 44 controls below. Removal alters the input distribution, '
           'so these results show what the model output depends on, not the causal role of the removed cells. **b**, AUROC of per-patient '
           'CK-gated cell counts across nine predefined gate thresholds (16-bit intensities).')
    md = [cap, '', '**a**', '', '| ' + ' | '.join(out.columns) + ' |', '|' + '---|' * len(out.columns)]
    md += ['| ' + ' | '.join(map(str, r)) + ' |' for r in out.itertuples(index=False)]
    md += ['', '**b**', '', '| ' + ' | '.join(gg.columns) + ' |', '|' + '---|' * len(gg.columns)]
    md += ['| ' + ' | '.join(map(str, r)) + ' |' for r in gg.itertuples(index=False)]
    (OUT / 'SupplementaryTable3_cell_removal.md').write_text('\n'.join(md) + '\n')
    (TEX / 'SupplementaryTable3a_body.tex').write_text('\n'.join(' & '.join(tex_escape(x) for x in r) + r' \\' for r in out.itertuples(index=False)) + '\n')
    (TEX / 'SupplementaryTable3b_body.tex').write_text('\n'.join(' & '.join(tex_escape(x) for x in r) + r' \\' for r in gg.itertuples(index=False)) + '\n')
    return out, gg


def build():
    t2 = table2_compact(table2()); t3, t3b = table3()
    print(t2.to_string(index=False)); print(t3.to_string(index=False)); print(t3b.to_string(index=False))


if __name__ == '__main__':
    build()
