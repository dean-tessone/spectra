"""Supplementary Fig. 1 - from blood slide to model input: BLUE v2 encoder training data, per-patient cell funnel and what the
model-input cells are (frozen BLUE v2 ten-class classifier)."""
import os
import numpy as np
import pandas as pd
import sx
from sx import INK, MUTED, SPECTRA, SPECTRA_L, CK, CTC

ENC_SHEET = os.environ.get('BLUEV2_TRAINING_SHEET', 'raw/bluev2_training_slides.csv')   # read-only; encoder training slide list
ENC_CROPS = 3_824_963   # counted from the encoder's HDF5 training set (691 slides), 2026-11-16
NAMES = {'Pancreatic Ductal Adenocarcinoma (PDAC)': 'Pancreatic', 'Upper Tract Urothelial Carcinoma (UTUC)': 'UTUC', 'UvealMelanoma': 'Uveal melanoma',
         'Myocardial Infraction': 'Myocardial infarction*', 'COVID19': 'COVID-19*', 'Multiple Myeloma': 'Myeloma'}


def violins(ax, rows, cols):
    for i, ((nm, v), c) in enumerate(zip(rows, cols)):
        v = np.log10(np.clip(v, .3, None))
        parts = ax.violinplot([v], positions=[i], orientation='horizontal', widths=.75, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(c); pc.set_alpha(.85); pc.set_edgecolor('none')
        med = np.median(v); ax.plot([med, med], [i - .3, i + .3], color=INK, lw=.9)
        mv = float(np.median(np.asarray(rows[i][1])))
        ax.text(1.02, i, f'{mv:,.0f}' if mv >= 1 else '<1', ha='left', va='center', fontsize=5.8, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], linespacing=1.15); ax.set_ylim(len(rows) - .4, -.6)
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.text(1.02, -.75, 'median', ha='left', fontsize=5.5, color=MUTED, va='bottom', transform=ax.get_yaxis_transform())


def build():
    sx.style()
    pg = sx.Page(7.0, 3.05)

    # a: encoder training data
    pg.letter(.02, .04, 'a')
    a = pg.ax(1.02, .55, 1.0, 2.0, panel='a')
    enc = pd.read_csv(ENC_SHEET)
    n = enc.cancer_type.map(lambda t: NAMES.get(t, t)).value_counts().sort_values()
    a.barh(range(len(n)), n.values, color=[MUTED if '*' in k else SPECTRA for k in n.index], height=.7, lw=0)
    a.set_yticks(range(len(n))); a.set_yticklabels(n.index, fontsize=5.6); a.tick_params(axis='y', length=0); a.spines['left'].set_visible(False)
    for i, v in enumerate(n.values):
        a.text(v + 3, i, str(v), va='center', fontsize=5.4)
    a.set_xlim(0, 165); a.set_xlabel('Slides')
    pg.text(.22, .04, f'BLUE v2 encoder: {len(enc)} slides, {enc.patient_id.nunique()} patients', fontsize=6.2, fontweight='bold')
    pg.text(.22, .17, f'{ENC_CROPS/1e6:.1f} million rare-event crops; ResNet-18,\ncontrastive learning without labels', fontsize=5.8, linespacing=1.15)
    pg.text(.22, 2.80, '*Non-cancer conditions', fontsize=5.4, color=MUTED)

    m = sx.markers(); m = m[m.primary_cancer_vs_reference_eligible]
    ps = lambda c: m[c]          # cells per patient: summed over the patient's analysed slides (two for 2,467 of 2,550)

    # b: funnel
    pg.letter(2.32, .04, 'b')
    b = pg.ax(3.55, .55, 1.0, 2.0, panel='b')
    rows = [('Rare-event candidates\n(anomaly detection)', ps('candidate_events')), ('Pass image-quality\nfilter', m.candidate_events - m.QC_rejected),
            ('Pass leukocyte and\nintensity filters\n(seen by SPECTRA)', ps('assessed_events'))]
    violins(b, rows, [SPECTRA_L, SPECTRA_L, SPECTRA])
    b.set_xlim(1.9, 4.5); b.set_xticks([2, 3, 4]); b.set_xticklabels(['100', '1k', '10k']); b.set_xlabel('Cells per patient')
    pg.text(2.52, .04, 'All nucleated cells: ≈5 million per patient', fontsize=6.2, fontweight='bold')
    pg.text(2.52, .17, f'Medians over {len(m):,} primary-analysis patients', fontsize=5.8)

    # c: what the model-input cells are
    pg.letter(4.95, .04, 'c')
    c = pg.ax(6.05, .55, .62, 2.0, panel='c')
    imm = m.Lymph + m.Gran + m.Mono
    rows = [('CTC-like (3 classes)', ps('CTC_like')), ('  imCTC', ps('imCTCs')), ('  CTC', ps('CTCs')), ('  pcCTC', ps('pcCTCs')),
            ('Large nuclei', ps('Large Nuclei')), ('Immune-like', imm), ('Endothelial-like', ps('CECs')),
            ('Megakaryocyte-like', ps('Megakaryocytes')), ('Fibroblast-like', ps('Fibroblasts')), ('CK-gated', ps('CK_gate'))]
    violins(c, rows, [CTC] * 4 + [MUTED] + [SPECTRA] * 4 + [CK])
    for t in c.get_yticklabels():
        t.set_fontsize(5.6)
    c.set_xlim(-.6, 3.5); c.set_xticks([-.52, 1, 3]); c.set_xticklabels(['0', '10', '1k']); c.set_xlabel('Cells per patient')
    pg.text(5.15, .04, 'Cells seen by SPECTRA, by class', fontsize=6.2, fontweight='bold')

    sx.export(pg, 'supplementary', 'SupplementaryFig1_cell_funnel', row_groups=[['a', 'b', 'c']],
              exemptions=[dict(panels=['a', 'b', 'c'], checks=['panel-width', 'horizontal-gutter'], reason='label-heavy horizontal panels')])
    src = {r[0].replace('\n', ' '): r[1].describe() for r in rows}
    pd.DataFrame(src).to_csv(sx.ROOT / 'supplementary/SupplementaryFig1_cell_funnel_source.csv')
    n.to_csv(sx.ROOT / 'supplementary/SupplementaryFig1_encoder_training_slides_source.csv')


if __name__ == '__main__':
    build()
