"""Shared style, data access and export for the redesigned SPECTRA manuscript figures.

Python/matplotlib only (nature-figure skill, Python backend). No image generation,
no external assets. Every quantitative mark is read from frozen analysis outputs.
"""
import os
from pathlib import Path
import sys, json, hashlib, re
from xml.etree import ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
SKILL = Path(os.environ.get('NATURE_FIGURE_SKILL', 'nature-figure'))
sys.path.insert(0, str(SKILL / 'scripts'))
try:   # optional layout audit (nature-figure helpers)
    from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
except ImportError:
    require_matplotlib_panel_alignment = None

TEST = REPO / 'analyses/reference_control_test_20261110_v1a'
TESTRUN = REPO / 'runs/reference_control_test_20261110_v1a'
MARK = REPO / 'analyses/bluev2_ctc_comparison_20261110_v1b'
LEDGER = REPO / 'cohort/reference_policy/population_patient_ledger.csv'
ATTR = REPO / 'runs/figure3_locked6_attribution_20261111'
MILCV = REPO / 'analyses/reference_mil_cv_20261110'

# ---- palette: one signal family (SPECTRA), one neutral family, two marker accents ----
INK = '#1F2A30'
MUTED = '#6B7780'
RULE = '#C9D1D6'
FAINT = '#EEF2F4'
SPECTRA = '#0E6E7E'      # hero
SPECTRA_D = '#08434D'
SPECTRA_L = '#9ECBD2'
CONTROL = '#E8892B'          # controls (orange; contrasts with teal cancers)
CK = '#C0392B'           # CK gate (red, matches CK channel)
CTC = '#6A4C93'          # automated CTC-like (purple)
GAIN = '#2F8F5B'; LOSS = '#B4483C'
CH = dict(DAPI='#3D6FD9', CK='#E0443A', CD45='#3FAE5A', VIM='#9AA3A8')  # channel swatches (display convention)

SHORT = {'Upper Tract Urothelial Carcinoma': 'UTUC', 'Multiple Myeloma': 'Myeloma & related', 'Uveal Melanoma': 'Uveal melanoma',
         'HeadNeck': 'Head and neck', 'Leptomeningeal Disease (LMD)': 'Leptomeningeal', 'Bone': 'Bone tumors',
         'Control': 'Non-cancer controls'}
CANCER_COL = {'Prostate': '#3F6E9E', 'Breast': '#C4687A', 'Lung': '#3E8C74', 'Bladder': '#8C73B0',
              'Upper Tract Urothelial Carcinoma': '#B88A3E', 'Multiple Myeloma': '#5F8FB7', 'Pancreatic': '#C27A4D',
              'Uveal Melanoma': '#7C8C3E', 'Bone': '#9A9A9A', 'HeadNeck': '#9A9A9A', 'Colon': '#9A9A9A', 'Ovarian': '#9A9A9A',
              'Leptomeningeal Disease (LMD)': '#9A9A9A', 'Control': CONTROL}


def short(t): return SHORT.get(t, t)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next(n for n in ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'] if n in names)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': [font], 'font.size': 6.5,
        'axes.labelsize': 6.5, 'axes.titlesize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
        'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.4, 'ytick.major.size': 2.4,
        'xtick.major.pad': 1.8, 'ytick.major.pad': 1.8, 'axes.labelpad': 2.5,
        'legend.frameon': False, 'legend.fontsize': 6, 'pdf.fonttype': 42, 'text.parse_math': True, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': INK, 'lines.solid_capstyle': 'round'})
    return font


class Page:
    """Figure laid out in physical inches measured from the top-left corner."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.panels = {}

    def ax(self, x, y, w, h, panel=None, **kw):
        a = self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h], **kw)
        if panel:
            self.panels[a] = panel
        return a

    def letter(self, x, y, s):
        self.fig.text(x / self.w, 1 - y / self.h, s, fontsize=8, fontweight='bold', ha='left', va='top', color=INK)   # Nature: 8 pt bold panel letters

    def text(self, x, y, s, **kw):
        kw.setdefault('va', 'top'); kw.setdefault('ha', 'left')
        return self.fig.text(x / self.w, 1 - y / self.h, s, **kw)


def export(page, folder, name, row_groups=(), column_groups=(), exemptions=()):
    fig = page.fig
    # house style: black text only (white kept for labels on images/filled markers); legend entries capitalised
    from matplotlib.colors import to_rgb
    for t in fig.findobj(Text):
        if not t.get_text().strip():
            continue
        if to_rgb(t.get_color()) != (1.0, 1.0, 1.0):
            t.set_color('black')
    for lg in [a_.get_legend() for a_ in fig.axes] + list(fig.legends):
        if lg is None:
            continue
        for t in lg.get_texts():
            s_ = t.get_text()
            if s_ and s_[0].islower():
                t.set_text(s_[0].upper() + s_[1:])
    folder = ROOT / folder; folder.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    axes = list(page.panels)
    ids = {a: page.panels[a] for a in axes}
    if len(axes) < 2 or require_matplotlib_panel_alignment is None:
        (folder / f'{name}.alignment.json').write_text(json.dumps(dict(status='NOT APPLICABLE', reason='single-panel figure')))
    else:
      require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=ids, row_groups=[dict(id=f'r{i}', panels=g) for i, g in enumerate(row_groups)],
                                       column_groups=[dict(id=f'c{i}', panels=g) for i, g in enumerate(column_groups)],
                                       exemptions=list(exemptions), json_out=folder / f'{name}.alignment.json',
                                       overlay_svg=folder / f'{name}.alignment.svg', tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    renderer = fig.canvas.get_renderer(); W, H = fig.canvas.get_width_height(); sizes = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        sizes.append(t.get_fontsize())
        if t.get_fontsize() < 5:
            raise ValueError('glyph < 5 pt: ' + t.get_text())
        b = t.get_window_extent(renderer)
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > W + .5 or b.y1 > H + .5:
            raise ValueError('text off page: ' + t.get_text() + ' ' + str(b.extents) + ' page ' + str((W, H)) + ' axes ' + str(page.panels.get(t.axes)))
    assert np.all(fig.get_size_inches() <= 7.0001)   # width <= 180 mm (Nature double column 180-183 mm)
    assert fig.get_size_inches()[1] <= 170 / 25.4 + 1e-4, 'figure taller than 170 mm (Nature maximum height)'
    fig.savefig(folder / f'{name}.pdf'); fig.savefig(folder / f'{name}.svg')
    svg = folder / f'{name}.svg'; txt = re.sub(r'<!DOCTYPE.*?>', '', svg.read_text(), flags=re.S); svg.write_text(txt)
    fig.savefig(folder / f'{name}.png', dpi=300)
    from PIL import Image as _Im   # Nature: RGB artwork (matplotlib writes RGBA); flatten onto the white page
    _p = folder / f'{name}.png'; _i = _Im.open(_p); _b = _Im.new('RGB', _i.size, 'white'); _b.paste(_i, mask=_i.getchannel('A')); _b.save(_p, dpi=(300, 300))
    fig.savefig(folder / f'{name}.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    (folder / f'{name}.render.json').write_text(json.dumps(dict(size_in=[page.w, page.h], min_font_pt=min(sizes), backend='Python/matplotlib'), indent=1))
    plt.close(fig)


# ------------------------------------------------------------------ data
def primary_predictions():
    pred = pd.read_csv(TESTRUN / 'private_primary_donor_predictions.csv', dtype={'donor_id': str}, float_precision='round_trip')
    led = pd.read_csv(LEDGER, dtype={'donor_id': str})
    pred = pred.merge(led[['donor_id', 'primary_cancer_vs_reference_eligible']], on='donor_id', validate='one_to_one')
    pred = pred[pred.primary_cancer_vs_reference_eligible].copy()
    assert len(pred) == 757 and pred.label.eq(1).sum() == 713
    ops = operating_points()
    pred['score'] = np.log(pred.probability.clip(1e-9, 1 - 1e-9) / (1 - pred.probability.clip(1e-9, 1 - 1e-9)))
    # official rule of the sealed test report: called when probability >= the ROC-derived cut-off (each cut-off equals one case's
    # score; 1e-12 absorbs the CSV round-trip of that score). Gives 403/713 at 90% and 283/713 at 95% specificity.
    pred['called90'] = pred.probability >= ops[.9] - 1e-12; pred['called95'] = pred.probability >= ops[.95] - 1e-12
    return pred


def operating_points():
    o = pd.read_csv(TEST / 'operating_points.csv')
    o = o[(o.endpoint == 'primary') & (o.cancer_type == 'Overall') & (o.comparison == 'Cancer vs primary reference')]
    return dict(zip(o.target_specificity, o.threshold))


def markers():
    m = pd.read_csv(MARK / 'private_primary_patient_counts.csv', dtype={'donor_id': str})
    return m[m.pool.eq('model_input')]


def logit(p): p = np.clip(p, 1e-9, 1 - 1e-9); return np.log(p / (1 - p))


def roc(y, s):
    from sklearn.metrics import roc_curve, roc_auc_score
    f, t, _ = roc_curve(y, s); return f, t, roc_auc_score(y, s)


def boost(img, target=235, max_gain=4.0, floor=205):
    """Display-only contrast stretch for dim tiles: if the 99.5th percentile of the brightest channel is below
    `floor`, scale linearly so it reaches `target` (gain capped). Applied identically to all channels."""
    img = np.asarray(img)
    hh, ww = img.shape[:2]; c = img[int(hh * .2):int(hh * .8), int(ww * .2):int(ww * .8)]  # centre, where the target cell is
    p = float(np.percentile(c.max(axis=-1), 99.0))
    if p >= floor or p <= 0:
        return img
    g = min(max_gain, target / p)
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def cluster_numbers():
    """Display number for each Leiden cluster (0-based id -> 1..23), ordered along a path through the atlas map."""
    m = pd.read_csv(REPO / 'analyses/figure3_locked6_attribution_20261111/leiden/cluster_display_numbers.csv')
    return dict(zip(m.cluster, m.number))


UM_PER_PX = {'40x': 0.18, '10x': 0.59, 'IMC': 1.0}


def scalebar(ax, n_px, um_per_px, um=5, lw=1.0, frac=.08, y=.9, right=False):
    """White scale bar on an imshow tile drawn in pixel coordinates (default extent); n_px = displayed image width in px,
    um_per_px = micrometres per displayed pixel. Bar length `um` micrometres, bottom left."""
    x0 = (1 - frac) * n_px - um / um_per_px if right else frac * n_px
    ax.plot([x0, x0 + um / um_per_px], [y * n_px] * 2, color='white', lw=lw, solid_capstyle='butt', zorder=5)
