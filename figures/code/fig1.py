"""Figure 1 - one flow from blood to score: assay workflow ending in the sorted slide, the SPECTRA model, and the primary cohort
(original vector illustrations; cartoon cells only, )."""
import sys
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import sx
import icons as I
from sx import INK, SPECTRA, SPECTRA_D, CONTROL


def build():
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch, Wedge, Polygon
    from matplotlib.colors import LinearSegmentedColormap
    import icons_br as B
    sx.style()
    pg = sx.Page(7.0, 5.74)
    # one palette: grey = ordinary blood cells, red = rare events, teal = everything SPECTRA does with them,
    # purple = the summary token, orange = non-cancer (as in the other figures); channel colours only on the antibodies and the fWSI image
    TEAL, TEAL_L, TEAL_M = SPECTRA, '#CFE4E8', '#9ECBD2'
    GREYS = ['#E4E7EA', '#D0D5DA', '#BAC1C8', '#A4ADB6', '#8E98A2']     # ordinary leukocytes: neutrophils, lymphocytes, monocytes, eosinophils, basophils
    RARE = '#D7263D'                                                     # rare events
    PUR, PUR_D = '#E7E0F3', '#8E6BBF'

    # ---------------- a: workflow, ending in the sorted slide ----------------
    pg.letter(.02, .04, 'a')
    XW, YH = 104.6, 27.0; S1 = 6.84 / XW
    a = pg.ax(.08, .10, 6.84, YH * S1); a.set_xlim(0, XW); a.set_ylim(0, YH); a.set_aspect('equal'); a.axis('off')
    T = 25.6
    def title(x, s_): a.text(x, T, s_, ha='center', va='top', fontsize=7, linespacing=1.1)
    B.tube(a, 1.2, 3.0, w=2.2, h=12.5)
    B.arrow(a, 3.8, 13.0, 6.6, 13.6, lw=1.1, ms=7, curve=-.35)
    B.centrifuge(a, 6.8, 6.0, w=10, h=9.5)
    B.arrow(a, 15.6, 10.0, 15.6, 5.8, lw=1.1, ms=7, curve=-.35)
    sx_, sy_ = B.slide(a, 8.0, 1.6, w=11.0, h=2.6)
    B.zoom_cells(a, 24.5, 11.0, 5.0)
    a.plot([sx_ - .5, 21.0], [sy_ + .9, 7.4], color='#2E2E2E', lw=.5, ls=(0, (2, 1.5)), zorder=2.9)
    a.plot([sx_ + .9, 27.8], [sy_ + .9, 7.2], color='#2E2E2E', lw=.5, ls=(0, (2, 1.5)), zorder=2.9)
    title(15.0, 'Sample Preparation')
    B.arrow(a, 30.5, 11.0, 34.5, 11.0)
    for i_, (c_, nm_) in enumerate([('#3E6FD8', 'DAPI'), ('#C0392B', 'CK'), ('#E0A526', 'Vimentin'), ('#5CB04C', 'CD45'), ('#5CB04C', 'CD31')]):
        yy = 21.0 - i_ * 3.3
        B.antibody(a, 37.0, yy, c_, s=1.05)
        a.text(38.6, yy + .6, nm_, fontsize=6.2, va='center', fontweight='bold')
    B.slide(a, 35.4, 3.0, w=11.0, h=2.6, dz=-1)    # two slides per patient: one behind the other
    B.slide(a, 34.6, 1.6, w=11.0, h=2.6)
    B.microscope(a, 47.6, 1.6, s=1.08)
    title(46.0, 'Stain & Image')
    B.arrow(a, 58.4, 11.0, 62.0, 11.0)
    # fWSI and (later) the sorted slide are the same object, drawn at the same size
    SW, SH, SY = 6.5, 15.0, 2.8
    B.fwsi_slide(a, 63.9, SY + .9, w=SW, h=SH, n=190, seed=8, dz=-1)   # the patient's second slide
    B.fwsi_slide(a, 63.0, SY, w=SW, h=SH, n=190)
    title(63.0 + SW / 2, 'fWSI')
    B.arrow(a, 70.6, 11.0, 73.6, 11.0)
    B.rb(a, 74.2, 8.8, 4.6, 4.6, '#FBFCFD', r=.3, z=3)
    B.cartoon_cell(a, 76.5, 11.1, 1.45, 'peach', z=3.2, outline=TEAL)
    B.arrow(a, 79.2, 11.0, 80.6, 11.0, lw=1.1, ms=7)
    B.network(a, 81.2, 5.5, w=8.0, h=11.0, node=TEAL, node_out=TEAL_M, box='#F2F8F9', box_ec='#C4DDE1', edge='#C4DDE1', bar_ec=TEAL)
    title(82.4, 'Segment & Embed')
    B.arrow(a, 90.8, 11.0, 93.2, 11.0)
    B.histogram(a, 94.4, 4.5, w=8.6, h=12.0, bars=[GREYS[min(k // 2 + 1, 4)] for k in range(8)] + [None, None], tail=RARE, n_tail=2)
    title(98.7, 'Anomaly\nDetection')

    # ---------------- sorted bar + thread: one slide sorted by anomaly score; its rare events go to the model ----------------
    ov = pg.ax(0, 0, pg.w, pg.h); ov.set_xlim(0, pg.w); ov.set_ylim(pg.h, 0); ov.axis('off'); ov.patch.set_alpha(0); ov.set_zorder(6)
    def P1(x, y): return (.08 + x * S1, .10 + (YH - y) * S1)
    MX, MY, MH = .08, 2.46, 3.20                     # model axes origin (inches) and height; 0.1 in per unit, y from -1.5 to 30.5
    def P2(x, y): return (MX + x * .1, MY + (30.5 - y) * .1)
    BRX, BRY = 5.6, 13.4                             # bracket gathering the patient's cells (model units)
    bx0, bx1, by0, bh = 3.62, 6.86, 2.00, .17        # bar: rare events at the left end, leukocytes to the right (shares illustrative)
    xa, ya_ = P1(98.7, 3.5)
    ov.annotate('', xy=(xa, by0 - .03), xytext=(xa, ya_), arrowprops=dict(arrowstyle='-|>', mutation_scale=10, lw=1.6, color='#2E2E2E', shrinkA=0, shrinkB=0), zorder=6)
    # rare events are <0.5% of cells; drawn at 1.5% of the bar so they can be seen (caption says so)
    segs = [('', .015, RARE), ('', .006, GREYS[4]), ('', .03, GREYS[3]), ('', .06, GREYS[2]), ('Lymphocytes', .296, GREYS[1]), ('Neutrophils', .593, GREYS[0])]
    xx = bx0; edges = []
    for nm_, f_, c_ in segs:
        wseg = f_ * (bx1 - bx0); ov.add_patch(Rectangle((xx, by0), wseg, bh, fc=c_, ec='none', zorder=6)); edges.append((xx, xx + wseg))
        if nm_:
            ov.text(xx + wseg / 2, by0 + bh / 2, nm_, ha='center', va='center', fontsize=5.6, zorder=7)
        xx += wseg
    ov.add_patch(Rectangle((bx0, by0), bx1 - bx0, bh, fc='none', ec=INK, lw=.6, zorder=7))
    pg.text(xa - .12, by0 - .035, 'Each slide sorted by anomaly score, ≈2.5 million cells', fontsize=5.8, ha='right', va='bottom')
    sm0, sm1 = edges[1][0], edges[3][1]             # basophils + eosinophils + monocytes
    ov.plot([sm0, sm0, sm1, sm1], [by0 + bh + .025, by0 + bh + .05, by0 + bh + .05, by0 + bh + .025], color=INK, lw=.5, zorder=7)
    pg.text((sm0 + sm1) / 2, by0 + bh + .07, 'Monocytes, eosinophils, basophils', fontsize=5.4, ha='center', va='top')
    yt = by0 + bh / 2; xv = P2(2.0, 0)[0]; xe, ye = P2(BRX - .25, BRY); r_ = .07
    V = [(bx0, yt), (xv + r_, yt), (xv, yt), (xv, yt + r_), (xv, ye - r_), (xv, ye), (xv + r_, ye), (xe - .06, ye)]
    Cc = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO]
    ov.add_patch(PathPatch(MPath(V, Cc), fc='none', ec=RARE, lw=1.1, capstyle='butt', zorder=6))
    ov.add_patch(Polygon([(xe - .07, ye - .045), (xe - .07, ye + .045), (xe + .01, ye)], closed=True, fc=RARE, ec='none', zorder=6))
    pg.text(bx0 - .06, yt - .035, 'Rare events of each patient, <0.5% of cells', fontsize=5.8, ha='right', va='bottom', fontweight='bold')

    # ---------------- b: model ----------------
    pg.letter(.02, 2.39, 'b')
    m = pg.ax(MX, MY, 3.9, MH); m.set_xlim(0, 39); m.set_ylim(-1.5, 30.5); m.set_aspect('equal'); m.axis('off')
    TX, TW_ = BRX + .9, 4.2
    heat = [TEAL, '#3F8E9B', TEAL_M, TEAL, TEAL_L]
    ys = [21.2, 16.6, 12.0, 7.4, 1.6]
    m.add_patch(FancyBboxPatch((TX, 26.0), TW_, 4.0, boxstyle='round,pad=0,rounding_size=.5', fc=PUR, ec=PUR_D, lw=.9, ls=(0, (2.5, 1.6)), zorder=3))
    m.text(TX + TW_ / 2, 28.0, 'Summary\ntoken', ha='center', va='center', fontsize=5.2, linespacing=1.1, zorder=4)
    for i, (yy, kind, rr) in enumerate(zip(ys, ['rose', 'amber', 'mint', 'peach', 'sky'], [1.25, 1.45, .95, 1.2, 1.1])):
        m.add_patch(Rectangle((TX, yy), TW_, 4.0, fc='#FBFCFD', ec=heat[i], lw=1.8, zorder=3))
        B.cartoon_cell(m, TX + TW_ / 2, yy + 2.0, rr, kind, z=3.2)
    for yy in (6.05, 6.5, 6.95):
        m.add_patch(Circle((TX + TW_ / 2, yy), .16, fc=INK, ec='none', zorder=3))
    m.plot([BRX + .5, BRX, BRX, BRX + .5], [25.2, 25.2, 1.6, 1.6], color=I.LINE, lw=.7, solid_capstyle='round', zorder=3)
    for yy in [28.0, 23.2, 18.6, 14.0, 9.4, 3.6]:
        I.arrow(m, TX + TW_ + .5, yy, TX + TW_ + 2.1, yy, lw=.9, ms=7)
    # transformer blocks (no internal diagram: a labelled stack)
    BX, BW = TX + TW_ + 2.5, 9.0
    for k in range(5, -1, -1):
        I.rbox(m, BX + k * .45, 2.0 + k * .45, BW, 24.4, '#EEF2FA' if k else '#F7F9FD', ec='#8797B8', r=1.0, lw=.8, shadow_=(k == 5))
    cx_ = BX + BW / 2
    m.text(cx_, 24.6, 'Transformer\nblock ×6', ha='center', va='center', fontsize=6.4, fontweight='bold', linespacing=1.15)
    I.rbox(m, BX + 1.0, 15.6, BW - 2.0, 4.6, '#FFFFFF', ec='#8797B8', r=.5, shadow_=False)
    m.text(cx_, 18.7, 'Self-attention', ha='center', va='center', fontsize=5.6)
    m.text(cx_, 17.0, 'Four heads', ha='center', va='center', fontsize=5.2)
    I.arrow(m, cx_, 15.2, cx_, 13.4, lw=.8, ms=6)
    I.rbox(m, BX + 1.0, 9.4, BW - 2.0, 3.6, '#FFFFFF', ec='#8797B8', r=.5, shadow_=False)
    m.text(cx_, 11.2, 'Feed-forward', ha='center', va='center', fontsize=5.6)
    m.text(cx_, 5.9, 'Cells and the\nsummary token\nattend to one\nanother', ha='center', va='center', fontsize=5.2, linespacing=1.2)
    # patient summary -> SPECTRA score dial
    ox = BX + BW + 2.25 + 1.7
    I.arrow(m, BX + BW + 2.25 + .4, 16.0, ox - .3, 16.0)
    m.add_patch(FancyBboxPatch((ox, 14.0), TW_, 4.0, boxstyle='round,pad=0,rounding_size=.5', fc='#CDBCE8', ec=PUR_D, lw=.9, zorder=3))
    m.text(ox + TW_ / 2, 16.0, 'Patient\nsummary', ha='center', va='center', fontsize=5.2, linespacing=1.1, zorder=4)
    gx, gy, ro, wi = ox + TW_ + 4.4, 14.9, 2.9, 1.1
    I.arrow(m, ox + TW_ + .3, 16.0, gx - ro - .2, 16.0)
    cm = LinearSegmentedColormap.from_list('score', [CONTROL, '#ECEFF1', TEAL]); NSEG = 60
    for q in range(NSEG):
        m.add_patch(Wedge((gx, gy), ro, 180 - (q + 1) * 180 / NSEG, 180 - q * 180 / NSEG + .6, width=wi, fc=cm(q / (NSEG - 1)), ec='none', zorder=3))
    ang = np.deg2rad(180 * (1 - .9))
    m.plot([gx, gx + (ro - .25) * np.cos(ang)], [gy, gy + (ro - .25) * np.sin(ang)], color=INK, lw=1.3, solid_capstyle='round', zorder=4)
    m.add_patch(Circle((gx, gy), .32, fc=INK, ec='none', zorder=5))
    m.text(gx - ro + wi / 2, gy - .35, '0', ha='center', va='top', fontsize=5.2); m.text(gx + ro - wi / 2, gy - .35, '1', ha='center', va='top', fontsize=5.2)
    m.text(gx, gy + ro + .55, 'SPECTRA\nscore', ha='center', va='bottom', fontsize=6.2, fontweight='bold', linespacing=1.1)
    m.text(gx, gy - 1.6, 'P(cancer)', ha='center', va='top', fontsize=5.6)
    # attribution back to each cell
    m.plot([gx, gx, TX + TW_ / 2], [gy - 3.3, -.6, -.6], color=SPECTRA_D, lw=.9, ls=(0, (3, 2)), zorder=3)
    I.arrow(m, TX + TW_ / 2, -.6, TX + TW_ / 2, 1.2, color=SPECTRA_D, lw=.9, ms=7)
    m.text((gx + TX) / 2 + 2, -.2, 'Attribution back to each cell', ha='center', va='bottom', fontsize=5.8, bbox=dict(fc='white', ec='none', pad=.5))
    m.text(.1, 7.6, 'Attribution', fontsize=5.4, va='bottom')
    for q, c_ in enumerate([TEAL_L, TEAL_M, '#3F8E9B', TEAL]):
        m.add_patch(Rectangle((.2 + q * 1.25, 6.1), 1.05, 1.05, fc='white', ec=c_, lw=1.5, zorder=3))
    m.text(.1, 5.6, 'Low', fontsize=5.2, va='top'); m.text(.2 + 3 * 1.25 + 1.05, 5.6, 'High', fontsize=5.2, va='top', ha='right')

    # ---------------- c: cohort ----------------
    pg.letter(4.12, 2.39, 'c')
    c = pg.ax(5.02, 2.74, 1.55, 2.62, panel='c')
    led = pd.read_csv(sx.LEDGER); led = led[led.primary_cancer_vs_reference_eligible]
    tab = pd.crosstab(led.cancer_types, led.split); tab['n'] = tab.sum(axis=1)
    ctrl = tab.loc[['Control']]; tab = tab.drop('Control').sort_values('n', ascending=False); tab = pd.concat([tab, ctrl])
    for i, (t, r) in enumerate(tab.iterrows()):
        col = CONTROL if t == 'Control' else sx.CANCER_COL[t]
        c.barh(i, r.train, color=col, height=.72, lw=0); c.barh(i, r.test, left=r.train, color=col, alpha=.45, height=.72, lw=0)
        c.text(r.n + 15, i, f'{int(r.n)}', va='center', fontsize=5.8)
    c.set_yticks(range(len(tab))); c.set_yticklabels([sx.short(t) for t in tab.index], fontsize=5.9); c.set_ylim(len(tab) - .4, -.6)
    c.tick_params(axis='y', length=0); c.spines['left'].set_visible(False)
    c.set_xlim(0, 1000); c.set_xticks([0, 500, 1000]); c.set_xlabel('Patients')
    c.add_patch(Rectangle((480, 8.6), 50, .65, color='#555555')); c.text(560, 8.95, 'Training (1,793)', fontsize=5.8, va='center')
    c.add_patch(Rectangle((480, 9.7), 50, .65, color='#555555', alpha=.45)); c.text(560, 10.05, 'Test (757)', fontsize=5.8, va='center')

    sx.export(pg, 'figure1', 'Figure1')
    tab.to_csv(sx.ROOT / 'figure1/cohort_source.csv')


if __name__ == '__main__':
    build()
