"""Flat, BioRender-like vector icons that follow the original workflow panel
(tube -> centrifuge -> plated cells, antibodies + microscope, fWSI slide, segment & embed network, anomaly histogram)."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, Polygon, FancyArrowPatch, Wedge, Arc

OUT = '#5E6B75'      # soft outline
GREY = '#D9DEE3'; GREY2 = '#B9C1C8'; GREY3 = '#8E99A3'


def rb(ax, x, y, w, h, fc, ec=OUT, lw=.6, r=.4, z=2, alpha=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha))


def arrow(ax, x0, y0, x1, y1, lw=1.6, ms=10, color='#2E2E2E', curve=0, z=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color, zorder=z,
                                 connectionstyle=f'arc3,rad={curve}', shrinkA=0, shrinkB=0))


def tube(ax, x, y, w=2.2, h=11):
    rb(ax, x, y, w, h * .8, '#F3F6F8', r=w / 2, lw=.6)
    rb(ax, x + .18, y + .18, w - .36, h * .62, '#8E1F2B', ec='none', r=(w - .36) / 2, z=3)
    ax.add_patch(Rectangle((x + .45, y + 1.2), .3, h * .45, fc='white', alpha=.3, zorder=3.1))
    rb(ax, x - .15, y + h * .78, w + .3, h * .2, '#E8833A', ec='#B8652B', r=.35, z=4)
    for i in range(4):
        ax.plot([x + .25 + i * (w - .2) / 3.5] * 2, [y + h * .8, y + h * .96], color='#C96F2F', lw=.5, zorder=4.1)


def centrifuge(ax, x, y, w=9, h=7):
    rb(ax, x, y, w, h * .62, '#EEF1F4', r=.8)                                   # body
    rb(ax, x + w * .06, y + h * .05, w * .88, h * .1, GREY, ec='none', r=.3, z=2.1)  # base band
    ax.add_patch(Ellipse((x + w * .5, y + h * .62), w * .86, h * .22, fc=GREY2, ec=OUT, lw=.6, zorder=2.2))     # bowl
    ax.add_patch(Ellipse((x + w * .5, y + h * .64), w * .66, h * .14, fc='#5D7590', ec=OUT, lw=.5, zorder=2.3))  # rotor
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        ax.add_patch(Ellipse((x + w * .5 + w * .25 * np.cos(t), y + h * .64 + h * .05 * np.sin(t)), w * .06, h * .035, fc='#C9D5E2', ec='none', zorder=2.4))
    ax.add_patch(Circle((x + w * .5, y + h * .64), w * .035, fc='#2E3B48', zorder=2.5))
    lid = Polygon([[x + w * .1, y + h * .66], [x + w * .9, y + h * .66], [x + w * .85, y + h * 1.0], [x + w * .15, y + h * 1.0]], closed=True, fc='#F6F8FA', ec=OUT, lw=.6, zorder=1.9)
    ax.add_patch(lid)
    ax.add_patch(Ellipse((x + w * .5, y + h * .86), w * .42, h * .16, fc='#E3E8EE', ec=GREY3, lw=.5, zorder=2))
    ax.add_patch(Circle((x + w * .5, y + h * .86), w * .03, fc='#2E3B48', zorder=2.1))
    rb(ax, x + w * .55, y + h * .2, w * .3, h * .14, '#FFFFFF', ec=GREY3, r=.2, lw=.4, z=2.6)
    for i in range(3):
        ax.add_patch(Circle((x + w * (.6 + i * .08), y + h * .27), w * .018, fc=['#3E82C9', '#9AA3AA', '#9AA3AA'][i], zorder=2.7))


def slide(ax, x, y, w=10, h=2.4, drop=True, angle_skew=.9):
    """Slanted glass slide (perspective) with frosted end and cells plated over the whole clear area."""
    sk = angle_skew
    ax.add_patch(Polygon([[x, y], [x + w, y], [x + w + sk, y + h], [x + sk, y + h]], closed=True, fc='#C7D3E3', ec=OUT, lw=.6, zorder=2))
    ax.add_patch(Polygon([[x, y], [x + w * .3, y], [x + w * .3 + sk, y + h], [x + sk, y + h]], closed=True, fc='#9AABC3', ec=OUT, lw=.5, zorder=2.1))
    ax.add_patch(Polygon([[x, y - .45], [x + w, y - .45], [x + w, y], [x, y]], closed=True, fc='#A9B8CC', ec=OUT, lw=.5, zorder=1.9))
    if drop:   # cells are plated across the whole slide (a monolayer over the clear area), not in a droplet
        P = lambda u, v: [x + u * w + sk * v, y + v * h]
        ax.add_patch(Polygon([P(.335, .1), P(.975, .1), P(.975, .9), P(.335, .9)], closed=True, fc='#EDB9B6', ec='#C77F82', lw=.4, zorder=2.2))
        rng = np.random.default_rng(4)
        for _ in range(150):
            u, v = rng.uniform(.35, .96), rng.uniform(.16, .84)
            px, py = P(u, v); ax.add_patch(Circle((px, py), .06, fc='#B85C63', ec='none', zorder=2.3))
    return x + w * .62 + sk / 2, y + h / 2


def zoom_cells(ax, cx, cy, r, seed=2):
    ax.add_patch(Circle((cx, cy), r, fc='white', ec=OUT, lw=.7, zorder=3))
    rng = np.random.default_rng(seed); pts = []
    while len(pts) < 26:
        a = rng.uniform(0, 2 * np.pi); rr = np.sqrt(rng.uniform()) * (r - .9)
        p = (cx + rr * np.cos(a), cy + rr * np.sin(a))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > 1.25 for q in pts):
            pts.append(p)
    for i, (px, py) in enumerate(pts):
        big = i in (3, 11)
        s = .85 if big else .6
        ax.add_patch(Circle((px, py), s, fc='#F7D6BF', ec='#E2A47E', lw=.5, zorder=3.1))
        ax.add_patch(Circle((px + .05, py + .05), s * (.55 if big else .45), fc='#E9A57F', ec='none', zorder=3.2))


def antibody(ax, x, y, color, s=1.0):
    k = dict(color=GREY3, lw=2.0 * s, solid_capstyle='round', zorder=4)
    ax.plot([x, x], [y, y + .9 * s], **k); ax.plot([x, x - .55 * s], [y + .9 * s, y + 1.6 * s], **k); ax.plot([x, x + .55 * s], [y + .9 * s, y + 1.6 * s], **k)
    k2 = dict(color='#C7CDD3', lw=.9 * s, solid_capstyle='round', zorder=4.1)
    ax.plot([x - .2 * s, x - .6 * s], [y + 1.0 * s, y + 1.55 * s], **k2); ax.plot([x + .2 * s, x + .6 * s], [y + 1.0 * s, y + 1.55 * s], **k2)
    ax.add_patch(Circle((x, y - .1 * s), .36 * s, fc=color, ec='white', lw=.5, zorder=4.2))


def microscope(ax, x, y, s=1.0):
    """Upright fluorescence microscope: base, pillar, arm, binocular head, nosepiece/objectives, stage, camera, light source."""
    S = lambda v: v * s
    rb(ax, x, y, S(6.2), S(.9), '#E6EAEE', r=.3)                                        # base
    rb(ax, x + S(4.3), y + S(.9), S(1.2), S(6.6), '#D5DBE0', r=.45)                     # pillar
    rb(ax, x + S(1.4), y + S(6.9), S(4.1), S(1.5), '#EEF1F4', r=.45, z=3)              # arm / head
    for dx in (0, .55):                                                                   # binocular eyepieces
        ax.add_patch(Polygon([[x + S(1.7 + dx), y + S(8.3)], [x + S(2.15 + dx), y + S(8.3)], [x + S(1.45 + dx), y + S(10.0)], [x + S(1.0 + dx), y + S(10.0)]],
                             closed=True, fc='#DDE2E7', ec=OUT, lw=.5, zorder=3.1))
        rb(ax, x + S(.9 + dx), y + S(9.9), S(.55), S(.5), '#3B4650', ec='none', r=.1, z=3.2)
    rb(ax, x + S(3.6), y + S(8.4), S(1.3), S(1.4), '#2F3A44', ec='#222', r=.2, z=3)       # camera
    for k, dx in enumerate((-.35, .1, .55)):                                              # objectives
        rb(ax, x + S(2.6 + dx), y + S(5.7 - .15 * (k == 1)), S(.32), S(1.2), '#A9B4BD' if k != 1 else '#7D8A95', r=.08, z=3)
    rb(ax, x + S(2.2), y + S(6.6), S(1.6), S(.45), '#C3CBD2', r=.15, z=3.05)              # nosepiece
    rb(ax, x + S(1.0), y + S(4.5), S(3.6), S(.45), '#3D4750', ec='#2B333A', r=.12, z=3)   # stage
    rb(ax, x + S(1.4), y + S(4.95), S(2.6), S(.18), '#C7D3E3', ec=OUT, r=.05, lw=.4, z=3.1)  # slide on stage
    rb(ax, x + S(4.1), y + S(1.6), S(1.8), S(.9), '#C3CBD2', r=.2, z=3)                   # focus knob
    rb(ax, x + S(6.7), y, S(2.8), S(2.0), '#E6EAED', r=.3)                                # light source
    for i, c in enumerate(['#3E82C9', '#E34B4B', '#6BBF59']):
        ax.add_patch(Circle((x + S(7.3) + i * S(.65), y + S(.65)), S(.17), fc=c, zorder=3))


def fwsi_slide(ax, x, y, w=3.0, h=11.0, seed=7, n=90):
    rb(ax, x, y, w, h, '#F4F6F8', r=.2)
    rb(ax, x + .25, y + h * .72, w - .5, h * .25, '#E3E7EB', ec='none', r=.15, z=2.1)
    rb(ax, x + .25, y + .3, w - .5, h * .66, '#0B0D10', ec='none', r=.15, z=2.2)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        px, py = x + .45 + rng.uniform() * (w - .9), y + .55 + rng.uniform() * (h * .66 - .5)
        col = rng.choice(['#3DBF62', '#3DBF62', '#3DBF62', '#3E82C9', '#E34B4B'], p=[.3, .3, .26, .09, .05])
        ax.add_patch(Circle((px, py), .12, fc=col, ec='none', zorder=2.3))


def network(ax, x, y, w=7, h=10, node='#A6435B', node_out='#C4B9BC', box='#F1F3FA', box_ec='#C9CFE6', edge='#C9CFE6', bar_ec='#8F9BD0'):
    layers = [8, 5, 3]; xs = np.linspace(x, x + w * .72, len(layers)); pos = []
    for li, (xx, n) in enumerate(zip(xs, layers)):
        ys = np.linspace(y + h * (.5 - .45 * n / 8), y + h * (.5 + .45 * n / 8), n); pos.append([(xx, yy) for yy in ys])
        rb(ax, xx - .55, ys[0] - .6, 1.1, ys[-1] - ys[0] + 1.2, box, ec=box_ec, r=.25, lw=.5, z=2)
    for a_, b_ in zip(pos[:-1], pos[1:]):
        for p in a_:
            for q in b_:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=edge, lw=.35, zorder=2.1)
    for li, layer in enumerate(pos):
        for (px, py) in layer:
            ax.add_patch(Circle((px, py), .32, fc=node if li < 2 else node_out, ec='none', zorder=3))
    cx = x + w * .95
    rb(ax, cx - .45, y + h * .22, .9, h * .56, box, ec=bar_ec, r=.45, lw=.9, z=3)


def histogram(ax, x, y, w=9, h=11, bars=None, tail='#B03A2E', n_tail=2, bar='#4A7BE0', bar_ec='#2F5FBF',
              heights=(.55, 1.0, .82, .58, .38, .24, .15, .09, .055, .035)):
    """Anomaly-score histogram: most cells score low (left), the rare tail is on the right. The n_tail right-most bars and the
    arrow take the colour `tail`."""
    hs = np.array(heights) * h; n = len(hs)
    bw = w / n
    for i, hh in enumerate(hs):
        fc = tail if i >= n - n_tail else (bars[i] if bars else bar)
        ax.add_patch(Rectangle((x + i * bw, y), bw * .95, hh, fc=fc, ec='white' if bars else bar_ec, lw=.4, zorder=2))
    ax.plot([x - .4, x + w + .2], [y, y], color='#2E2E2E', lw=.8, zorder=3); ax.plot([x - .4, x - .4], [y, y + h * 1.05], color='#2E2E2E', lw=.8, zorder=3)
    xa = x + w - bw * n_tail / 2
    ax.add_patch(FancyArrowPatch((xa, y + h * .72), (xa, y + h * .16), arrowstyle='-|>', mutation_scale=10, lw=1.5, color=tail, zorder=4))


CELL_KINDS = {   # cytoplasm, outline, nucleus, elongation, lobe amplitude, lobe count, rotation (deg), nucleus scale
    'peach': ('#F7D6BF', '#E2A47E', '#E9A57F', 1.00, .04, 5, 0, .50),
    'rose': ('#F5C1C6', '#D98A93', '#E08A98', 1.45, .07, 3, 28, .42),
    'lavender': ('#DDD3EE', '#A893CF', '#B49BD6', 1.15, .14, 6, 75, .40),
    'amber': ('#FBE7B5', '#DDB04F', '#EBC56B', 1.15, .14, 6, 75, .40),
    'mint': ('#CFEBD6', '#86C29A', '#8FCBA3', 1.00, .03, 4, 0, .66),
    'sky': ('#CFE3F3', '#8DB5DA', '#93BDE0', 1.75, .06, 2, -35, .36),
}


def _blob(cx, cy, r, elong, amp, k, rot, n=160):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False); rad = r * (1 + amp * np.sin(k * t + .7) + amp * .5 * np.sin(2 * t))
    x, y = elong * rad * np.cos(t) / np.sqrt(elong), rad * np.sin(t) / np.sqrt(elong) * (1 if elong == 1 else .9)
    c, s = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    return np.c_[cx + c * x - s * y, cy + s * x + c * y]


def cartoon_cell(ax, cx, cy, r, kind='peach', z=3.2, outline=None):
    """Flat cartoon cell (same style as zoom_cells); outline = colour of a dashed segmentation contour, if any."""
    fc, ec, nuc, elong, amp, k, rot, ns = CELL_KINDS[kind]
    ax.add_patch(Polygon(_blob(cx, cy, r, elong, amp, k, rot), closed=True, fc=fc, ec=ec, lw=.6, zorder=z, joinstyle='round'))
    ax.add_patch(Polygon(_blob(cx + .06 * r, cy + .05 * r, r * ns, min(elong, 1.25), amp * .5, 3, rot), closed=True, fc=nuc, ec='none', zorder=z + .1))
    if outline:
        ax.add_patch(Polygon(_blob(cx, cy, r * 1.17, elong, amp, k, rot), closed=True, fc='none', ec=outline, lw=.7, ls=(0, (2, 1.4)), zorder=z + .2))
