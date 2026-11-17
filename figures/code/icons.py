"""Original vector illustrations (matplotlib patches) for the SPECTRA workflow and model.
Drawn in data units on an equal-aspect axes; no external artwork."""
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, Polygon, FancyArrowPatch, PathPatch, Wedge
from matplotlib.path import Path

LINE = '#3A4750'
GLASS = '#EAF2F6'
SHADOW = '#000000'


def shadow(ax, patch_fn, dx=.25, dy=-.25, alpha=.10, **kw):
    p = patch_fn(dx, dy); p.set_facecolor(SHADOW); p.set_edgecolor('none'); p.set_alpha(alpha); p.set_zorder(kw.get('z', 1) - .5); ax.add_patch(p)


def rbox(ax, x, y, w, h, fc, ec=LINE, lw=.8, r=.6, z=2, alpha=1, shadow_=True):
    if shadow_:
        shadow(ax, lambda dx, dy: FancyBboxPatch((x + dx, y + dy), w, h, boxstyle=f'round,pad=0,rounding_size={r}'), z=z)
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
    ax.add_patch(p); return p


def arrow(ax, x0, y0, x1, y1, color=LINE, lw=1.1, ms=9, ls='-', z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color, linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def blood_tube(ax, cx, y0, h=9.0, w=2.6):
    """Collection tube with cap, label band and blood fill."""
    body = lambda dx, dy: FancyBboxPatch((cx - w / 2 + dx, y0 + dy), w, h * .82, boxstyle=f'round,pad=0,rounding_size={w / 2}')
    shadow(ax, body, z=2)
    ax.add_patch(FancyBboxPatch((cx - w / 2, y0), w, h * .82, boxstyle=f'round,pad=0,rounding_size={w / 2}', fc=GLASS, ec=LINE, lw=.8, zorder=2))
    # blood (two-tone for depth)
    ax.add_patch(FancyBboxPatch((cx - w / 2 + .25, y0 + .25), w - .5, h * .5, boxstyle=f'round,pad=0,rounding_size={(w - .5) / 2}', fc='#9E1B32', ec='none', zorder=3))
    ax.add_patch(Rectangle((cx - w / 2 + .25, y0 + h * .38), w - .5, h * .14, fc='#B8283F', ec='none', zorder=3.1))
    ax.add_patch(Rectangle((cx - w / 2 + .55, y0 + 1.2), .35, h * .36, fc='white', ec='none', alpha=.35, zorder=3.2))  # glint
    # label band
    ax.add_patch(Rectangle((cx - w / 2, y0 + h * .56), w, h * .14, fc='white', ec=LINE, lw=.6, zorder=3.3))
    for i in range(4):
        ax.plot([cx - w / 2 + .4, cx - w / 2 + .4 + (w - .8) * (1 - .2 * (i % 2))], [y0 + h * (.59 + .025 * i)] * 2, color='#9AA5AD', lw=.4, zorder=3.4)
    # cap
    cap = lambda dx, dy: FancyBboxPatch((cx - w * .62 + dx, y0 + h * .80 + dy), w * 1.24, h * .2, boxstyle='round,pad=0,rounding_size=.35')
    shadow(ax, cap, z=4)
    ax.add_patch(FancyBboxPatch((cx - w * .62, y0 + h * .80), w * 1.24, h * .2, boxstyle='round,pad=0,rounding_size=.35', fc='#5B6F86', ec=LINE, lw=.8, zorder=4))
    for i in range(5):
        xx = cx - w * .5 + i * w / 4
        ax.plot([xx, xx], [y0 + h * .83, y0 + h * .97], color='#7F93A8', lw=.6, zorder=4.1)


def glass_slide(ax, x, y, w=9.0, h=3.6, cells=True, seed=1, label_end=True):
    rbox(ax, x, y, w, h, '#F4F8FA', r=.25)
    if label_end:
        ax.add_patch(Rectangle((x + .05, y + .05), w * .22, h - .1, fc='#D9DEE2', ec='none', zorder=2.2))
        for i in range(3):
            ax.plot([x + .35, x + w * .19], [y + h * (.35 + .15 * i)] * 2, color='#AEB7BE', lw=.5, zorder=2.3)
    cx, cy, r = x + w * .62, y + h / 2, h * .38
    ax.add_patch(Circle((cx, cy), r, fc='#FBE9EA', ec='#D9A3AA', lw=.6, zorder=2.4))
    if cells:
        rng = np.random.default_rng(seed); n = 240
        ang = rng.uniform(0, 2 * np.pi, n); rad = r * np.sqrt(rng.uniform(0, .92, n))
        col = np.where(rng.uniform(size=n) < .05, '#C0392B', '#5B79C9')
        ax.scatter(cx + rad * np.cos(ang), cy + rad * np.sin(ang), s=1.0, c=col, lw=0, zorder=2.5)
    return cx, cy, r


def antibody(ax, x, y, color, s=1.0, z=4):
    """Y-shaped antibody with a fluorophore."""
    k = dict(color='#7C8790', lw=1.6 * s, solid_capstyle='round', zorder=z)
    ax.plot([x, x], [y, y + .9 * s], **k)
    ax.plot([x, x - .55 * s], [y + .9 * s, y + 1.55 * s], **k)
    ax.plot([x, x + .55 * s], [y + .9 * s, y + 1.55 * s], **k)
    ax.add_patch(Circle((x, y - .05 * s), .32 * s, fc=color, ec='white', lw=.4, zorder=z + .1))


def scanner(ax, x, y, w=6.0, h=5.5):
    """Fluorescence whole-slide scanner: housing, objective, excitation light, stage with slide."""
    rbox(ax, x, y, w, h * .25, '#DCE3E8', r=.3)                         # base / stage
    rbox(ax, x + w * .72, y + h * .25, w * .2, h * .62, '#C9D3DA', r=.3)  # column
    rbox(ax, x + w * .18, y + h * .72, w * .78, h * .26, '#E4EAEE', r=.4)  # head
    obj = Polygon([[x + w * .38, y + h * .72], [x + w * .58, y + h * .72], [x + w * .54, y + h * .55], [x + w * .42, y + h * .55]], closed=True, fc='#A9B5BE', ec=LINE, lw=.8, zorder=3)
    ax.add_patch(obj)
    ax.add_patch(Polygon([[x + w * .43, y + h * .55], [x + w * .53, y + h * .55], [x + w * .58, y + h * .27], [x + w * .38, y + h * .27]], closed=True,
                         fc='#7FD3C4', ec='none', alpha=.45, zorder=2.8))   # excitation cone
    ax.add_patch(Rectangle((x + w * .12, y + h * .25), w * .58, h * .04, fc='#F4F8FA', ec=LINE, lw=.6, zorder=3))  # slide on stage
    for i, c in enumerate(['#2B5BD7', '#C0392B', '#9AA3A8', '#2E9A4A']):
        ax.add_patch(Circle((x + w * .28 + i * w * .1, y + h * .85), w * .03, fc=c, ec='none', zorder=3.2))


def tiles(ax, x, y, w, h, images, ncol=4):
    """Mosaic of real image tiles (whole-slide tiling)."""
    nrow = int(np.ceil(len(images) / ncol)); tw = w / ncol; th = h / nrow
    for i, im in enumerate(images):
        r, c = divmod(i, ncol)
        ax.imshow(im, extent=(x + c * tw, x + (c + 1) * tw, y + h - (r + 1) * th, y + h - r * th), zorder=3, interpolation='lanczos')
    for r in range(nrow + 1):
        ax.plot([x, x + w], [y + r * th] * 2, color='white', lw=.6, zorder=3.5)
    for c in range(ncol + 1):
        ax.plot([x + c * tw] * 2, [y, y + h], color='white', lw=.6, zorder=3.5)
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=LINE, lw=.8, zorder=3.6))


def embedding_bar(ax, x, y, h, n=12, seed=3, w=.9):
    rng = np.random.default_rng(seed); v = rng.normal(size=n)
    cmap = __import__('matplotlib').colormaps['RdBu_r']
    for i, val in enumerate(v):
        ax.add_patch(Rectangle((x, y + i * h / n), w, h / n, fc=cmap(.5 + .35 * np.tanh(val)), ec='white', lw=.3, zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, fc='none', ec=LINE, lw=.7, zorder=3.1))


def tail_histogram(ax, x, y, w, h, color='#0E6E7E'):
    """Anomaly score distribution with the rare-event tail highlighted."""
    xs = np.linspace(0, 1, 18); hs = np.exp(-((xs - .16) / .2) ** 2) * .95 + .05 + .10 * (xs > .74)
    bw = w / len(xs)
    for i, (xx, hh) in enumerate(zip(xs, hs)):
        tail = i >= 14
        ax.add_patch(Rectangle((x + i * bw, y), bw * .86, h * hh, fc=color if tail else '#C9D3DA', ec='none', zorder=3))
    ax.plot([x, x + w], [y, y], color=LINE, lw=.8, zorder=3.2)
    ax.plot([x, x], [y, y + h], color=LINE, lw=.8, zorder=3.2)
