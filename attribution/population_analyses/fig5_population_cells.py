"""Figure 5c/d image strips (group-meeting revision 2026-11-12): instead of one thumbnail per cluster, show several cells per
cluster whose share of the evidence changed with adaptation, from different test patients of that task.

Shown clusters per task: the six largest absolute mean changes in positive-evidence share (frozen probe
minus pan-cancer model, equal-weight mean of cancer and control group means) among clusters where both groups move in the same
direction. Candidate cells: within each test patient, cells of that cluster in the top 2% by relevance under the head that
uses them more (probe for gains, pan-cancer model for losses); one cell per patient, ordered by within-patient rank, cancers and
controls interleaved. Each candidate is imaged as a 40x re-image when one exists (Figure 3 centring rule), else a 10x crop.
Candidates are written to a contact sheet; the final cells are picked by eye for focus and centring (recorded in
fig5_population_cells_selected_private.csv). Post hoc and descriptive."""
import os
from pathlib import Path
import sys, json
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'validation/single_cell_cnv'))
sys.path.insert(0, str(REPO / 'model/pipeline'))
import figure_data as FD  # noqa: E402
from common import read  # noqa: E402
RUN = REPO / 'runs/specific_adaptation_20261111/attribution'
OUT = REPO / 'analyses/population_analyses/fig5_cells'; OUT.mkdir(parents=True, exist_ok=True)
SH = REPO / 'analyses/specific_adaptation_20261111/attribution'
NUM = pd.read_csv(REPO / 'analyses/figure3_locked6_attribution_20261111/leiden/cluster_display_numbers.csv', index_col=0).number
N_CAND = 12


def shown(task, n=6):
    sh = pd.read_csv(SH / f'{task}_cluster_share_shift.csv').pivot_table(index='cluster', columns='label', values='delta_probe')
    same = sh[np.sign(sh[0]) == np.sign(sh[1])]; m = same.mean(axis=1); top = m.abs().sort_values(ascending=False).index[:n]
    return [k for k in top if m[k] > 0], [k for k in top if m[k] < 0], m


def candidates():
    rec = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
    rows = []
    for task in ['lung', 'breast']:
        gain, loss, m = shown(task)
        c = pd.read_parquet(RUN / f'{task}_cells_clustered_private.parquet')
        for k in gain + loss:
            h = 'probe' if k in gain else 'source'
            c['rk'] = c.groupby('donor_id')[h].rank(ascending=False, pct=True)
            g = c[(c.cluster == k) & (c.rk <= .02) & (c[h] > 0)].sort_values('rk').drop_duplicates('donor_id')
            g = pd.concat([g[g.label == 1].head(N_CAND), g[g.label == 0].head(N_CAND // 2)]).sort_values('rk')
            rows.append(g.assign(task=task, display=NUM[k], direction='gain' if k in gain else 'loss', head=h, change_pp=m[k] * 100))
    cand = pd.concat(rows, ignore_index=True)
    meta = []
    for sid, g in cand.groupby('slide_id'):
        src = pd.read_parquet(rec[sid]['source'], columns=['frame_id', 'x', 'y', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']).iloc[g.source_row.to_numpy()]
        meta.append(pd.concat([g.reset_index(drop=True), src.reset_index(drop=True)], axis=1))
    cand = pd.concat(meta, ignore_index=True)
    ims, src_img = [], []
    for r in cand.itertuples():
        im = FD.forty_centred(r.slide_id, int(r.frame_id), int(r.x), int(r.y), [r.DAPI_mean, r.TRITC_mean, r.CY5_mean, r.FITC_mean])
        if im is not None:
            ims.append(im.astype(np.float32)); src_img.append('40x'); continue
        t = FD.tenx(r.slide_id, int(r.frame_id), int(r.x), int(r.y), hw=20)
        if t is None:
            ims.append(np.zeros((128, 128, 3), np.float32)); src_img.append('none'); continue
        from PIL import Image as PI
        ims.append(np.asarray(PI.fromarray((t * 255).astype(np.uint8)).resize((128, 128), PI.LANCZOS), np.float32) / 255); src_img.append('10x')
    cand['image_source'] = src_img; cand['image_index'] = np.arange(len(cand))
    np.savez_compressed(OUT / 'candidates_private.npz', images=np.stack(ims))
    cand.drop(columns=['rk'], errors='ignore').to_csv(OUT / 'candidates_private.csv', index=False)
    print(cand.groupby(['task', 'display', 'direction']).image_source.value_counts().unstack(fill_value=0))


def sheet():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cand = pd.read_csv(OUT / 'candidates_private.csv', dtype={'slide_id': str}); ims = np.load(OUT / 'candidates_private.npz')['images']
    for task, t in cand.groupby('task', sort=False):
        groups = list(t.groupby(['direction', 'display'], sort=False)); ncol = max(len(g) for _, g in groups)
        fig, axs = plt.subplots(len(groups), ncol, figsize=(ncol * 1.3, len(groups) * 1.45))
        for i, ((dirn, disp), g) in enumerate(groups):
            for j in range(ncol):
                a = axs[i, j]; a.axis('off')
                if j < len(g):
                    r = g.iloc[j]; a.imshow(ims[int(r.image_index)])
                    a.set_title(f'#{int(r.image_index)} {"C" if r.label else "N"} {r.image_source}', fontsize=7, color='k' if r.label else '#D9822B')
            axs[i, 0].text(-.15, .5, f'{dirn} {disp}', transform=axs[i, 0].transAxes, ha='right', va='center', fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / f'contact_{task}_private.png', dpi=110); plt.close(fig)


def index40():
    """Every 40x re-image (four channels) on the task patients' slides, matched to any model-input cell (same frame, unique
    cell within +/-10 px), with its cluster and within-patient relevance ranks under each head."""
    import re
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    rec = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
    PAT = re.compile(r'(\d+)-(\d+)-(-?\d+)-(-?\d+)-(DAPI|TRITC|FITC|CY5)\.tif', re.I)
    out = []
    for task in ['lung', 'breast']:
        c = pd.read_parquet(RUN / f'{task}_cells_clustered_private.parquet')
        for h in ['source', 'probe']:
            c[f'{h}_rk'] = c.groupby('donor_id')[h].rank(ascending=False, pct=True)

        def scan(sid):
            p = Path(os.environ.get('SPECTRA_40X_ROOT', 'raw/40x')) / sid / 'ClientBin/40x'
            if not p.is_dir():
                return sid, []
            f = defaultdict(set)
            for q in p.iterdir():
                m = PAT.fullmatch(q.name)
                if m:
                    f[tuple(map(int, m.groups()[:4]))].add(m.group(5).upper())
            return sid, [k for k, v in f.items() if len(v) == 4]
        with ThreadPoolExecutor(16) as ex:
            acq = dict(ex.map(scan, c.slide_id.unique()))
        for sid, hits in acq.items():
            if not hits:
                continue
            g = c[c.slide_id == sid]
            src = pd.read_parquet(rec[sid]['source'], columns=['frame_id', 'x', 'y', 'DAPI_mean', 'TRITC_mean', 'CY5_mean', 'FITC_mean']).iloc[g.source_row.to_numpy()]
            g = pd.concat([g.reset_index(drop=True), src.reset_index(drop=True)], axis=1)
            for h in hits:
                m = g[(g.frame_id == h[1]) & ((g.x.astype(int) - h[2]).abs() <= 10) & ((g.y.astype(int) - h[3]).abs() <= 10)]
                if len(m) == 1:
                    out.append(m.assign(task=task, acquisition='-'.join(map(str, h))))
    t = pd.concat(out, ignore_index=True); t.to_csv(OUT / 'all_40x_task_cells_private.csv', index=False)
    t['display'] = t.cluster.map(NUM)
    print(t.groupby(['task', 'display', 'label']).size().unstack(fill_value=0).to_string())


if __name__ == '__main__' and sys.argv[1] in ('candidates', 'sheet', 'index40'):
    {'candidates': candidates, 'sheet': sheet, 'index40': index40}[sys.argv[1]]()
    pass


def candidates40():
    """40x-only candidates for the shown clusters and a re-imaging list for gaps."""
    t = pd.read_csv(OUT / 'all_40x_task_cells_private.csv', dtype={'slide_id': str})
    rows, todo = [], []
    for task in ['lung', 'breast']:
        gain, loss, m = shown(task)
        c = pd.read_parquet(RUN / f'{task}_cells_clustered_private.parquet')
        for k in gain + loss:
            h = 'probe' if k in gain else 'source'
            g = t[(t.task == task) & (t.cluster == k) & (t[h] > 0)].sort_values(f'{h}_rk').drop_duplicates('donor_id')
            for lab in (1, 0):
                gg = g[g.label == lab].head(8)
                rows.append(gg.assign(display=NUM[k], direction='gain' if k in gain else 'loss', head=h))
                if len(gg) < 4:  # re-imaging list: best cells of this cluster and group without a 40x image, one per patient
                    cc = c[(c.cluster == k) & (c.label == lab) & (c[h] > 0)].copy()
                    cc['rk'] = c[c.label == lab].groupby('donor_id')[h].rank(ascending=False, pct=True).reindex(cc.index)
                    cc = cc[~cc.set_index(['slide_id', 'source_row']).index.isin(t.set_index(['slide_id', 'source_row']).index)]
                    cc = cc.sort_values('rk').drop_duplicates('donor_id').head(8)
                    todo.append(cc.assign(task=task, display=NUM[k], group='cancer' if lab else 'control', head=h, have_40x=len(gg)))
    cand = pd.concat(rows, ignore_index=True)
    ims = [FD.forty_centred(r.slide_id, int(r.frame_id), int(r.x), int(r.y), [r.DAPI_mean, r.TRITC_mean, r.CY5_mean, r.FITC_mean]) for r in cand.itertuples()]
    cand['image_source'] = ['40x' if i is not None else 'none' for i in ims]; cand['image_index'] = np.arange(len(cand))
    np.savez_compressed(OUT / 'candidates_private.npz', images=np.stack([i if i is not None else np.zeros((128, 128, 3)) for i in ims]).astype(np.float32))
    cand.to_csv(OUT / 'candidates_private.csv', index=False)
    rec = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
    td = pd.concat(todo, ignore_index=True); parts = []
    for sid, g in td.groupby('slide_id'):
        src = pd.read_parquet(rec[sid]['source'], columns=['frame_id', 'cell_id', 'x', 'y']).iloc[g.source_row.to_numpy()]
        parts.append(pd.concat([g.reset_index(drop=True), src.reset_index(drop=True)], axis=1))
    td = pd.concat(parts).sort_values(['task', 'display', 'group', 'rk'])
    td[['task', 'display', 'group', 'have_40x', 'donor_id', 'slide_id', 'frame_id', 'cell_id', 'x', 'y', 'head', 'rk']].rename(columns={'rk': 'within_patient_relevance_rank_pct'}) \
        .to_csv(REPO / 'analyses/population_analyses/fig5_40x_reimaging_list_private.csv', index=False)
    print(cand.groupby(['task', 'display', 'label']).size().unstack(fill_value=0)); print(td.groupby(['task', 'display', 'group']).size())


if __name__ == '__main__' and sys.argv[1] == 'candidates40':
    candidates40(); sheet()


def fixed_field():
    """Re-crop the selected Fig. 5 cells at one fixed 40x field (half-width 72 px x 0.18 um/px, a 26 um field;
    ) so a single scale bar applies to every tile; centring rule unchanged."""
    cand = pd.read_csv(OUT / 'candidates_private.csv', dtype={'slide_id': str}).set_index('image_index')
    sel = pd.read_csv(OUT / 'selected_private.csv'); sel = sel[sel.image_index >= 0]; out = {}
    for i in sel.image_index:
        r = cand.loc[i]
        out[str(i)] = FD.forty_centred(r.slide_id, int(r.frame_id), int(r.x), int(r.y), [r.DAPI_mean, r.TRITC_mean, r.CY5_mean, r.FITC_mean], size=144, fixed_half=72).astype(np.float32)
    np.savez_compressed(OUT / "selected_fixed_field_private.npz", um_per_px_display=2 * 72 * 0.18 / 144, **out)
    print('re-cropped', len(out))


if __name__ == '__main__' and sys.argv[1] == 'fixed':
    fixed_field()


def updown(n=8):
    """Fig. 5e,f: individual 40x-imaged test cells whose share of their patient's positive evidence rises or
    falls most after adaptation (refitted decision layer minus pan-cancer model), one cell per patient; no cluster selection."""
    t = pd.read_csv(OUT / 'all_40x_task_cells_private.csv', dtype={'slide_id': str})
    rows, out = [], {}
    for task in ['lung', 'breast']:
        c = pd.read_parquet(RUN / f'{task}_cells_clustered_private.parquet', columns=['donor_id', 'slide_id', 'source_row', 'source', 'probe'])
        for h in ['source', 'probe']:
            c[f'{h}_share'] = c[h].clip(lower=0) / c[h].clip(lower=0).groupby(c.donor_id).transform('sum')
        c['delta_pp'] = (c.probe_share - c.source_share) * 100
        g = t[t.task == task].drop_duplicates(['slide_id', 'source_row']).merge(c[['slide_id', 'source_row', 'source_share', 'probe_share', 'delta_pp']], on=['slide_id', 'source_row'], how='left', validate='one_to_one')
        sh = pd.read_csv(REPO / f'analyses/specific_adaptation_20261111/attribution/{task}_cluster_share_shift.csv')
        pv = sh.pivot_table(index='cluster', columns='label', values='delta_probe').fillna(0)
        cdir = pv.mean(axis=1) * 100   # cluster-level change, equal-weight mean of cancers and controls (percentage points)
        g['cluster_dir'] = g.cluster.map(cdir).fillna(0)
        for direction, asc in [('up', False), ('down', True)]:
            s = g.sort_values('delta_pp', ascending=asc)
            # exclude cells whose cluster clearly moves the opposite way (>= 0.5 points, as in Fig. 5c,d)
            s = s[(s.delta_pp > 0) & (s.cluster_dir > -.5)] if direction == 'up' else s[(s.delta_pp < 0) & (s.cluster_dir < .5)]
            got, seen = [], set()
            for r in s.itertuples():   # next cell when no 40x object can be centred; one cell per patient
                if r.donor_id in seen or len(got) == n:
                    continue
                im = FD.forty_centred(r.slide_id, int(r.frame_id), int(r.x), int(r.y), [r.DAPI_mean, r.TRITC_mean, r.CY5_mean, r.FITC_mean], size=144, fixed_half=72)
                if im is None:
                    continue
                k = len(out); out[str(k)] = im.astype(np.float32); seen.add(r.donor_id)
                got.append(pd.Series(r._asdict()).drop('Index').to_dict() | dict(direction=direction, tile=k))
            rows.append(pd.DataFrame(got))
    sel = pd.concat(rows, ignore_index=True)
    sel.to_csv(OUT / 'updown_private.csv', index=False)
    np.savez_compressed(OUT / 'updown_fixed_field_private.npz', um_per_px_display=2 * 72 * 0.18 / 144, **out)
    print(sel.groupby(['task', 'direction', 'label']).delta_pp.agg(['size', 'min', 'max']).round(2).to_string())


if __name__ == '__main__' and sys.argv[1] == 'updown':
    updown()
