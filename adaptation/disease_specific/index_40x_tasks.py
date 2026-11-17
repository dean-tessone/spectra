"""40x re-image availability for the top-30 cells (pan-cancer and probe heads) of every test task patient;
same matching rule as Figure 3 (same frame, unique acquisition within +/-10 px, four channels)."""
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import re, sys
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'model/pipeline'))
from common import read
receipt = read(REPO / 'runs/paper_regeneration_20261109_v1a/data/cache_receipt.json')['slides']
A = REPO / 'runs/specific_adaptation_20261111/attribution'
rows = []
for task in ['lung', 'breast']:
    t = pd.read_parquet(A / f'{task}_cells_private.parquet')
    for h in ['source', 'probe']:
        rows.append(t.sort_values(h, ascending=False).groupby('donor_id').head(30).assign(task=task))
sel = pd.concat(rows).drop_duplicates(['slide_id', 'source_row'])
PAT = re.compile(r'(\d+)-(\d+)-(-?\d+)-(-?\d+)-(DAPI|TRITC|FITC|CY5)\.tif', re.I)
def scan(sid):
    p = Path(os.environ.get('SPECTRA_40X_ROOT', 'raw/40x')) / sid / 'ClientBin/40x'
    if not p.is_dir(): return sid, None
    f = defaultdict(set)
    for q in p.iterdir():
        m = PAT.fullmatch(q.name)
        if m: f[tuple(map(int, m.groups()[:4]))].add(m.group(5).upper())
    return sid, f
out = []
with ThreadPoolExecutor(8) as pool:
    for sid, files in pool.map(scan, sorted(sel.slide_id.unique())):
        if not files: continue
        g = sel[sel.slide_id == sid]
        src = pd.read_parquet(receipt[sid]['source'], columns=['frame_id', 'x', 'y']).iloc[g.source_row.to_numpy()]
        byf = defaultdict(list)
        for (a, b, x, y), ch in files.items(): byf[b].append((a, b, x, y, len(ch)))
        for (_, r), (_, s) in zip(g.iterrows(), src.iterrows()):
            for a, b, x, y, n in byf.get(int(s.frame_id), []):
                if abs(x - s.x) <= 10 and abs(y - s.y) <= 10:
                    out.append(dict(task=r.task, donor_id=r.donor_id, label=r.label, slide_id=sid, source_row=int(r.source_row), acquisition=f'{a}-{b}-{x}-{y}', channels=n,
                                    source=r.source, probe=r.probe))
o = pd.DataFrame(out)
if len(o):
    o['candidates'] = o.groupby(['slide_id', 'source_row']).acquisition.transform('size'); o['eligible'] = o.candidates.eq(1) & o.channels.eq(4)
o.to_csv(REPO / 'analyses/specific_adaptation_20261111/attribution/40x_task_matches_private.csv', index=False)
print(len(o)); print(o[o.eligible].groupby(['task', 'label', 'donor_id']).size().sort_values(ascending=False).head(20) if len(o) else 'none')
