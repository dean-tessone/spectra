"""Match mass-90 attributed cells to 40x re-imaged acquisitions (read-only), original Figure 3 rule:
same slide and frame, filename order cell-frame-x-y-channel, |dx|,|dy|<=10 px, exactly one candidate,
all four channels, acquisition not assigned to more than one cell."""
import os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import re, json
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
R = REPO / 'runs/figure3_locked6_attribution_20261111'
c = pd.read_parquet(R / 'all_cells.parquet', columns=['donor_id', 'slide_id', 'source_row', 'frame_id', 'x', 'y', 'mass90', 'label', 'cancer_types', 'called_at_90spec'])
c = c[c.mass90]
PAT = re.compile(r'(\d+)-(\d+)-(-?\d+)-(-?\d+)-(DAPI|TRITC|FITC|CY5)\.tif', re.I)
def scan(slide):
    p = Path(os.environ.get('SPECTRA_40X_ROOT', 'raw/40x')) / slide / 'ClientBin/40x'
    if not p.is_dir(): return slide, None
    files = defaultdict(set)
    for f in p.iterdir():
        m = PAT.fullmatch(f.name)
        if m: files[tuple(map(int, m.groups()[:4]))].add(m.group(5).upper())
    return slide, files
rec = []; inv = {}
with ThreadPoolExecutor(8) as pool:
    for slide, files in pool.map(scan, sorted(c.slide_id.unique())):
        inv[slide] = None if files is None else len(files)
        if not files: continue
        byframe = defaultdict(list)
        for (a, b, x, y), ch in files.items(): byframe[b].append((a, b, x, y, len(ch)))
        for r in c[c.slide_id == slide].itertuples():
            for a, b, x, y, n in byframe.get(int(r.frame_id), []):
                if abs(x - r.x) <= 10 and abs(y - r.y) <= 10:
                    rec.append(dict(slide_id=slide, source_row=int(r.source_row), acquisition=f'{a}-{b}-{x}-{y}', dx=x - r.x, dy=y - r.y, channels=n))
r = pd.DataFrame(rec)
r['candidates'] = r.groupby(['slide_id', 'source_row']).acquisition.transform('size')
r['eligible'] = r.candidates.eq(1) & r.channels.eq(4) & ~r.duplicated(['slide_id', 'acquisition'], keep=False)
out = REPO / 'analyses/figure3_locked6_attribution_20261111'; r.to_csv(out / '40x_matching_audit_private.csv', index=False)
json.dump(dict(slides=len(inv), slides_with_40x=sum(1 for v in inv.values() if v), matches=len(r), eligible=int(r.eligible.sum())), open(out / '40x_index_receipt.json', 'w'), indent=1)
print(open(out / '40x_index_receipt.json').read())
