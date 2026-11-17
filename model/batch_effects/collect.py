"""Read-only collection of acquisition metadata (scanner instrument, scan date, exposures, experiment ID) from
bzScanner/proc/slideinfo.txt in the raw image store for every slide in the revised slide CSV. Nothing is written to the mount."""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'analyses/batch_effects_20261111'; OUT.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(os.environ.get('SPECTRA_SLIDE_TABLE', 'data/slides.csv'), dtype=str)

def one(sid):
    hits = []
    for root in (Path(os.environ.get('SPECTRA_10X_ROOT', 'raw/10x')),):
        t = root / f'tubeID_{sid[:5]}'
        if t.is_dir():
            for e in t.iterdir():
                f = e / f'slideID_{sid}/bzScanner/proc/slideinfo.txt'
                if f.is_file():
                    hits.append((e.name, f))
    if not hits:
        return dict(slide_id=sid, found=False)
    expts = sorted({h[0] for h in hits})
    kv = {}
    for line in hits[0][1].read_text(errors='replace').splitlines():
        if '=' in line:
            k, v = line.split('=', 1); kv[k.strip()] = v.strip()
    ex = {ch: kv.get(ch, '').split(',') for ch in ['DAPI', 'FITC', 'TRITC', 'CY5']}
    return dict(slide_id=sid, found=True, expt_id=expts[0], n_expt_dirs=len(expts), instrument=kv.get('INSTRUMENT'), scan_date=kv.get('DATE'), org=kv.get('ORG'),
                **{f'{ch}_exposure': (v[1] if len(v) > 1 else None) for ch, v in ex.items()})

with ThreadPoolExecutor(16) as pool:
    rows = list(pool.map(one, s.slide_id.unique()))
m = pd.DataFrame(rows)
m['scan_date'] = pd.to_datetime(m.scan_date, format='%a %b %d %H:%M:%S %Y', errors='coerce')
m.to_csv(OUT / 'slide_acquisition_metadata_private.csv', index=False)
print(m.found.value_counts(), m.instrument.value_counts(dropna=False), m.scan_date.describe(), m.n_expt_dirs.value_counts(), sep='\n')
