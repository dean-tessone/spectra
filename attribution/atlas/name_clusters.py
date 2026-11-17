"""Rule-based descriptive names from median marker intensities (16-bit means; not cell identities)."""
from pathlib import Path
import pandas as pd
L = Path(__file__).resolve().parents[2] / 'analyses/figure3_locked6_attribution_20261111/leiden'
t = pd.read_csv(L / 'cluster_table.csv', index_col=0)
def name(r):
    parts = []
    ck = 'CK+' if r.CK > 15000 else ('CK dim' if r.CK > 9000 else None)
    cd = 'CD45/31 high' if r.CD45_CD31 > 35000 else ('CD45/31+' if r.CD45_CD31 > 15000 else ('CD45/31 dim' if r.CD45_CD31 > 8000 else None))
    if ck: parts.append(ck)
    if cd: parts.append(cd)
    if ck and not cd: parts.append('CD45/31−')
    if r.VIM > 10000: parts.append('VIM+')
    if r.DAPI > 20000: parts.append('DAPI bright')
    if not parts: parts.append('marker-low')
    if r.DAPI < 6000: parts.append('DAPI dim')
    parts.append('large' if r.area > 300 else ('small' if r.area < 140 else ''))
    return ', '.join(p for p in parts if p)
t['name'] = t.apply(name, axis=1)
t[['name']].to_csv(L / 'cluster_names.csv')
print(t[['name', 'cells', 'CK', 'CD45_CD31', 'VIM', 'DAPI', 'area']].to_string())
