from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'runs/paper_regeneration_20261109_v1a'
DATA=Path(os.environ.get('SPECTRA_DATA_ROOT','data')).resolve()
INTAKE=REPO/'analyses/cohort_revision_20261109'
def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def local(path):
    p=Path(path).resolve(strict=True)
    if not p.is_relative_to(DATA):raise ValueError(f'Required local NVMe backing: {p}')
    return p
