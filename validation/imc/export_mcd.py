"""Export IMC acquisitions from .mcd (read-only) to local npz for slides without per-ROI text exports.
Run with an environment providing readimc. Output: runs/imc_linkage_20261111/mcd_cache/
{slide}_{description}_{acq}.npz with image (C,H,W float32) and channel labels."""
import os
import sys, re
from pathlib import Path
import numpy as np
from readimc import MCDFile
IMC = Path(os.environ.get('SPECTRA_IMC_ROOT', 'raw/imc')); OUT = Path(__file__).resolve().parents[2] / 'runs/imc_linkage_20261111/mcd_cache'; OUT.mkdir(parents=True, exist_ok=True)
for sid in sys.argv[1:]:
    with MCDFile(IMC / sid / f'{sid}.mcd') as f:
        for s in f.slides:
            for a in s.acquisitions:
                desc = re.sub(r'[^0-9A-Za-z_]', '_', a.description or 'none')
                dest = OUT / f'{sid}_{desc}_{a.id}.npz'
                if dest.exists():
                    continue
                try:
                    img = f.read_acquisition(a).astype(np.float32)
                except Exception as e:
                    print(sid, a.id, 'read error', e); continue
                np.savez_compressed(dest, image=img, labels=np.array(a.channel_labels), names=np.array(a.channel_names))
                print(sid, a.id, desc, img.shape, flush=True)
