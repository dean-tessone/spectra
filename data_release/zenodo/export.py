"""Public data release: per-event embeddings for the top-10,000 outlier events of every study slide, with
cell type (frozen BLUE v2 phenotype classifier), leukocyte score (wbcclass) and junk score (QC-v1 unusable probability) in the
same table, plus a separate slide table with cancer type. No patient identifiers: slide_id and tube_id (first five characters
of slide_id) only; no clinical information beyond cancer type.

    python export.py run [workers]     # write shards (resumable: finished shards with a receipt are skipped)
    python export.py validate          # re-read every shard and check the release against its sources
"""
from pathlib import Path
import hashlib, importlib.util, json, os, sys, time
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('SPECTRA_RELEASE_DIR', Path.home() / 'Documents/SPECTRA_zenodo_20261116'))   # local disk
INV = REPO / 'runs/paper_regeneration_20261109_v1a/cohort/local_cache_inventory.csv'
POP = REPO / 'cohort/reference_policy/all_population_slides.csv'
BLUE = REPO / 'runs/bluev2_ctc_comparison_20261110_v1b'
QC_LIMIT = .46509784932900855          # junk-score cut-off used for model inputs
SLIDES_PER_SHARD = 200
CLASSES = ['Large Nuclei', 'imCTCs', 'CTCs', 'pcCTCs', 'CECs', 'Megakaryocytes', 'Fibroblasts', 'Lymph', 'Gran', 'Mono']
ROLE = {'Cancer': 'cancer', 'Healthy reference': 'reference_control', 'Non-cancer pancreatic workup, included in primary reference': 'reference_control',
        'Lung diagnostic workup, separate comparison': 'lung_workup_control', 'BRCA-mutant breast risk, separate comparison': 'brca_carrier_control'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def classifier():
    spec = importlib.util.spec_from_file_location('bluev2', BLUE / 'package/Phenotyping.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    model = m.load_model(BLUE / 'package/BLUEV2_LR_weights.npz'); assert model['class_names'].tolist() == CLASSES
    return m, model


def samples():
    """Slide table: one row per slide, no patient identifiers."""
    p = pd.read_csv(POP, dtype=str, low_memory=False)
    s = pd.DataFrame(dict(slide_id=p.slide_id, tube_id=p.slide_id.str[:5], cancer_type=p.cancer_types, split=p.split,
                          cv_fold=p.fold.astype(int), primary_group=p.primary_evaluation == 'True', population=p.control_population.map(ROLE)))
    assert s.slide_id.is_unique and s.population.notna().all() and (s.tube_id == p.tube_id).all()
    assert ((s.cancer_type == 'Control') == (s.population != 'cancer')).all()
    return s.sort_values(['split', 'slide_id'], ascending=[False, True]).reset_index(drop=True)


def shards(s):
    out = []
    for split in ['train', 'test']:
        ids = s[s.split == split].slide_id.tolist()
        out += [(f'{split}/events_{split}_{i // SLIDES_PER_SHARD:03d}.parquet', ids[i:i + SLIDES_PER_SHARD]) for i in range(0, len(ids), SLIDES_PER_SHARD)]
    return out


def slide_table(row, official, model, check_source):
    import pyarrow as pa
    raw_b = Path(row.raw_cache).read_bytes(); qc_b = Path(row.qc_cache).read_bytes()
    assert hashlib.sha256(raw_b).hexdigest() == row.raw_sha256 and hashlib.sha256(qc_b).hexdigest() == row.qc_sha256, row.name
    assert sha(row.source) == row.source_sha256, row.name
    import io
    z = np.load(io.BytesIO(raw_b), allow_pickle=False); qc = np.load(io.BytesIO(qc_b), allow_pickle=False)
    f = pd.read_parquet(row.source, columns=['frame_id', 'cell_id', 'TRITC_mean', 'CY5_mean', 'FITC_mean', 'wbcclass'] + ([f'z{j}' for j in range(512)] if check_source else []))
    n = len(z); assert z.shape == (n, 512) and 0 < n <= 10000 and z.dtype == np.float32 and qc.shape == (n,) and len(f) == n   # 7 slides have <10,000 candidates
    if check_source:
        np.testing.assert_array_equal(f[[f'z{j}' for j in range(512)]].to_numpy(np.float32), z)
    cls = np.concatenate([official.predict_probabilities(z[i:i + 4096], model).argmax(1) for i in range(0, len(z), 4096)]).astype(np.int8)
    wbc = f.wbcclass.to_numpy(np.float32)
    model_input = (qc <= QC_LIMIT) & (wbc < .9) & (f[['TRITC_mean', 'CY5_mean', 'FITC_mean']].to_numpy() >= 5000).any(1)
    prev = np.load(BLUE / 'slides' / row.name / 'classes.npz')['class_id']          # earlier run classified model inputs only
    np.testing.assert_array_equal(prev >= 0, model_input); np.testing.assert_array_equal(prev[model_input], cls[model_input])
    cols = dict(slide_id=pa.array([row.name] * n).dictionary_encode(), tube_id=pa.array([row.name[:5]] * n).dictionary_encode(),
                frame_id=pa.array(f.frame_id.to_numpy().astype(np.int32)), cell_id=pa.array(f.cell_id.to_numpy().astype(np.int32)),
                cell_type=pa.DictionaryArray.from_arrays(pa.array(cls), pa.array(CLASSES)), wbcclass=pa.array(wbc),
                junk_score=pa.array(qc.astype(np.float32)), model_input=pa.array(model_input))
    cols['embedding'] = pa.FixedSizeListArray.from_arrays(pa.array(z.ravel()), 512)   # 512-d float32 per event (one column compresses ~40% better than 512)
    return pa.table(cols), dict(events=n, model_input=int(model_input.sum()), cell_types=np.bincount(cls, minlength=10).tolist())


def write_shard(job):
    import pyarrow.parquet as pq
    name, ids, inv = job; dst = OUT / name; rec = dst.with_suffix('.receipt.json')
    if rec.exists():
        return json.loads(rec.read_text())
    official, model = classifier(); dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_suffix('.partial'); w = None; per = []
    for k, sid in enumerate(ids):
        t, info = slide_table(inv.loc[sid], official, model, check_source=(k % 25 == 0))
        if w is None:
            w = pq.ParquetWriter(tmp, t.schema, compression='zstd', compression_level=9)
        w.write_table(t, row_group_size=10000); per.append(dict(slide_id=sid, **info))
    w.close(); tmp.replace(dst)
    r = dict(file=name, slides=len(ids), rows=sum(x['events'] for x in per), bytes=dst.stat().st_size, sha256=sha(dst), per_slide=per, finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    rec.write_text(json.dumps(r)); return r


def run(workers=6):
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    s = samples(); inv = pd.read_csv(INV, dtype=str).set_index('slide_id')
    assert set(inv.index) == set(s.slide_id) and len(s) == 6049
    s.to_csv(OUT / 'samples.csv', index=False)
    pd.DataFrame(dict(cell_type_id=range(10), cell_type=CLASSES)).to_csv(OUT / 'cell_types.csv', index=False)
    jobs = [(n, ids, inv.loc[ids]) for n, ids in shards(s)]
    with Pool(workers) as pool:
        for r in pool.imap_unordered(write_shard, jobs):
            print(r['file'], r['slides'], round(r['bytes'] / 1e9, 2), 'GB', flush=True)
    recs = [json.loads((OUT / n).with_suffix('.receipt.json').read_text()) for n, _ in shards(s)]
    pd.DataFrame([{k: r[k] for k in ['file', 'slides', 'rows', 'bytes', 'sha256']} for r in recs]).to_csv(OUT / 'MANIFEST_shards.csv', index=False)
    print('done', sum(r['bytes'] for r in recs) / 1e9, 'GB')


def validate():
    import pyarrow.parquet as pq
    s = pd.read_csv(OUT / 'samples.csv', dtype={'slide_id': str, 'tube_id': str}); m = pd.read_csv(OUT / 'MANIFEST_shards.csv')
    seen, rows, mi = [], 0, 0; banned = {'patient_id', 'friendly_id', 'donor_id', 'study', 'draw_date', 'sites'}
    assert not banned & set(s.columns)
    for r in m.itertuples():
        p = OUT / r.file; assert sha(p) == r.sha256, r.file
        f = pq.ParquetFile(p); names = f.schema_arrow.names; assert not banned & set(names)
        assert names == ['slide_id', 'tube_id', 'frame_id', 'cell_id', 'cell_type', 'wbcclass', 'junk_score', 'model_input', 'embedding']
        t = f.read(columns=['slide_id', 'tube_id', 'junk_score', 'model_input', 'cell_type']).to_pandas()
        assert (t.tube_id.astype(str) == t.slide_id.astype(str).str[:5]).all() and t.groupby('slide_id', observed=True).size().le(10000).all() and len(t) == r.rows
        seen += t.slide_id.astype(str).unique().tolist(); rows += len(t); mi += int(t.model_input.sum())
    assert sorted(seen) == sorted(s.slide_id) and rows == int(m.rows.sum())
    res = dict(slides=len(seen), rows=rows, model_input_events=mi, shards=len(m), total_bytes=int(m.bytes.sum()), banned_columns_present=False,
               split_slides=s.split.value_counts().to_dict(), population_slides=s.population.value_counts().to_dict())
    (OUT / 'validation.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    run(int(sys.argv[2]) if len(sys.argv) > 2 else 6) if sys.argv[1] == 'run' else validate()
