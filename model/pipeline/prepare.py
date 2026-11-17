"""Restore historical QC-only primary-donor normalization on revised fit sets."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from common import *
OLD=REPO/'runs/paper_regeneration_20261109'
MOMENTS=DATA/'runs/pancancer_model_lock_v1/data/qc_donor_moments.parquet'
AUTH=DATA/'runs/pancancer_model_lock_v1/preparation_complete.json'
MOMENT_CONTEXT=DATA/'runs/pancancer_model_lock_v1/data/receipt.json'

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'preparation_complete.json').exists():
        receipt=read(ROOT/'preparation_complete.json')
        for section in ['inputs_sha256','outputs_sha256']:
            for p,h in receipt[section].items():assert sha(p)==h,p
        return
    inherited=read(OLD/'preparation_complete.json')
    for section in ['inputs_sha256','outputs_sha256']:
        for p,h in inherited[section].items():assert sha(p)==h,p
    auth=read(AUTH);assert sha(MOMENTS)==auth['outputs'][str(MOMENTS)]
    source=read(OLD/'data/cache_receipt.json')
    for r in source['slides'].values():
        for key in ['source','raw_cache','qc_cache']:
            st=local(r[key]).stat();assert [st.st_size,st.st_mtime_ns]==r[key+'_stat']
        assert sha(r['indices'])==r['indices_sha256']
    (ROOT/'cohort').mkdir(exist_ok=True);(ROOT/'data').mkdir(exist_ok=True)
    for p in (OLD/'cohort').glob('*.csv'):shutil.copyfile(p,ROOT/'cohort'/p.name)
    shutil.copyfile(OLD/'data/cache_receipt.json',ROOT/'data/cache_receipt.json')
    ledger=pd.read_csv(ROOT/'cohort/donor_ledger.csv');train=ledger[ledger.split.eq('train')];people=sorted(train.donor_id)
    manifest=pd.read_csv(ROOT/'cohort/all_groups.csv',dtype={'slide_id':str});primary=manifest[manifest.primary_evaluation & manifest.split.eq('train')]
    expected=primary.groupby('donor_id').slide_id.agg(lambda x:tuple(sorted(x))).to_dict()
    old_entries=read(MOMENT_CONTEXT)['entries']
    old_primary={e['donor_id']:tuple(sorted(e['slide_ids'])) for e in old_entries.values() if e['primary_evaluation']}
    assert all(old_primary.get(p)==expected[p] for p in people),'QC moment primary physical slides changed'
    moments=pd.read_parquet(MOMENTS).set_index('donor_id');assert set(people)<=set(moments.index)
    cols_m=[f'm{i}' for i in range(512)];cols_s=[f's{i}' for i in range(512)]
    selected=moments.loc[people];means=selected[cols_m].to_numpy();seconds=selected[cols_s].to_numpy()
    # Independent exact-source recounts authenticate reuse of compact unsupervised
    # moments. Do not use old fitted normalizers or old patient-model weights.
    recount=[]
    for donor in [people[0],people[len(people)//2],people[-1]]:
        sums=np.zeros(512);squares=np.zeros(512);count=0
        for slide in expected[donor]:
            r=source['slides'][slide];qc=np.load(r['qc_cache'],allow_pickle=False);idx=np.flatnonzero(qc<=.46509784932900855)
            x=np.load(r['raw_cache'],mmap_mode='r',allow_pickle=False)[idx].astype(np.float64)
            sums+=x.sum(0);squares+=np.square(x).sum(0);count+=len(x)
        np.testing.assert_allclose(sums/count,moments.loc[donor,cols_m].to_numpy(float),rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(squares/count,moments.loc[donor,cols_s].to_numpy(float),rtol=1e-12,atol=1e-14)
        assert count==int(moments.loc[donor,'qc_cells'])
        recount.append(dict(donor_id=donor,qc_cells=count,exact_source_moment_recount_passed=True))
    write(ROOT/'qc_moment_reuse_audit.json',dict(source_moments_sha256=sha(MOMENTS),source_receipt_sha256=sha(AUTH),primary_slide_identity_matches=len(people),independent_recounts=recount,old_fitted_scalers_reused=False,normalizer_population='QC-only primary cells, equal donor moments',model_input_filters_unchanged=True))
    np.savez_compressed(ROOT/'data/donor_moments.npz',donors=np.array(people),mean=means,second=seconds)
    for fold in ['full',0,1,2]:
        fit=sorted(train.loc[train.fold.ne(fold),'donor_id']) if fold!='full' else people
        held=sorted(train.loc[train.fold.eq(fold),'donor_id']) if fold!='full' else []
        rows=moments.loc[fit];mean=rows[cols_m].to_numpy().mean(0);second=rows[cols_s].to_numpy().mean(0);scale=np.sqrt(np.maximum(second-mean**2,1e-12))
        folder=ROOT/f'data/contexts/{fold}';folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/'normalization.npz',mean=mean.astype(np.float32),scale=scale.astype(np.float32))
        write(folder/'provenance.json',dict(fit_donors=fit,held_donors=held,test_features_used=False,weighting='Equal fitting donor; QC-only cells from canonical-primary slides within donor',normalization_sha256=sha(folder/'normalization.npz'),normalizer_population='Historical QC-only primary policy, new fitting subsets and fitted scales',compact_QC_moments_sha256=sha(MOMENTS)))
    inputs={**inherited['inputs_sha256'],str(OLD/'preparation_complete.json'):sha(OLD/'preparation_complete.json'),str(MOMENTS):sha(MOMENTS),str(AUTH):sha(AUTH),str(MOMENT_CONTEXT):sha(MOMENT_CONTEXT),str(Path(__file__)):sha(Path(__file__))}
    outputs=[*sorted((ROOT/'cohort').glob('*.csv')),ROOT/'data/cache_receipt.json',ROOT/'data/donor_moments.npz',ROOT/'qc_moment_reuse_audit.json',*sorted((ROOT/'data/contexts').glob('*/*'))]
    record=dict(completed_utc=utc(),inputs_sha256=inputs,outputs_sha256={str(p):sha(p) for p in outputs},patients=inherited['patients'],slides=inherited['slides'],groups=inherited['groups'],counts=inherited['counts'],earliest_endpoint_patients=inherited['earliest_endpoint_patients'],correction='Restore QC-only canonical-primary normalization; fit on revised training donors only',parent_run_preserved=str(OLD),full_feature_copies=0,indices_shared_from_parent=True)
    write(ROOT/'preparation_complete.json',record);write(ROOT/'status.json',dict(phase='corrected_preparation_complete',updated_utc=utc()))
    print('QC-only primary normalization restored for1865fitting donors; primary identities and3source recounts verified',flush=True)
if __name__=='__main__':main()
