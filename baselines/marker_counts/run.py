"""CPU frozen reference phenotyping over authenticated local candidate caches."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,json,shutil,importlib.util,time,argparse,fcntl
import numpy as np
import pandas as pd
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import ROOT as MAIN,read,write,sha,utc,local
ROOT=REPO/'runs/bluev2_ctc_comparison_20261110_v1b'
SOURCE=Path(os.environ.get('BLUEV2_CLASSIFIER', 'raw/BLUEV2_reference_classifier'))
UPSTREAM=Path(os.environ.get('SPECTRA_DATA_ROOT', 'data')) / 'encoder_provenance.json'
QC_LIMIT=.46509784932900855
CLASSES=['Large Nuclei','imCTCs','CTCs','pcCTCs','CECs','Megakaryocytes','Fibroblasts','Lymph','Gran','Mono']
GATES=[(ck,cy) for ck in [30000,40000,50000] for cy in [5000,10000,15000]]

def classifier():
 spec=importlib.util.spec_from_file_location('frozen_bluev2_package',ROOT/'package/Phenotyping.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m,m.load_model(ROOT/'package/BLUEV2_LR_weights.npz')

def verify():
 r=read(ROOT/'registration.json')
 for p,h in r['inputs_sha256'].items():assert sha(p)==h,p
 return r

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'registration.json').exists():return verify()
 package=ROOT/'package';package.mkdir(exist_ok=True)
 files=[Path(__file__),Path(__file__).with_name('report.py'),MAIN/'cohort/local_cache_inventory.csv',MAIN/'cohort/all_groups.csv',MAIN/'data/cache_receipt.json',MAIN/'upstream_provenance_audit.json',UPSTREAM,REPO/'cohort/reference_policy/population_patient_ledger.csv',MAIN/'ck_comparison/slide_gate_counts.csv']
 original={}
 for src in sorted(SOURCE.iterdir()):
  if not src.is_file():continue
  dest=package/src.name;shutil.copyfile(src,dest);assert sha(dest)==sha(src);original[src.name]=sha(src);files.append(dest)
 meta=read(package/'model_metadata.json');up=read(UPSTREAM)
 assert meta['representation_checkpoint']['sha256']==up['checkpoint_sha256']
 assert meta['class_names']==CLASSES and meta['embedding_dimension']==512
 provenance=read(MAIN/'upstream_provenance_audit.json');assert sha(UPSTREAM)==provenance['encoder_provenance_sha256']
 inventory=pd.read_csv(MAIN/'cohort/local_cache_inventory.csv',dtype=str,usecols=['slide_id','source','raw_cache','qc_cache','source_sha256','raw_sha256','qc_sha256'])
 assert len(inventory)==6049 and inventory.slide_id.is_unique
 write(ROOT/'registration.json',dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in files},original_package_sha256=original,class_names=CLASSES,CTC_class_ids=[1,2,3],primary_pool='actual_SPECTRA_inputs_all_three_filters',secondary_pool=None,primary_gate=dict(CK_strict_gt=40000,CY5_strict_lt=10000),robustness_gates=GATES,slides=6049,encoder_configured_identity_matches=True,per_parquet_encoder_identity_verified=False,phenotype_training_patient_overlap='unknown',CPU_threads=2,full_feature_copies=0,model_fitting=False,SPECTRA_test_inference=False))
 write(ROOT/'status.json',dict(phase='prepared',completed_slides=0,total_slides=6049,updated_utc=utc()))
 return verify()

def preflight():
 verify();official,model=classifier();assert model['class_names'].tolist()==CLASSES
 inv=pd.read_csv(MAIN/'cohort/local_cache_inventory.csv',dtype=str,usecols=['slide_id','source','raw_cache','qc_cache'])
 checks=[]
 for i in [0,len(inv)//4,len(inv)//2,3*len(inv)//4,len(inv)-1]:
  row=inv.iloc[i];raw=np.load(local(row.raw_cache),mmap_mode='r');source=pd.read_parquet(local(row.source),columns=[f'z{j}' for j in range(512)]).to_numpy(dtype=np.float32)
  np.testing.assert_array_equal(raw,source)
  record=read(MAIN/'data/cache_receipt.json')['slides'][row.slide_id]
  accepted=np.load(record['indices'],allow_pickle=False)[:73]
  x=np.asarray(raw[accepted],dtype=np.float32);p=official.predict_probabilities(x,model)
  frame=official.classify_frame(pd.DataFrame(x,columns=[f'z{j}' for j in range(512)]),model)
  np.testing.assert_array_equal(frame.predicted_class_id.to_numpy(),p.argmax(1))
  for batch in [1,7,32]:
   chunk=np.concatenate([official.predict_probabilities(x[j:j+batch],model) for j in range(0,len(x),batch)])
   np.testing.assert_allclose(p,chunk,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(p.argmax(1),chunk.argmax(1))
  np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
  checks.append(dict(slide_id=row.slide_id,source_raw_row_alignment=True,official_prediction_and_chunk_equivalence=True))
 ck=np.array([40000,40001,50000,40001]);cy=np.array([0,9999,10000,10001]);assert ((ck>40000)&(cy<10000)).tolist()==[False,True,False,False]
 write(ROOT/'preflight.json',dict(passed=True,checks=checks,strict_gate_boundaries=True,class_ids=list(range(10)),registration_sha256=sha(ROOT/'registration.json'),completed_utc=utc()))
 print('BLUEV2 preflight passed',flush=True)

def process(row,record,previous,official,model):
 slide=row.slide_id;folder=ROOT/'slides'/slide
 if (folder/'complete.json').exists():
  done=read(folder/'complete.json');assert done['registration_sha256']==sha(ROOT/'registration.json')
  for p,h in done['outputs_sha256'].items():assert sha(p)==h
  return read(folder/'counts.json')
 for field,hashfield in [('source','source_sha256'),('raw_cache','raw_sha256'),('qc_cache','qc_sha256')]:assert sha(local(row[field]))==row[hashfield],f'{slide}/{field}'
 idx=np.load(record['indices'],allow_pickle=False);assert sha(record['indices'])==record['indices_sha256']
 raw=np.load(local(row.raw_cache),mmap_mode='r',allow_pickle=False);qc=np.load(local(row.qc_cache),allow_pickle=False)
 f=pd.read_parquet(local(row.source),columns=['TRITC_mean','CY5_mean','FITC_mean','wbcclass'])
 assert raw.shape==(len(qc),512) and len(f)==len(qc)==record['candidate_cells']
 usable=qc<=QC_LIMIT
 native=(usable & (f.wbcclass.to_numpy()<.9) & ((f[['TRITC_mean','CY5_mean','FITC_mean']].to_numpy()>=5000).any(1)))
 np.testing.assert_array_equal(idx,np.flatnonzero(native));assert len(idx)==record['retained_cells']
 input_mask=np.zeros(len(qc),bool);input_mask[idx]=True;assert not (input_mask&~usable).any()
 ids=np.flatnonzero(input_mask);class_id=np.full(len(raw),-1,np.int8);sumprob=np.zeros(10);entropy_sum=0.;lowconf=0
 for start in range(0,len(ids),4096):
  chosen=ids[start:start+4096];p=official.predict_probabilities(raw[chosen],model);class_id[chosen]=p.argmax(1).astype(np.int8)
  sumprob+=p.sum(0);entropy_sum+=float(-(p*np.log(np.clip(p,1e-12,1))).sum());lowconf+=int((p.max(1)<.5).sum())
 assert np.all(class_id[input_mask]>=0) and np.all(class_id[~input_mask]==-1)
 ck=f.TRITC_mean.to_numpy();cy=f.CY5_mean.to_numpy();pools={};grid=[]
 for name,mask in [('model_input',input_mask)]:
  counts=np.bincount(class_id[mask],minlength=10);assert counts.sum()==mask.sum()
  gate=(ck>40000)&(cy<10000);overlap=np.bincount(class_id[mask&gate],minlength=10)
  pools[name]=dict(assessed_events=int(mask.sum()),class_counts=counts.tolist(),combined_CTC_like=int(counts[1:4].sum()),CK_gate_events=int((mask&gate).sum()),CK_gate_class_counts=overlap.tolist(),CTC_like_and_CK=int(overlap[1:4].sum()))
 for c,y in GATES:
  gate=(ck>c)&(cy<y);old=previous.loc[(slide,c,y)]
  qc_n=int((gate&usable).sum());input_n=int((gate&input_mask).sum())
  assert int(gate.sum())==int(old.gated_raw_events) and qc_n==int(old.gated_qc_usable_events) and input_n==int(old.gated_model_input_events),f'Existing CK count mismatch:{slide}/{c}/{y}'
  grid.append(dict(CK_threshold=c,CY5_threshold=y,raw_gate=int(gate.sum()),technical_QC_gate=qc_n,model_input_gate=input_n))
 data=dict(slide_id=slide,candidate_events=len(raw),technical_QC_rejected=int((~usable).sum()),pools=pools,input_class_probability_sums=sumprob.tolist(),input_entropy_sum=entropy_sum,input_low_confidence_below_point5=lowconf,gate_grid=grid)
 folder.mkdir(parents=True,exist_ok=True);write(folder/'counts.json',data)
 tmp=folder/'classes.partial.npz';np.savez_compressed(tmp,class_id=class_id);tmp.replace(folder/'classes.npz')
 write(folder/'complete.json',dict(registration_sha256=sha(ROOT/'registration.json'),outputs_sha256={str(p):sha(p) for p in [folder/'counts.json',folder/'classes.npz']},source_row_mapping='Array index is authenticated raw/source candidate row; -1=excluded_by_at_least_one_of_three_filters',completed_utc=utc()))
 return data

def queue():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r=prepare()
  if not (ROOT/'preflight.json').exists():preflight()
  assert read(ROOT/'preflight.json')['registration_sha256']==sha(ROOT/'registration.json')
  inventory=pd.read_csv(MAIN/'cohort/local_cache_inventory.csv',dtype=str,usecols=['slide_id','source','raw_cache','qc_cache','source_sha256','raw_sha256','qc_sha256'])
  actual=read(MAIN/'data/cache_receipt.json')['slides'];previous=pd.read_csv(MAIN/'ck_comparison/slide_gate_counts.csv',dtype={'slide_id':str}).set_index(['slide_id','ck_threshold','cy5_threshold'])
  official,model=classifier();counts=[];begun=time.monotonic()
  for i,(_,row) in enumerate(inventory.iterrows(),1):
   counts.append(process(row,actual[row.slide_id],previous,official,model))
   if i==1 or i%25==0 or i==len(inventory):
    write(ROOT/'status.json',dict(phase='CPU_phenotype_counting',completed_slides=i,total_slides=len(inventory),elapsed_seconds=time.monotonic()-begun,controller_pid=os.getpid(),updated_utc=utc()))
    print('Counted',i,'/',len(inventory),'slides',round(time.monotonic()-begun,1),'seconds',flush=True)
  rows=[];grid=[]
  for d in counts:
   for pool,x in d['pools'].items():
    row=dict(slide_id=d['slide_id'],pool=pool,candidate_events=d['candidate_events'],QC_rejected=d['technical_QC_rejected'],assessed_events=x['assessed_events'],CTC_like=x['combined_CTC_like'],CK_gate=x['CK_gate_events'],CTC_like_and_CK=x['CTC_like_and_CK'])
    for j,name in enumerate(CLASSES):row[name]=x['class_counts'][j];row['CK_and_'+name]=x['CK_gate_class_counts'][j]
    rows.append(row)
   grid.extend([dict(slide_id=d['slide_id'],**g) for g in d['gate_grid']])
  pd.DataFrame(rows).to_csv(ROOT/'slide_counts.csv',index=False);pd.DataFrame(grid).to_csv(ROOT/'slide_CK_gate_robustness.csv',index=False)
  write(ROOT/'counts_complete.json',dict(completed_utc=utc(),slides=len(counts),registration_sha256=sha(ROOT/'registration.json'),outputs_sha256={str(p):sha(p) for p in [ROOT/'slide_counts.csv',ROOT/'slide_CK_gate_robustness.csv']},model_fitting=False))
  import subprocess
  subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],cwd=REPO,check=True)
  write(ROOT/'complete.json',dict(completed_utc=utc(),slides=len(counts),model_fitting=False,SPECTRA_changed=False))
  write(ROOT/'status.json',dict(phase='complete',completed_slides=len(counts),updated_utc=utc()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','queue']);a=p.parse_args()
 try:
  if a.action=='prepare':print(prepare()['slides'],'slides registered')
  elif a.action=='preflight':preflight()
  else:queue()
 except Exception as e:
  write(ROOT/'status.json',dict(phase='failed',error=repr(e),updated_utc=utc()));raise
