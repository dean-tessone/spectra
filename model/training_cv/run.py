"""Fresh historical-recipe CV under the reference-control policy."""
from pathlib import Path
import argparse,sys,json,os,subprocess,fcntl,importlib.util
import pandas as pd
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import ROOT as MAIN,read,write,sha,utc
ROOT=REPO/'runs/reference_control_refit_20261110'
POLICY=REPO/'cohort/reference_policy'
BASE=REPO/'model/pipeline/run_sources.py'
spec=importlib.util.spec_from_file_location('reference_fit_engine',BASE);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine);engine.ROOT=ROOT

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'registration.json').exists():return engine.verify_registration()
 upstream=read(MAIN/'preparation_complete.json')
 for p in [MAIN/'data/donor_moments.npz',MAIN/'cohort/donor_ledger.csv',MAIN/'data/cache_receipt.json']:
  assert sha(p)==upstream['outputs_sha256'][str(p)]
 ledger=pd.read_csv(POLICY/'population_patient_ledger.csv');train=ledger[ledger.split.eq('train')];test=set(ledger[ledger.split.eq('test')].donor_id)
 eligible=train[train.primary_cancer_vs_reference_eligible];reserved=set(train[~train.primary_cancer_vs_reference_eligible].donor_id)
 assert len(eligible)==1793 and len(reserved)==72
 with np.load(MAIN/'data/donor_moments.npz') as z:ids=list(z['donors']);means=z['mean'];seconds=z['second']
 index={d:i for i,d in enumerate(ids)}
 files=[Path(__file__),BASE,POLICY/'POLICY.md',POLICY/'population_patient_ledger.csv',POLICY/'receipt.json',MAIN/'data/donor_moments.npz',MAIN/'data/cache_receipt.json',MAIN/'cohort/donor_ledger.csv']
 audit=[]
 for fold in range(3):
  fit=set(eligible[eligible.fold.ne(fold)].donor_id);held=set(eligible[eligible.fold.eq(fold)].donor_id)
  assert not fit&(held|test|reserved) and not held&(test|reserved)
  rows=[index[d] for d in sorted(fit)];mean=means[rows].mean(0);scale=np.sqrt(np.maximum(seconds[rows].mean(0)-mean**2,1e-12))
  d=ROOT/f'data/contexts/{fold}';d.mkdir(parents=True,exist_ok=True)
  np.savez(d/'normalization.npz',mean=mean.astype(np.float32),scale=scale.astype(np.float32))
  write(d/'provenance.json',dict(fit_donors=sorted(fit),held_donors=sorted(held),excluded_reserved_controls=sorted(reserved),test_features_used=False,normalization_sha256=sha(d/'normalization.npz'),normalization_policy='QC-only primary cells, equal fitting donor moments'))
  f=train[train.donor_id.isin(fit)];h=train[train.donor_id.isin(held)]
  audit.append(dict(fold=fold,fitting_cases=int(f.label.eq(1).sum()),fitting_controls=int(f.label.eq(0).sum()),held_cases=int(h.label.eq(1).sum()),held_controls=int(h.label.eq(0).sum()),reserved_controls_used=0))
  files.extend([d/'normalization.npz',d/'provenance.json'])
 pd.DataFrame(audit).to_csv(ROOT/'exclusion_audit.csv',index=False);files.append(ROOT/'exclusion_audit.csv')
 write(ROOT/'registration.json',dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in files},recipe='d6_do04_lr2e4',epochs=15,seeds=[17,29],folds=3,CV_fits=6,test_used=False,patient_initialization='fresh_random',full_refit_launched=False))
 write(ROOT/'status.json',dict(phase='prepared',updated_utc=utc()))
 return engine.verify_registration()

def queue():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r=prepare()
  assert (REPO/'runs/lung_control_specialization_20261110/complete.json').exists()
  for fold in range(3):
   for seed in r['seeds']:
    with (ROOT/f'fold_{fold}_seed_{seed}.log').open('a') as log:
     c=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'fit','--fold',str(fold),'--seed',str(seed)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
     write(ROOT/'status.json',dict(phase='CV_fitting',fold=fold,seed=seed,controller_pid=os.getpid(),active_pid=c.pid,updated_utc=utc()))
     if c.wait():raise RuntimeError(f'Fit failed:{fold}/{seed}')
  write(ROOT/'complete.json',dict(completed_utc=utc(),CV_fits=6,full_refit_launched=False,test_used=False,registration_sha256=sha(ROOT/'registration.json')))
  write(ROOT/'status.json',dict(phase='CV_complete',updated_utc=utc()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','queue','fit']);p.add_argument('--fold',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
 try:
  if a.action=='prepare':r=prepare();print('Prepared',r['CV_fits'],'fixed15epochCVfits')
  elif a.action=='queue':queue()
  else:
   r=engine.verify_registration();engine.fit('transformer',r['recipe'],str(a.fold),a.seed,r['epochs'],'reference_cv')
 except Exception as e:
  write(ROOT/'status.json',dict(phase='failed',error=repr(e),updated_utc=utc()));raise
