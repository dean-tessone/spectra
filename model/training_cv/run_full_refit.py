"""Fixed historical-recipe full fitting; seal three seeds, no test evaluation."""
from pathlib import Path
import sys,os,argparse,fcntl,subprocess,importlib.util
import pandas as pd
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import ROOT as MAIN,read,write,sha,utc
ROOT=REPO/'runs/reference_control_full_refit_20261110'
CV=REPO/'runs/reference_control_refit_20261110'
POLICY=REPO/'cohort/reference_policy'
REPORT=REPO/'analyses/reference_control_refit_20261110'
BASE=REPO/'model/pipeline/run_sources.py'
spec=importlib.util.spec_from_file_location('reference_full_engine',BASE);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine);engine.ROOT=ROOT

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'registration.json').exists():return engine.verify_registration()
 assert read(CV/'complete.json')['CV_fits']==6
 for p,h in read(CV/'registration.json')['inputs_sha256'].items():assert sha(p)==h
 ledger=pd.read_csv(POLICY/'population_patient_ledger.csv')
 fit=sorted(ledger[ledger.split.eq('train')&ledger.primary_cancer_vs_reference_eligible].donor_id)
 excluded=set(ledger[ledger.split.eq('train')&~ledger.primary_cancer_vs_reference_eligible].donor_id);test=set(ledger[ledger.split.eq('test')].donor_id)
 assert len(fit)==1793 and not set(fit)&(test|excluded)
 with np.load(MAIN/'data/donor_moments.npz') as z:
  index={d:i for i,d in enumerate(z['donors'])};ix=[index[d] for d in fit];mean=z['mean'][ix].mean(0);scale=np.sqrt(np.maximum(z['second'][ix].mean(0)-mean**2,1e-12))
 d=ROOT/'data/contexts/full';d.mkdir(parents=True,exist_ok=True)
 np.savez(d/'normalization.npz',mean=mean.astype(np.float32),scale=scale.astype(np.float32))
 write(d/'provenance.json',dict(fit_donors=fit,held_donors=[],reserved_training_controls=sorted(excluded),test_features_used=False,normalization_sha256=sha(d/'normalization.npz'),weighting='QC-only primary cells, equal fitting donor moments'))
 files=[Path(__file__),BASE,CV/'registration.json',CV/'complete.json',REPORT/'summary.json',REPORT/'receipt.json',POLICY/'POLICY.md',POLICY/'population_patient_ledger.csv',MAIN/'data/donor_moments.npz',MAIN/'data/cache_receipt.json',d/'normalization.npz',d/'provenance.json']
 write(ROOT/'registration.json',dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in files},recipe='d6_do04_lr2e4',epochs=15,seeds=[17,29,43],fits=3,patients=1793,cases=1689,controls=104,initialization='fresh_random',duration_fixed=True,test_used=False,evaluation='Disabled; this stage seals models only',threshold_calibration='None; reports ROC-derived sensitivity at fixed specificity',population_policy_followed_test_inspection=True))
 write(ROOT/'status.json',dict(phase='prepared',updated_utc=utc()))
 return engine.verify_registration()

def queue():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r=prepare()
  for seed in r['seeds']:
   with (ROOT/f'seed_{seed}.log').open('a') as log:
    c=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'fit','--seed',str(seed)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    write(ROOT/'status.json',dict(phase='full_fitting',seed=seed,controller_pid=os.getpid(),active_pid=c.pid,updated_utc=utc()))
    if c.wait():raise RuntimeError(f'Full fit failed:{seed}')
  models={}
  for seed in r['seeds']:
   folder=ROOT/f"full_refit/transformer/{r['recipe']}/fold_full/seed_{seed}"
   done=read(folder/'complete.json')
   for p,h in done['outputs_sha256'].items():assert sha(p)==h
   cp=folder/'best.pt';models[str(seed)]=dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),fixed_epoch=15)
  normalizer=ROOT/'data/contexts/full/normalization.npz'
  write(ROOT/'model_lock.json',dict(sealed_utc=utc(),registration_sha256=sha(ROOT/'registration.json'),models=models,normalizer=str(normalizer),normalizer_sha256=sha(normalizer),model_config=dict(input_dim=512,width=256,heads=4,expansion=4,depth=6,dropout=.4),primary_control_studies=['NBD','EPIC','INTC17-C','PANC-Ctrl'],primary_aggregation='One earliest eligible primary group per donor, up to two slides; equal seed probability averaging',threshold_calibration='None',test_evaluated=False,main_raw_QC_index_receipt=str(MAIN/'data/cache_receipt.json'),main_raw_QC_index_receipt_sha256=sha(MAIN/'data/cache_receipt.json')))
  write(ROOT/'complete.json',dict(completed_utc=utc(),fits=3,model_lock_sha256=sha(ROOT/'model_lock.json'),test_used=False))
  write(ROOT/'status.json',dict(phase='ensemble_sealed_test_not_evaluated',updated_utc=utc()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','queue','fit']);p.add_argument('--seed',type=int);a=p.parse_args()
 try:
  if a.action=='prepare':print(prepare()['fits'],'full fits prepared')
  elif a.action=='queue':queue()
  else:
   r=engine.verify_registration();engine.fit('transformer',r['recipe'],'full',a.seed,r['epochs'],'full_refit')
 except Exception as e:
  write(ROOT/'status.json',dict(phase='failed',error=repr(e),updated_utc=utc()));raise
