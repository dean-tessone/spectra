"""Isolated depth-only CV; preserve original reference source and full fits."""
from pathlib import Path
import sys,os,argparse,subprocess,fcntl,time,importlib.util,shutil
import pandas as pd
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import read,write,sha,utc
ROOT=REPO/'runs/reference_depth_comparison_20261110'
SIX=REPO/'runs/reference_control_refit_20261110'
PREDECESSOR=REPO/'runs/reference_control_full_refit_20261110'
BASE=REPO/'model/pipeline/run_sources.py'
spec=importlib.util.spec_from_file_location('matched_depth_engine',BASE);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine);engine.ROOT=ROOT

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'registration.json').exists():return engine.verify_registration()
 assert read(SIX/'complete.json')['CV_fits']==6
 for p,h in read(SIX/'registration.json')['inputs_sha256'].items():assert sha(p)==h
 files=[Path(__file__),Path(__file__).with_name('report.py'),BASE,SIX/'registration.json',SIX/'complete.json']
 ledger=pd.read_csv(REPO/'cohort/reference_policy/population_patient_ledger.csv')
 test=set(ledger[ledger.split.eq('test')].donor_id)
 reserved=set(ledger[~ledger.primary_cancer_vs_reference_eligible].donor_id)
 for fold in range(3):
  source=SIX/f'data/contexts/{fold}';dest=ROOT/f'data/contexts/{fold}';dest.mkdir(parents=True,exist_ok=True)
  for name in ['normalization.npz','provenance.json']:
   shutil.copyfile(source/name,dest/name);assert sha(source/name)==sha(dest/name)
   files += [source/name,dest/name]
  proof=read(dest/'provenance.json');fit=set(proof['fit_donors']);held=set(proof['held_donors'])
  assert not fit&(held|test|reserved) and not held&(test|reserved)
  for seed in [17,29]:
   d=SIX/f'reference_cv/transformer/d6_do04_lr2e4/fold_{fold}/seed_{seed}'
   done=read(d/'complete.json')
   for name in ['validation_epoch_15.csv','schedule.json']:
    p=d/name;assert sha(p)==done['outputs_sha256'][str(p)];files.append(p)
 files.append(REPO/'cohort/reference_policy/population_patient_ledger.csv')
 write(ROOT/'registration.json',dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in files},recipe='d3_do04_lr2e4',epochs=15,seeds=[17,29],folds=3,CV_fits=6,depth_only=True,normalizers_exact_copy=True,test_used=False,primary_selection_metric='mean fold pooledAUROC',full_fit_launched=False))
 write(ROOT/'status.json',dict(phase='prepared',updated_utc=utc()))
 return engine.verify_registration()

def queue():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r=prepare()
  while not (PREDECESSOR/'complete.json').exists():
   s=read(PREDECESSOR/'status.json')
   if s.get('phase')=='failed':raise RuntimeError('Six-layer full-fit predecessor failed')
   pid=s.get('controller_pid')
   if pid:
    try:os.kill(pid,0)
    except ProcessLookupError:raise RuntimeError('Six-layer predecessor stopped before completion')
   write(ROOT/'status.json',dict(phase='waiting_for_six_layer_full_fit',controller_pid=os.getpid(),updated_utc=utc()))
   time.sleep(30)
  for fold in range(3):
   for seed in r['seeds']:
    with (ROOT/f'fold_{fold}_seed_{seed}.log').open('a') as log:
     c=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'fit','--fold',str(fold),'--seed',str(seed)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
     write(ROOT/'status.json',dict(phase='three_layer_CV_fitting',fold=fold,seed=seed,controller_pid=os.getpid(),active_pid=c.pid,updated_utc=utc()))
     if c.wait():raise RuntimeError(f'Fit failed:{fold}/{seed}')
  subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],cwd=REPO,check=True)
  write(ROOT/'complete.json',dict(completed_utc=utc(),CV_fits=6,registration_sha256=sha(ROOT/'registration.json'),choice_sha256=sha(ROOT/'depth_choice.json'),test_used=False))
  write(ROOT/'status.json',dict(phase='depth_comparison_complete',updated_utc=utc()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','queue','fit']);p.add_argument('--fold',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
 try:
  if a.action=='prepare':print(prepare()['CV_fits'],'matched three-layer CV fits prepared')
  elif a.action=='queue':queue()
  else:
   r=engine.verify_registration();engine.fit('transformer',r['recipe'],str(a.fold),a.seed,r['epochs'],'depth_cv')
 except Exception as e:
  write(ROOT/'status.json',dict(phase='failed',error=repr(e),updated_utc=utc()));raise
