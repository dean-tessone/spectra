"""Evaluate a presealed fixed SPECTRA ensemble only after matched MIL CV."""
from pathlib import Path
import sys,os,argparse,subprocess,time,fcntl,importlib.util
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import ROOT as MAIN,read,write,sha,utc
from runtime import Cache,score,model_for
from report_detection import diagnostic,operating
ROOT=REPO/'runs/reference_control_test_20261110_v1a';OUT=REPO/'analyses/reference_control_test_20261110_v1a'
FULL=REPO/'runs/reference_control_full_refit_20261110';MIL=REPO/'runs/reference_mil_cv_20261110'
BASE=REPO/'model/pipeline/run_sources.py'
spec=importlib.util.spec_from_file_location('sealed_test_setup',BASE);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)


def verify():
 r=read(ROOT/'registration.json')
 for p,h in r['inputs_sha256'].items():assert sha(p)==h,p
 return r

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'registration.json').exists():return verify()
 lock=read(FULL/'model_lock.json');complete=read(FULL/'complete.json');assert sha(FULL/'model_lock.json')==complete['model_lock_sha256']
 assert read(REPO/'runs/reference_depth_comparison_20261110/depth_choice.json')['selected_depth']==6
 assert set(lock['models'])=={'17','29','43'} and lock['model_config']['depth']==6
 for p,h in read(FULL/'registration.json')['inputs_sha256'].items():assert sha(p)==h
 files=[Path(__file__),Path(__file__).with_name('report.py'),BASE,REPO/'model/pipeline/runtime.py',REPO/'model/pipeline/report_detection.py',FULL/'model_lock.json',FULL/'complete.json',FULL/'registration.json',FULL/'data/contexts/full/provenance.json',Path(lock['normalizer']),MIL/'registration.json',REPO/'runs/reference_depth_comparison_20261110/depth_choice.json',REPO/'cohort/reference_policy/population_patient_ledger.csv',REPO/'cohort/reference_policy/POLICY.md',MAIN/'data/cache_receipt.json',MAIN/'cohort/local_cache_inventory.csv',MAIN/'cohort/all_groups.csv',MAIN/'upstream_provenance_audit.json',REPO/'analyses/clinical_cohort_annotations_20261110/patient_clinical_ledger.csv',REPO/'analyses/bluev2_ctc_comparison_20261110_v1b/private_primary_patient_counts.csv']
 assert sha(lock['normalizer'])==lock['normalizer_sha256']
 for seed,m in lock['models'].items():assert sha(m['checkpoint'])==m['checkpoint_sha256'];assert m['fixed_epoch']==15;files.append(Path(m['checkpoint']))
 ledger=pd.read_csv(REPO/'cohort/reference_policy/population_patient_ledger.csv')
 fit=set(read(FULL/'data/contexts/full/provenance.json')['fit_donors']);test=set(ledger[ledger.split.eq('test')].donor_id);primary=ledger[ledger.split.eq('test')&ledger.primary_cancer_vs_reference_eligible]
 assert len(fit)==1793 and len(test)==788 and len(primary)==757 and not fit&test
 assert primary.label.eq(1).sum()==713 and primary.label.eq(0).sum()==44
 reserved_training=set(ledger[ledger.split.eq('train')&~ledger.primary_cancer_vs_reference_eligible].donor_id);assert not fit&reserved_training
 write(ROOT/'registration.json',dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in files},primary_cases=713,primary_controls=44,all_test_donors=788,reserved_test_controls=31,seeds=[17,29,43],recipe='d6_do04_lr2e4',epochs=15,primary_endpoint='one primary group per donor',predecessor=str(MIL),test_authorized_by='study protocol',new_fitting=False,model_retuning=False))
 write(ROOT/'status.json',dict(phase='prepared_test_not_accessed',updated_utc=utc()))
 return verify()


def tie_auc(d):
 pos=d[d.label==1].probability.to_numpy();neg=d[d.label==0].probability.to_numpy()
 return float(np.mean((pos[:,None]>neg[None,:]).astype(float)+.5*(pos[:,None]==neg[None,:])))

def evaluate():
 reg=verify();assert read(MIL/'complete.json')['fits']==30
 assert (REPO/'analyses/reference_mil_cv_20261110/model_comparison.csv').exists()
 if (ROOT/'complete.json').exists():return
 engine.setup(17);OUT.mkdir(parents=True,exist_ok=True)
 ledger=pd.read_csv(REPO/'cohort/reference_policy/population_patient_ledger.csv');test=set(ledger[ledger.split.eq('test')].donor_id)
 cache=Cache(test);entries=cache.entries.copy();manifest=pd.read_csv(MAIN/'cohort/all_groups.csv',dtype={'slide_id':str});m=manifest[manifest.donor_id.isin(test)]
 primary=m[m.primary_evaluation].drop_duplicates('donor_id').set_index('donor_id');strict=set()
 for gid,g in m.groupby('group_id'):
  donor=g.donor_id.iloc[0]
  if g.tube_id.eq(primary.loc[donor,'tube_id']).all() and g.draw_date.eq(primary.loc[donor,'draw_date']).all():strict.add(gid)
 lock=read(FULL/'model_lock.json')
 with np.load(lock['normalizer']) as z:mean,scale=[torch.tensor(z[k],device='cuda') for k in ['mean','scale']]
 assert torch.isfinite(mean).all() and torch.isfinite(scale).all() and (scale>0).all()
 frames=[];replays=[]
 for seed in reg['seeds']:
  write(ROOT/'status.json',dict(phase='sealed_test_inference',seed=seed,worker_pid=os.getpid(),updated_utc=utc()))
  model=model_for('transformer',reg['recipe']).cuda();state=torch.load(lock['models'][str(seed)]['checkpoint'],map_location='cpu',weights_only=False)
  assert state['epoch']==15 and state['identity']['seed']==seed and state['identity']['fold']=='full'
  model.load_state_dict(state['model'],strict=True);assert sum(p.numel() for p in model.parameters())==4870913
  d=score(model,'transformer',cache,mean,scale,primary=False);assert d.group_id.is_unique and set(d.donor_id)==test
  ids=d.group_id.head(8).tolist();cache.entries={g:entries[g] for g in reversed(ids)}
  replay=score(model,'transformer',cache,mean,scale,primary=False).set_index('group_id');cache.entries=entries
  error=float(np.max(np.abs(replay.loc[ids].probability.to_numpy()-d.set_index('group_id').loc[ids].probability.to_numpy())));assert error<=1e-6
  replays.append(dict(seed=seed,groups=8,max_absolute_probability_error=error))
  d['seed']=seed;d.to_csv(ROOT/f'private_seed_{seed}_group_predictions.csv',index=False);frames.append(d)
  del model,state;torch.cuda.empty_cache()
 groups=pd.concat(frames);assert groups.groupby('group_id').size().eq(3).all()
 averaged=groups.groupby(['group_id','donor_id','label','cancer_types','study','primary_evaluation'],as_index=False).agg(probability=('probability','mean'),retained_cells=('retained_cells','first'))
 assert np.isfinite(averaged.probability).all() and averaged.probability.between(0,1).all()
 for endpoint,chosen in [('primary',averaged[averaged.primary_evaluation]),('strict_same_draw_tube_supplemental',averaged[averaged.group_id.isin(strict)]),('all_eligible_groups_supplemental',averaged)]:
  donors=chosen.groupby(['donor_id','label','cancer_types','study'],as_index=False).agg(probability=('probability','mean'),groups=('group_id','size'),retained_cells=('retained_cells','sum'))
  assert len(donors)==788 and donors.donor_id.is_unique
  donors.to_csv(ROOT/f'private_{endpoint}_donor_predictions.csv',index=False)
 write(ROOT/'inference_complete.json',dict(completed_utc=utc(),registration_sha256=sha(ROOT/'registration.json'),test_donors=788,primary_donors=757,primary_replay=replays,predecessor_complete_sha256=sha(MIL/'complete.json'),model_lock_sha256=sha(FULL/'model_lock.json'),outputs_sha256={str(p):sha(p) for p in ROOT.glob('private_*csv')},new_fitting=False))
 subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],cwd=REPO,check=True)
 write(ROOT/'complete.json',dict(completed_utc=utc(),inference_receipt_sha256=sha(ROOT/'inference_complete.json'),report_receipt_sha256=sha(OUT/'receipt.json'),primary_cases=713,primary_controls=44,model_lock_unchanged=sha(FULL/'model_lock.json')==reg['inputs_sha256'][str(FULL/'model_lock.json')],new_fitting=False))
 write(ROOT/'status.json',dict(phase='sealed_test_evaluation_complete',updated_utc=utc()))

def queue():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare()
  while not (MIL/'complete.json').exists():
   s=read(MIL/'status.json')
   if s.get('phase')=='failed':raise RuntimeError('MIL CV predecessor failed; wait for repair rather than bypass')
   pid=s.get('controller_pid')
   if pid:
    try:os.kill(pid,0)
    except ProcessLookupError:raise RuntimeError('MIL predecessor stopped without completion')
   write(ROOT/'status.json',dict(phase='waiting_for_MIL_CV',controller_pid=os.getpid(),test_accessed=False,updated_utc=utc()));time.sleep(30)
  evaluate()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','queue','evaluate']);a=p.parse_args()
 try:
  if a.action=='prepare':print(prepare()['primary_cases'],'cases and44referencecontrols registered;no testinference')
  elif a.action=='queue':queue()
  else:evaluate()
 except Exception as e:
  write(ROOT/'status.json',dict(phase='failed',error=repr(e),updated_utc=utc()));raise
