"""Fixed-budget matched MIL CV over the reference controls."""
from pathlib import Path
import sys,os,argparse,subprocess,fcntl,shutil,importlib.util
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import read,write,sha,utc,ROOT as MAIN
from runtime import Cache,model_for,schedule,loader,forward,score,SB,MB
ROOT=REPO/'runs/reference_mil_cv_20261110'
SIX=REPO/'runs/reference_control_refit_20261110'
FAMILIES=['abmil','additive','mean','clam_sb','clam_mb']
BASE=REPO/'model/pipeline/run_sources.py'
spec=importlib.util.spec_from_file_location('matched_mil_engine',BASE);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine);engine.ROOT=ROOT
RECIPE='fixed15_lr2e4'

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'registration.json').exists():return engine.verify_registration()
 for p,h in read(SIX/'registration.json')['inputs_sha256'].items():assert sha(p)==h
 assert read(REPO/'runs/reference_depth_comparison_20261110/depth_choice.json')['selected_depth']==6
 assert read(REPO/'runs/reference_control_full_refit_20261110/status.json')['phase']=='ensemble_sealed_test_not_evaluated'
 files=[Path(__file__),Path(__file__).with_name('report.py'),BASE,REPO/'model/pipeline/runtime.py',SIX/'registration.json',SIX/'complete.json',REPO/'runs/reference_depth_comparison_20261110/depth_choice.json',MAIN/'cohort/all_groups.csv',MAIN/'data/cache_receipt.json',REPO/'cohort/reference_policy/population_patient_ledger.csv',REPO/'analyses/reference_control_allgroup_oof_20261110_v1a/private_allgroup_OOF_predictions.csv']
 for rel in ['model/architectures/mil.py','model/architectures/mean_pool.py','model/architectures/clam_sb.py','model/architectures/clam_mb.py','src/spectra/model.py']:
  files.append(REPO/rel)
 ledger=pd.read_csv(REPO/'cohort/reference_policy/population_patient_ledger.csv');test=set(ledger[ledger.split.eq('test')].donor_id);reserved=set(ledger[~ledger.primary_cancer_vs_reference_eligible].donor_id)
 for fold in range(3):
  source=SIX/f'data/contexts/{fold}';dest=ROOT/f'data/contexts/{fold}';dest.mkdir(parents=True,exist_ok=True)
  for name in ['normalization.npz','provenance.json']:
   shutil.copyfile(source/name,dest/name);assert sha(source/name)==sha(dest/name);files += [source/name,dest/name]
  proof=read(dest/'provenance.json');fit=set(proof['fit_donors']);held=set(proof['held_donors'])
  assert not fit&(held|test|reserved) and not held&(test|reserved)
  for seed in [17,29]:
   d=SIX/f'reference_cv/transformer/d6_do04_lr2e4/fold_{fold}/seed_{seed}';done=read(d/'complete.json')
   for name in ['schedule.json','validation_epoch_15.csv']:
    p=d/name;assert sha(p)==done['outputs_sha256'][str(p)];files.append(p)
 write(ROOT/'registration.json',dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in files},families=FAMILIES,recipe=RECIPE,epochs=15,seeds=[17,29],folds=3,fits=30,normalizers_exact_copy=True,primary_endpoint='one primary group per held donor',supplemental_endpoint='strict primary draw/tube group probability averaging',test_used=False,transformer_changed=False,auto_promotion=False))
 write(ROOT/'status.json',dict(phase='prepared',updated_utc=utc()))
 return engine.verify_registration()

def preflight():
 engine.verify_registration();engine.setup(17)
 results={}
 for fold in range(3):
  proof=read(ROOT/f'data/contexts/{fold}/provenance.json');cache=Cache(proof['fit_donors']);held=Cache(proof['held_donors'])
  for seed in [17,29]:
   design=schedule(cache,seed,15);original=read(SIX/f'reference_cv/transformer/d6_do04_lr2e4/fold_{fold}/seed_{seed}/schedule.json')
   import json
   assert json.loads(json.dumps(design))==original
  results[f'fold_{fold}']=dict(complete_local_cache=True,fit_donors=len(proof['fit_donors']),held_donors=len(proof['held_donors']),exact_sampling_schedules=True)
 proof=read(ROOT/'data/contexts/0/provenance.json');cache=Cache(proof['fit_donors'])
 ids=[g for label in [0,1] for g in sorted([g for g,e in cache.entries.items() if e['label']==label],key=lambda g:sum(cache.entries[g]['lengths']),reverse=True)[:4]]
 dl=loader(cache,batches=[[(g,1/8) for g in ids]]);batch=next(iter(dl))
 with np.load(ROOT/'data/contexts/0/normalization.npz') as z:mean,scale=[torch.tensor(z[k],device='cuda') for k in ['mean','scale']]
 for family in FAMILIES:
  torch.cuda.reset_peak_memory_stats();model=model_for(family,RECIPE).cuda();model.train()
  out,mask=forward(model,family,batch,mean,scale);labels=batch['label'].cuda()
  losses=SB.training_losses(model,out,labels,mask) if family=='clam_sb' else MB.training_losses(model,out,labels,mask) if family=='clam_mb' else F.binary_cross_entropy_with_logits(out['logits'],labels,reduction='none')
  loss=losses.mean();assert torch.isfinite(loss);loss.backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
  model.eval();x=((batch['features'][:2,:128].cuda().float()-mean)/scale).half().float();m=batch['attention_mask'][:2,:128].cuda()
  with torch.no_grad():
   one=model(x,m)['logits'];two=model(x.flip(1),m.flip(1))['logits'];pad=model(torch.cat([x,torch.randn_like(x[:,:17])*100],1),torch.cat([m,torch.zeros_like(m[:,:17])],1))['logits']
  permutation=float((one-two).abs().max());padding=float((one-pad).abs().max())
  assert permutation<5e-5 and padding<5e-5
  results[family]=dict(parameters=sum(p.numel() for p in model.parameters()),finite_gradients=True,permutation_maxerror=permutation,padding_maxerror=padding,peak_memory_bytes=torch.cuda.max_memory_allocated())
  del model,out,loss,losses;torch.cuda.empty_cache()
 write(ROOT/'preflight.json',dict(passed=True,checks=results,registration_sha256=sha(ROOT/'registration.json'),test_used=False,completed_utc=utc()))
 print('Matched MIL preflight passed',flush=True)

def supplemental(family,fold,seed):
 engine.verify_registration();engine.setup(seed)
 folder=ROOT/f'mil_cv/{family}/{RECIPE}/fold_{fold}/seed_{seed}'
 if (folder/'supplemental_complete.json').exists():
  done=read(folder/'supplemental_complete.json')
  for p,h in done['outputs_sha256'].items():assert sha(p)==h
  return
 done=read(folder/'complete.json')
 for name in ['best.pt','validation_epoch_15.csv']:
  p=folder/name;assert sha(p)==done['outputs_sha256'][str(p)]
 baseline=SIX/f'reference_cv/transformer/d6_do04_lr2e4/fold_{fold}/seed_{seed}/schedule.json'
 assert sha(folder/'schedule.json')==sha(baseline)
 proof=read(ROOT/f'data/contexts/{fold}/provenance.json');held=proof['held_donors'];cache=Cache(held);entries=cache.entries.copy()
 manifest=pd.read_csv(MAIN/'cohort/all_groups.csv',dtype={'slide_id':str});m=manifest[manifest.donor_id.isin(held)];primary=m[m.primary_evaluation].drop_duplicates('donor_id').set_index('donor_id');strict=set()
 for gid,g in m.groupby('group_id'):
  donor=g.donor_id.iloc[0]
  if g.tube_id.eq(primary.loc[donor,'tube_id']).all() and g.draw_date.eq(primary.loc[donor,'draw_date']).all():strict.add(gid)
 source=pd.read_csv(folder/'validation_epoch_15.csv');assert set(source.donor_id)==set(held)
 with np.load(ROOT/f'data/contexts/{fold}/normalization.npz') as z:mean,scale=[torch.tensor(z[k],device='cuda') for k in ['mean','scale']]
 state=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);assert state['epoch']==15
 model=model_for(family,RECIPE).cuda();model.load_state_dict(state['model'],strict=True)
 selected=source.groupby('label',group_keys=False).head(4);cache.entries={g:entries[g] for g in selected.group_id}
 replay=score(model,family,cache,mean,scale,primary=False).set_index('group_id');error=float(np.abs(replay.loc[selected.group_id].probability.to_numpy()-selected.probability.to_numpy()).max());assert error<=1e-6
 extra={g:e for g,e in entries.items() if not e['primary_evaluation'] and g in strict};cache.entries=extra
 predictions=score(model,family,cache,mean,scale,primary=False) if extra else source.iloc[:0].copy()
 predictions.to_csv(folder/'strict_extra_group_predictions.csv',index=False)
 allgroups=pd.concat([source,predictions],ignore_index=True);assert allgroups.group_id.is_unique
 donors=allgroups.groupby(['donor_id','label','cancer_types','study'],as_index=False).agg(probability=('probability','mean'),groups=('group_id','size'))
 donors.to_csv(folder/'strict_same_tube_donor_predictions.csv',index=False)
 outputs=[folder/'strict_extra_group_predictions.csv',folder/'strict_same_tube_donor_predictions.csv']
 write(folder/'supplemental_complete.json',dict(outputs_sha256={str(p):sha(p) for p in outputs},primary_replay_error=error,extra_groups=len(extra),test_used=False,completed_utc=utc()))


def queue():
 ROOT.mkdir(parents=True,exist_ok=True)
 with (ROOT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r=prepare()
  if not (ROOT/'preflight.json').exists():subprocess.run([sys.executable,str(Path(__file__)),'preflight'],cwd=REPO,check=True)
  pre=read(ROOT/'preflight.json');assert pre['passed'] and pre['registration_sha256']==sha(ROOT/'registration.json')
  for family in FAMILIES:
   for fold in range(3):
    for seed in r['seeds']:
     with (ROOT/f'{family}_fold_{fold}_seed_{seed}.log').open('a') as log:
      c=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'fit','--family',family,'--fold',str(fold),'--seed',str(seed)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
      write(ROOT/'status.json',dict(phase='MIL_CV_fitting',family=family,fold=fold,seed=seed,controller_pid=os.getpid(),active_pid=c.pid,updated_utc=utc()))
      if c.wait():raise RuntimeError(f'MIL CV fit failed:{family}/{fold}/{seed}')
  subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],cwd=REPO,check=True)
  write(ROOT/'complete.json',dict(completed_utc=utc(),fits=30,registration_sha256=sha(ROOT/'registration.json'),test_used=False,transformer_changed=False))
  write(ROOT/'status.json',dict(phase='MIL_CV_comparison_complete',updated_utc=utc()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','queue','fit']);p.add_argument('--family',choices=FAMILIES);p.add_argument('--fold',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
 try:
  if a.action=='prepare':print(prepare()['fits'],'MIL CV fits prepared')
  elif a.action=='preflight':preflight()
  elif a.action=='queue':queue()
  else:
   engine.verify_registration();engine.fit(a.family,RECIPE,str(a.fold),a.seed,15,'mil_cv');supplemental(a.family,a.fold,a.seed)
 except Exception as e:
  write(ROOT/'status.json',dict(phase='failed',error=repr(e),updated_utc=utc()));raise
