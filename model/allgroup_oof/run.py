"""Frozen held-fold inference for supplemental same-draw/tube OOF averaging."""
from pathlib import Path
import sys,os,json,importlib.util,copy
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import ROOT as MAIN,read,write,sha,utc
from runtime import Cache,score,model_for
ROOT=REPO/'runs/reference_control_allgroup_oof_20261110_v1a'
CV=REPO/'runs/reference_control_refit_20261110'
OUT=REPO/'analyses/reference_control_allgroup_oof_20261110_v1a'
BASE=REPO/'model/pipeline/run_sources.py'
spec=importlib.util.spec_from_file_location('supplemental_inference_engine',BASE);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
TYPES=['Bladder','Bone','Breast','Lung','Multiple Myeloma','Pancreatic','Prostate','Upper Tract Urothelial Carcinoma','Uveal Melanoma']

def prepare():
 ROOT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'registration.json').exists():
  r=read(ROOT/'registration.json')
  for p,h in r['inputs_sha256'].items():assert sha(p)==h
  return r
 files=[Path(__file__),CV/'registration.json',CV/'complete.json',MAIN/'cohort/all_groups.csv',MAIN/'data/cache_receipt.json']
 for p,h in read(CV/'registration.json')['inputs_sha256'].items():assert sha(p)==h
 for fold in range(3):
  files += [CV/f'data/contexts/{fold}/normalization.npz',CV/f'data/contexts/{fold}/provenance.json']
  for seed in [17,29]:
   d=CV/f'reference_cv/transformer/d6_do04_lr2e4/fold_{fold}/seed_{seed}'
   done=read(d/'complete.json')
   for name in ['best.pt','validation_epoch_15.csv']:
    p=d/name;assert sha(p)==done['outputs_sha256'][str(p)];files.append(p)
 write(ROOT/'registration.json',dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in files},folds=3,seeds=[17,29],model_recipe='d6_do04_lr2e4',epoch=15,inference_only=True,training=False,test_used=False,aggregation='Two explicit endpoints: strict primary-draw/tube groups, and all eligible same-draw groups; equal group probability mean within seed then equal seed mean',primary_predictions='Reuse sealed held-primary predictions, validate8groups per fit by replay',new_extra_groups_per_seed=348,allocator_fraction=.2,free_reserve_GiB=8))
 return read(ROOT/'registration.json')

def main():
 r=prepare();OUT.mkdir(parents=True,exist_ok=True)
 if (ROOT/'complete.json').exists():return
 seed_donors=[];replays=[]
 ledger=pd.read_csv(REPO/'cohort/reference_policy/population_patient_ledger.csv')
 test=set(ledger[ledger.split.eq('test')].donor_id);reserved=set(ledger[~ledger.primary_cancer_vs_reference_eligible].donor_id)
 manifest=pd.read_csv(MAIN/'cohort/all_groups.csv',dtype={'slide_id':str})
 for fold in range(3):
  proof=read(CV/f'data/contexts/{fold}/provenance.json');fit=set(proof['fit_donors']);held=set(proof['held_donors'])
  assert not held&(fit|test|reserved)
  m=manifest[manifest.donor_id.isin(held)]
  primary=m[m.primary_evaluation].drop_duplicates('donor_id').set_index('donor_id')
  strict_groups=set()
  for gid,g in m.groupby('group_id'):
   donor=g.donor_id.iloc[0]
   assert g.tube_id.nunique()==1 and g.draw_date.eq(primary.loc[donor,'draw_date']).all(), 'Cross-draw group not eligible for these endpoints'
   if g.tube_id.eq(primary.loc[donor,'tube_id']).all():strict_groups.add(gid)
  cache=Cache(held);entries=cache.entries.copy()
  extra={g:e for g,e in entries.items() if not e['primary_evaluation']}
  groups_per_donor=pd.Series([e['donor_id'] for e in entries.values()]).value_counts()
  for seed in [17,29]:
   write(ROOT/'status.json',dict(phase='held_fold_extra_group_inference',fold=fold,seed=seed,extra_groups=len(extra),worker_pid=os.getpid(),updated_utc=utc()))
   engine.setup(seed)
   folder=ROOT/f'fold_{fold}/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
   source=CV/f'reference_cv/transformer/d6_do04_lr2e4/fold_{fold}/seed_{seed}'
   original=pd.read_csv(source/'validation_epoch_15.csv');assert set(original.donor_id)==held
   with np.load(CV/f'data/contexts/{fold}/normalization.npz') as z:mean,scale=[torch.tensor(z[k],device='cuda') for k in ['mean','scale']]
   state=torch.load(source/'best.pt',map_location='cpu',weights_only=False);assert state['epoch']==15
   assert state['identity']['fold']==str(fold) and state['identity']['seed']==seed
   model=model_for('transformer','d6_do04_lr2e4').cuda();model.load_state_dict(state['model'],strict=True)
   # Include repeated-control donors in the replay, plus primary groups of cases.
   ctrl=original[original.label.eq(0)].sort_values('donor_id');case=original[original.label.eq(1)].sort_values('donor_id')
   repeated=ctrl[ctrl.donor_id.map(groups_per_donor).gt(1)].head(4)
   selected=pd.concat([repeated,ctrl.head(4),case.head(4)]).drop_duplicates('group_id').head(8)
   cache.entries={gid:entries[gid] for gid in selected.group_id}
   replay=score(model,'transformer',cache,mean,scale,primary=False).set_index('group_id')
   maxerror=float(np.max(np.abs(replay.loc[selected.group_id].probability.to_numpy()-selected.probability.to_numpy())))
   assert maxerror<=1e-6,f'Primary replay mismatch:{maxerror}'
   replays.append(dict(fold=fold,seed=seed,groups=len(selected),max_absolute_probability_error=maxerror))
   cache.entries=extra
   if extra:pred=score(model,'transformer',cache,mean,scale,primary=False)
   else:pred=original.iloc[:0].copy()
   pred.to_csv(folder/'extra_group_predictions.csv',index=False)
   combined=pd.concat([original,pred],ignore_index=True);assert not combined.group_id.duplicated().any()
   assert set(combined.group_id)==set(entries)
   for endpoint in ['strict_same_draw_tube','all_eligible_same_draw_groups']:
    part=combined[combined.group_id.isin(strict_groups)] if endpoint=='strict_same_draw_tube' else combined
    d=part.groupby(['donor_id','label','cancer_types','study'],as_index=False).agg(probability=('probability','mean'),groups=('group_id','size'))
    d['fold']=fold;d['seed']=seed;d['endpoint']=endpoint;seed_donors.append(d)
    d.to_csv(folder/f'{endpoint}_donor_predictions.csv',index=False)
   del model,state,mean,scale;torch.cuda.empty_cache();cache.entries=entries
   print('Completed supplemental inference',fold,seed,len(pred),'extra groups',flush=True)
 avg=pd.concat(seed_donors).groupby(['endpoint','fold','donor_id','label','cancer_types','study'],as_index=False).agg(probability=('probability','mean'),groups=('groups','first'))
 avg.to_csv(OUT/'private_allgroup_OOF_predictions.csv',index=False)
 metrics=[];summaries=[]
 primary_summary=read(REPO/'analyses/reference_control_refit_20261110/summary.json')
 for endpoint,g in avg.groupby('endpoint'):
  assert len(g)==1793 and g.donor_id.is_unique and g.label.eq(0).sum()==104
  local=[]
  for fold,d in [*g.groupby('fold'),('pooled_OOF_secondary',g)]:
   controls=d[d.label==0];aucs={}
   for typ,ca in d[d.label==1].groupby('cancer_types'):
    a=pd.concat([ca,controls]);aucs[typ]=roc_auc_score(a.label,a.probability)
   row=dict(endpoint=endpoint,fold=fold,pooled_AUROC=roc_auc_score(d.label,d.probability),fixed_nine_macro_AUROC=np.mean([aucs[t] for t in TYPES]),cases=int(d.label.eq(1).sum()),controls=len(controls))
   metrics.append(row);local.append(row)
  table=pd.DataFrame(local);f=table[table.fold.ne('pooled_OOF_secondary')]
  summaries.append(dict(endpoint=endpoint,pooled_OOF_AUROC_secondary=float(table.iloc[-1].pooled_AUROC),mean_fold_pooled_AUROC=float(f.pooled_AUROC.mean()),mean_fold_macro_AUROC=float(f.fixed_nine_macro_AUROC.mean()),donors=1793,cases=1689,controls=104,donors_with_additional_groups=int(g.groups.gt(1).sum()),additional_groups=int(g.groups.sum()-len(g)),test_used=False))
 pd.DataFrame(metrics).to_csv(OUT/'fold_metrics.csv',index=False)
 pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)
 write(ROOT/'primary_replay.json',replays)
 (OUT/'README.md').write_text('Frozen6layer15epoch3fold2seed held-donor inference, no training/test access. Same eligible reference-policy1793training donors. Two endpoints explicitly distinguish strict same primary draw/tube from all eligible same-draw groups, which include additional tubes for4controls. Equal probability mean over groups within donor then seeds. Current cancer patients have only primary groups; additional groups occur in reference controls, so this remains supplemental with differing sampling effort. Historical all-group aggregation also included additional case groups under historical eligibility; old/new is not an exact matched acquisition comparison. Primary scores reused after8group-per-fit exactBF16replay tolerance1e-6. Failed v1 stopped at acquisition audit before inference; preserved separately. No retuning/model promotion.\n')
 write(ROOT/'complete.json',dict(completed_utc=utc(),registration_sha256=sha(ROOT/'registration.json'),summary_sha256=sha(OUT/'summary.csv'),inference_only=True,test_used=False))
 write(ROOT/'status.json',dict(phase='complete',updated_utc=utc()));print(pd.DataFrame(summaries).to_string(index=False),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  write(ROOT/'status.json',dict(phase='failed',error=repr(e),updated_utc=utc()));raise
