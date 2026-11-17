"""Training-only matched-depth comparison, no final-test access."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import read,write,sha,utc
ROOT=REPO/'runs/reference_depth_comparison_20261110';SIX=REPO/'runs/reference_control_refit_20261110'
OUT=REPO/'analyses/reference_depth_comparison_20261110';OUT.mkdir(parents=True,exist_ok=True)
TYPES=['Bladder','Bone','Breast','Lung','Multiple Myeloma','Pancreatic','Prostate','Upper Tract Urothelial Carcinoma','Uveal Melanoma']
registration=read(ROOT/'registration.json')
for p,h in registration['inputs_sha256'].items():assert sha(p)==h
rows=[];per=[];ops=[];oofs=[];inputs={};matched={}
for depth in [6,3]:
 for fold in range(3):
  frames=[]
  for seed in [17,29]:
   d=(SIX/f'reference_cv/transformer/d6_do04_lr2e4/fold_{fold}/seed_{seed}' if depth==6 else ROOT/f'depth_cv/transformer/d3_do04_lr2e4/fold_{fold}/seed_{seed}')
   done=read(d/'complete.json');p=d/'validation_epoch_15.csv'
   assert sha(p)==done['outputs_sha256'][str(p)];inputs[str(p)]=sha(p)
   if depth==3:
    baseline=SIX/f'reference_cv/transformer/d6_do04_lr2e4/fold_{fold}/seed_{seed}/schedule.json'
    assert sha(d/'schedule.json')==sha(baseline),'Depth comparison changed training sampling'
   frames.append(pd.read_csv(p))
  a,b=frames;assert a.donor_id.is_unique and b.donor_id.is_unique
  x=a[['donor_id','label','cancer_types','study','probability']].merge(b[['donor_id','label','probability']],on='donor_id',validate='one_to_one',suffixes=('_17','_29'))
  assert len(x)==len(a)==len(b) and x.label_17.eq(x.label_29).all()
  x=x.rename(columns={'label_17':'label'});x['probability']=(x.probability_17+x.probability_29)/2;x['fold']=fold;x['depth']=depth
  if depth==6:matched[fold]=set(x.donor_id)
  else:assert matched[fold]==set(x.donor_id)
  oofs.append(x)
allpred=pd.concat(oofs,ignore_index=True)
for depth,g in allpred.groupby('depth'):
 assert len(g)==1793 and g.donor_id.is_unique
 for fold,d in [*g.groupby('fold'),('pooled_OOF_secondary',g)]:
  controls=d[d.label==0];aucs={}
  for cancer,ca in d[d.label==1].groupby('cancer_types'):
   sub=pd.concat([ca,controls]);auc=roc_auc_score(sub.label,sub.probability);aucs[cancer]=auc
   per.append(dict(depth=depth,fold=fold,cancer_type=cancer,cases=len(ca),controls=len(controls),AUROC=auc))
  rows.append(dict(depth=depth,fold=fold,pooled_AUROC=roc_auc_score(d.label,d.probability),fixed_nine_macro_AUROC=np.mean([aucs[t] for t in TYPES])))
  fpr,tpr,th=roc_curve(d.label,d.probability,drop_intermediate=False)
  for target in [.9,.95]:
   good=np.flatnonzero(fpr<=1-target+1e-12);j=good[np.argmax(tpr[good])]
   ops.append(dict(depth=depth,fold=fold,target_specificity=target,achieved_specificity=1-fpr[j],sensitivity=tpr[j],threshold=th[j],cases=int(d.label.eq(1).sum()),controls=len(controls)))
r=pd.DataFrame(rows);r.to_csv(OUT/'fold_metrics.csv',index=False)
pd.DataFrame(per).to_csv(OUT/'per_cancer_metrics.csv',index=False);pd.DataFrame(ops).to_csv(OUT/'operating_points.csv',index=False)
allpred.to_csv(OUT/'private_OOF_predictions.csv',index=False)
s=r[r.fold.ne('pooled_OOF_secondary')].groupby('depth',as_index=False).agg(mean_fold_pooled_AUROC=('pooled_AUROC','mean'),mean_fold_fixed_nine_macro_AUROC=('fixed_nine_macro_AUROC','mean'))
s['parameters']=s.depth.map({3:2501633,6:4870913});s.to_csv(OUT/'summary.csv',index=False)
choice=s.sort_values(['mean_fold_pooled_AUROC','parameters'],ascending=[False,True]).iloc[0]
write(ROOT/'depth_choice.json',dict(selected_depth=int(choice.depth),primary_metric='mean fold pooledAUROC',selected_metric=float(choice.mean_fold_pooled_AUROC),test_used=False,full_fit_launched=False,registration_sha256=sha(ROOT/'registration.json'),completed_utc=utc()))
write(OUT/'receipt.json',dict(inputs_sha256=inputs,script_sha256=sha(Path(__file__)),registration_sha256=sha(ROOT/'registration.json')))
(OUT/'README.md').write_text('Matched depth-only fixed15epochCV on reference controls.3foldsx2seeds, identical held donors/normalizers/training sampling. Primary depth choice mean within-fold pooledAUROC; fixed-nine macro secondary, smaller model only breaks exact primary ties. PooledOOF secondary, no AUROC CIs or calibration. ROC operating points descriptive. No final-test access or automatic detector promotion. New three-layer CV followed prior population/test inspection; not independent confirmation.\n')
print(s.to_string(index=False))
