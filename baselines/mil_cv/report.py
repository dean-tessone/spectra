"""Training-only fixed-budget MIL table versus the locked transformer."""
import os
from pathlib import Path
import sys,json,shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import read,write,sha,utc
from runtime import model_for
ROOT=REPO/'runs/reference_mil_cv_20261110';SIX=REPO/'runs/reference_control_refit_20261110'
OUT=REPO/'analyses/reference_mil_cv_20261110'
FAMILIES=['transformer','abmil','additive','mean','clam_sb','clam_mb'];RECIPE='fixed15_lr2e4'
TYPES=['Bladder','Bone','Breast','Lung','Multiple Myeloma','Pancreatic','Prostate','Upper Tract Urothelial Carcinoma','Uveal Melanoma']
NAMES={'transformer':'SPECTRA, six layers','abmil':'ABMIL','additive':'Additive MIL','mean':'Mean pooling','clam_sb':'CLAM-SB','clam_mb':'CLAM-MB'}

def ensemble(frames):
 a,b=frames
 assert a.donor_id.is_unique and b.donor_id.is_unique
 d=a[['donor_id','label','cancer_types','study','probability']].merge(b[['donor_id','label','probability']],on='donor_id',validate='one_to_one',suffixes=('_17','_29'))
 assert len(d)==len(a)==len(b) and d.label_17.eq(d.label_29).all()
 d=d.rename(columns={'label_17':'label'});d['probability']=(d.probability_17+d.probability_29)/2
 return d[['donor_id','label','cancer_types','study','probability']]

def main():
 OUT.mkdir(parents=True,exist_ok=True);reg=read(ROOT/'registration.json')
 for p,h in reg['inputs_sha256'].items():assert sha(p)==h
 inputs={};allpred=[]
 sup_path=REPO/'analyses/reference_control_allgroup_oof_20261110_v1a/private_allgroup_OOF_predictions.csv'
 sup=pd.read_csv(sup_path);sup=sup[sup.endpoint.eq('strict_same_draw_tube')]
 matched={}
 for family in FAMILIES:
  for fold in range(3):
   primary=[];supplemental=[]
   for seed in [17,29]:
    folder=(SIX/f'reference_cv/transformer/d6_do04_lr2e4/fold_{fold}/seed_{seed}' if family=='transformer' else ROOT/f'mil_cv/{family}/{RECIPE}/fold_{fold}/seed_{seed}')
    done=read(folder/'complete.json');p=folder/'validation_epoch_15.csv';assert sha(p)==done['outputs_sha256'][str(p)];inputs[str(p)]=sha(p);primary.append(pd.read_csv(p))
    if family!='transformer':
     q=folder/'strict_same_tube_donor_predictions.csv';sd=read(folder/'supplemental_complete.json');assert sha(q)==sd['outputs_sha256'][str(q)];inputs[str(q)]=sha(q);supplemental.append(pd.read_csv(q))
   p=ensemble(primary);s=(sup[sup.fold.eq(fold)][['donor_id','label','cancer_types','study','probability']].copy() if family=='transformer' else ensemble(supplemental))
   for endpoint,d in [('primary',p),('strict_same_draw_tube_supplemental',s)]:
    if family=='transformer':matched[(fold,endpoint)]=set(d.donor_id)
    else:assert set(d.donor_id)==matched[(fold,endpoint)]
    d=d.copy();d['fold']=fold;d['family']=family;d['endpoint']=endpoint;allpred.append(d)
 predictions=pd.concat(allpred,ignore_index=True);predictions.to_csv(OUT/'private_OOF_predictions.csv',index=False)
 metrics=[];per=[];operating=[]
 for (family,endpoint),g in predictions.groupby(['family','endpoint']):
  assert len(g)==1793 and g.donor_id.is_unique and g.label.eq(0).sum()==104
  for fold,d in [*g.groupby('fold'),('pooled_OOF_secondary',g)]:
   controls=d[d.label==0];aucs={}
   for cancer,ca in d[d.label==1].groupby('cancer_types'):
    sub=pd.concat([ca,controls]);aucs[cancer]=roc_auc_score(sub.label,sub.probability)
    per.append(dict(family=family,endpoint=endpoint,fold=fold,cancer_type=cancer,cases=len(ca),controls=len(controls),AUROC=aucs[cancer]))
   metrics.append(dict(family=family,endpoint=endpoint,fold=fold,pooled_AUROC=roc_auc_score(d.label,d.probability),fixed_nine_macro_AUROC=np.mean([aucs[t] for t in TYPES]),cases=int(d.label.eq(1).sum()),controls=len(controls)))
   fpr,tpr,threshold=roc_curve(d.label,d.probability,drop_intermediate=False)
   for target in [.9,.95]:
    good=np.flatnonzero(fpr<=1-target+1e-12);j=good[np.argmax(tpr[good])]
    operating.append(dict(family=family,endpoint=endpoint,fold=fold,target_specificity=target,achieved_specificity=1-fpr[j],threshold=threshold[j],sensitivity=tpr[j],true_positives=int(d[d.label==1].probability.ge(threshold[j]).sum()),cases=int(d.label.eq(1).sum()),true_negatives=int(controls.probability.lt(threshold[j]).sum()),controls=len(controls)))
 m=pd.DataFrame(metrics);m.to_csv(OUT/'fold_metrics.csv',index=False)
 pd.DataFrame(per).to_csv(OUT/'per_cancer_metrics.csv',index=False);pd.DataFrame(operating).to_csv(OUT/'operating_points.csv',index=False)
 summary=m[m.fold.ne('pooled_OOF_secondary')].groupby(['family','endpoint'],as_index=False).agg(mean_fold_pooled_AUROC=('pooled_AUROC','mean'),mean_fold_fixed_nine_macro_AUROC=('fixed_nine_macro_AUROC','mean'),minimum_fold_AUROC=('pooled_AUROC','min'),maximum_fold_AUROC=('pooled_AUROC','max'))
 pooled=m[m.fold.eq('pooled_OOF_secondary')][['family','endpoint','pooled_AUROC','fixed_nine_macro_AUROC']].rename(columns={'pooled_AUROC':'pooled_OOF_AUROC_secondary','fixed_nine_macro_AUROC':'pooled_OOF_macro_secondary'})
 summary=summary.merge(pooled,on=['family','endpoint'],validate='one_to_one');summary['model']=summary.family.map(NAMES)
 parameters={f:sum(p.numel() for p in model_for(f,'d6_do04_lr2e4').parameters()) for f in FAMILIES};summary['parameters']=summary.family.map(parameters)
 summary['display_order']=summary.family.map({f:i for i,f in enumerate(FAMILIES)});summary=summary.sort_values(['endpoint','display_order']).drop(columns='display_order')
 summary.to_csv(OUT/'model_comparison.csv',index=False)
 write(OUT/'receipt.json',dict(inputs_sha256=inputs,source_script_sha256=sha(Path(__file__)),registration_sha256=sha(ROOT/'registration.json'),test_used=False,transformer_changed=False,completed_utc=utc()))
 (OUT/'README.md').write_text('Fixed-budget reference-policy CV:5baseline families x3foldsx2seeds plus reused locked6layer transformer CV.15epochs,identical donor folds/fitting-only normalizers/input filters/sampling/optimizer budget. Primary mean within-fold2seedAUROC and fixed9macro;pooledOOF secondary. Models have distinct fixed native architectures/dropouts and CLAM instance losses, not independently tuned optima. Primary onegroupperdonor;strictdraw/tube averaging supplemental and only27reference controls receive extra groups. ROC sensitivity90/95%specificity descriptive within eachfold/comparison, not prospective calibration. No AUROC CIs or final-test access, no model promotion/full refits. Prior test inspection/population refinement means this is developmentalCV, not untouched confirmation.\n')
 dst=Path(os.environ.get('SPECTRA_SHARED_OUT', 'shared_outputs')) / 'reference_mil_cv_20261110';dst.mkdir(parents=True,exist_ok=True);hashes={}
 for name in ['model_comparison.csv','fold_metrics.csv','per_cancer_metrics.csv','operating_points.csv','README.md','receipt.json']:
  a=OUT/name;b=dst/name
  if b.exists():assert a.read_bytes()==b.read_bytes(),'Version conflicting shared output'
  else:shutil.copyfile(a,b)
  assert sha(a)==sha(b);hashes[name]=sha(b)
 write(OUT/'delivery_receipt.json',dict(destination=str(dst),outputs_sha256=hashes))
 print(summary.to_string(index=False))

if __name__=='__main__':main()
