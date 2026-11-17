"""Sealed, fixed-recipe reference-policy CV reporting, no final-test access."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'runs/reference_control_refit_20261110'
OUT=REPO/'analyses/reference_control_refit_20261110';OUT.mkdir(parents=True,exist_ok=True)
TYPES=['Bladder','Bone','Breast','Lung','Multiple Myeloma','Pancreatic','Prostate','Upper Tract Urothelial Carcinoma','Uveal Melanoma']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];per=[];operating=[];predictions=[];inputs={}
for fold in range(3):
 frames=[]
 for seed in [17,29]:
  d=ROOT/f'reference_cv/transformer/d6_do04_lr2e4/fold_{fold}/seed_{seed}';p=d/'validation_epoch_15.csv'
  done=json.loads((d/'complete.json').read_text());assert sha(p)==done['outputs_sha256'][str(p)]
  inputs[str(p)]=sha(p);a=pd.read_csv(p);assert a.donor_id.is_unique
  frames.append(a)
 a,b=frames
 x=a[['donor_id','label','cancer_types','study','probability']].merge(b[['donor_id','label','probability']],on='donor_id',validate='one_to_one',suffixes=('_17','_29'))
 assert len(x)==len(a)==len(b) and x.label_17.eq(x.label_29).all()
 x=x.rename(columns={'label_17':'label'});x['probability']=(x.probability_17+x.probability_29)/2;x['fold']=fold
 predictions.append(x)
values=pd.concat(predictions,ignore_index=True)
assert len(values)==1793 and values.donor_id.is_unique and values.label.eq(1).sum()==1689 and values.label.eq(0).sum()==104
for fold,d in [*values.groupby('fold'),('pooled_OOF_secondary',values)]:
 controls=d[d.label==0];aucs={}
 for cancer,g in d[d.label==1].groupby('cancer_types'):
  joined=pd.concat([g,controls]);auc=roc_auc_score(joined.label,joined.probability);aucs[cancer]=auc
  per.append(dict(fold=fold,cancer_type=cancer,cases=len(g),controls=len(controls),AUROC=auc))
 rows.append(dict(fold=fold,cases=int(d.label.eq(1).sum()),controls=len(controls),pooled_AUROC=roc_auc_score(d.label,d.probability),fixed_nine_macro_AUROC=np.mean([aucs[t] for t in TYPES])))
 fpr,tpr,threshold=roc_curve(d.label,d.probability,drop_intermediate=False)
 for target in [.9,.95]:
  good=np.where(fpr<=1-target+1e-12)[0];j=good[np.argmax(tpr[good])]
  operating.append(dict(fold=fold,target_specificity=target,achieved_specificity=1-fpr[j],sensitivity=tpr[j],threshold=threshold[j],cases=int(d.label.eq(1).sum()),controls=len(controls),true_positives=int(d[d.label==1].probability.ge(threshold[j]).sum()),true_negatives=int(controls.probability.lt(threshold[j]).sum())))
r=pd.DataFrame(rows);r.to_csv(OUT/'fold_metrics.csv',index=False)
pd.DataFrame(per).to_csv(OUT/'per_cancer_metrics.csv',index=False)
pd.DataFrame(operating).to_csv(OUT/'operating_points.csv',index=False)
values.to_csv(OUT/'private_OOF_predictions.csv',index=False)
f=r[r.fold.ne('pooled_OOF_secondary')];summary={'completed_CV_fits':6,'seeds':[17,29],'mean_fold_pooled_AUROC':f.pooled_AUROC.mean(),'mean_fold_fixed_nine_macro_AUROC':f.fixed_nine_macro_AUROC.mean(),'pooled_OOF_AUROC_secondary':r.iloc[-1].pooled_AUROC,'patients':1793,'cases':1689,'controls':104,'final_test_used':False}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'receipt.json').write_text(json.dumps(dict(inputs_SHA256=inputs,script_SHA256=sha(Path(__file__)),registration_SHA256=sha(ROOT/'registration.json')),indent=2)+'\n')
(OUT/'README.md').write_text('Reference-policy fixed historical-recipe CV:6layers/dropout0.4/LR0.0002/15epochs,3donor folds,2seed probability ensembles17/29. Primary is unweighted mean of within-fold AUROCs; pooled OOF is secondary because cross-fold score scales may differ. Cases1689,reference controls104: NBD,EPIC,INTC17-C,PANC-Ctrl. Reserved lung/BRCA-risk controls absent from source fitting, fitting normalizers and primary held endpoint. Fixed-nine training macro includes Bone and Uveal, unlike final-test macro8eligibility. No test access in these fits/report. ROC operating points are descriptive fold-specific sensitivities at specificity targets, not calibrated deployment thresholds. No AUROC CIs. Prior historical0.849test and0.857macro cannot be compared directly to trainingCV as matched validation. Fresh full-cohort model has not been fit by this stage.\n')
print(r.to_string(index=False));print(json.dumps(summary,indent=2))
