"""Deterministic primary-group retained-event count baseline, matched CV."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import ROOT as MAIN,read,write,sha,utc
ROOT=REPO/'runs/reference_count_cv_20261111';OUT=REPO/'analyses/reference_count_cv_20261111'
SIX=REPO/'runs/reference_control_refit_20261110'
TYPES=['Bladder','Bone','Breast','Lung','Multiple Myeloma','Pancreatic','Prostate','Upper Tract Urothelial Carcinoma','Uveal Melanoma']

def main():
 ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 ledger_path=REPO/'cohort/reference_policy/population_patient_ledger.csv'
 files=[Path(__file__),ledger_path,MAIN/'cohort/all_groups.csv',MAIN/'data/cache_receipt.json',SIX/'registration.json']
 for fold in range(3):files += [SIX/f'data/contexts/{fold}/provenance.json']
 if (ROOT/'registration.json').exists():
  for p,h in read(ROOT/'registration.json')['inputs_sha256'].items():assert sha(p)==h
 else:write(ROOT/'registration.json',dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in files},model='Spline logistic regression of log1p total retained primary-group events',n_knots=3,degree=3,include_bias=True,logistic_C=1,class_weight='balanced',max_iter=10000,random_state=17,parameters=6,folds=3,neural_epochs_not_applicable=True,test_used=False))
 ledger=pd.read_csv(ledger_path);train=ledger[ledger.split.eq('train')&ledger.primary_cancer_vs_reference_eligible]
 assert len(train)==1793 and train.label.eq(0).sum()==104
 manifest=pd.read_csv(MAIN/'cohort/all_groups.csv',dtype={'slide_id':str});m=manifest[manifest.primary_evaluation&manifest.donor_id.isin(train.donor_id)].copy();assert m.slide_id.is_unique
 receipt=read(MAIN/'data/cache_receipt.json')['slides'];counts={}
 for slide in m.slide_id:
  r=receipt[slide];assert sha(r['indices'])==r['indices_sha256']
  n=len(np.load(r['indices'],allow_pickle=False));assert n==r['retained_cells'];counts[slide]=n
 m['retained_events']=m.slide_id.map(counts)
 patient=m.groupby('donor_id',as_index=False).agg(retained_events=('retained_events','sum'),slides=('slide_id','nunique'))
 patient=patient.merge(train[['donor_id','label','cancer_types','fold']],on='donor_id',validate='one_to_one');assert len(patient)==1793
 predictions=[];per=[];metrics=[]
 for fold in range(3):
  proof=read(SIX/f'data/contexts/{fold}/provenance.json');fit=set(proof['fit_donors']);held=set(proof['held_donors']);assert not fit&held
  a=patient[patient.donor_id.isin(fit)];b=patient[patient.donor_id.isin(held)].copy();assert set(a.donor_id)==fit and set(b.donor_id)==held
  model=make_pipeline(SplineTransformer(n_knots=3,degree=3,include_bias=True),LogisticRegression(C=1,class_weight='balanced',max_iter=10000,random_state=17))
  model.fit(np.log1p(a[['retained_events']]),a.label);assert model[-1].coef_.shape==(1,5)
  assert np.all(model[-1].n_iter_<10000)
  b['probability']=model.predict_proba(np.log1p(b[['retained_events']]))[:,1];predictions.append(b)
  write(ROOT/f'fold_{fold}_model.json',dict(fit_donors=sorted(fit),held_donors=sorted(held),coefficients=model[-1].coef_.tolist(),intercept=model[-1].intercept_.tolist(),spline_knots=model[0].bsplines_[0].t.tolist(),spline_degree=model[0].bsplines_[0].k,iterations=model[-1].n_iter_.tolist(),fitting_only_transforms=True))
 oof=pd.concat(predictions,ignore_index=True);assert len(oof)==1793 and oof.donor_id.is_unique
 oof.to_csv(OUT/'private_OOF_predictions.csv',index=False)
 for fold,d in [*oof.groupby('fold'),('pooled_OOF_secondary',oof)]:
  ctr=d[d.label==0];aucs={}
  for typ,g in d[d.label==1].groupby('cancer_types'):
   sub=pd.concat([g,ctr]);aucs[typ]=roc_auc_score(sub.label,sub.probability);per.append(dict(fold=fold,cancer_type=typ,AUROC=aucs[typ],cases=len(g),controls=len(ctr)))
  metrics.append(dict(fold=fold,pooled_AUROC=roc_auc_score(d.label,d.probability),fixed_nine_macro_AUROC=np.mean([aucs[t] for t in TYPES]),cases=int(d.label.eq(1).sum()),controls=len(ctr)))
 r=pd.DataFrame(metrics);r.to_csv(OUT/'fold_metrics.csv',index=False);pd.DataFrame(per).to_csv(OUT/'per_cancer_metrics.csv',index=False)
 mean=r[r.fold.ne('pooled_OOF_secondary')];pooled=r.iloc[-1]
 row=dict(family='count',endpoint='primary',mean_fold_pooled_AUROC=mean.pooled_AUROC.mean(),mean_fold_fixed_nine_macro_AUROC=mean.fixed_nine_macro_AUROC.mean(),minimum_fold_AUROC=mean.pooled_AUROC.min(),maximum_fold_AUROC=mean.pooled_AUROC.max(),pooled_OOF_AUROC_secondary=pooled.pooled_AUROC,pooled_OOF_macro_secondary=pooled.fixed_nine_macro_AUROC,model='Count-based model',parameters=6)
 pd.DataFrame([row]).to_csv(OUT/'model_comparison.csv',index=False)
 (OUT/'README.md').write_text('Matched deterministic count-only baseline on exact1793reference-policy training donors/folds. Predictor is total actual retained events across primarygroup slides,after all3filters,log1p then fitting-only cubicSplineTransformer(n_knots3,degree3,include_biasTrue). L2LogRegC1,classbalanced,maxiter10000,seed17,5coefficients+intercept=6parameters. Three deterministic fits,not artificial duplicates across2seeds. Same heldpatients as neural Table2;meanfoldpooled/fixed9macro primary,pooledOOFsecondary. No testaccess/refittingofneuralmodels/gates/normalizers. Do not confuse spline learned knots with6regressionparameters;5spline basis functions plusintercept matches priorcount baseline specification.\n')
 write(OUT/'receipt.json',dict(completed_utc=utc(),registration_sha256=sha(ROOT/'registration.json'),outputs_sha256={str(p):sha(p) for p in OUT.iterdir() if p.suffix in ['.csv','.md']},test_used=False,parameters=6))
 write(ROOT/'complete.json',dict(completed_utc=utc(),fits=3,test_used=False,receipt_sha256=sha(OUT/'receipt.json')))
 print(r.to_string(index=False));print(pd.DataFrame([row]).to_string(index=False))

if __name__=='__main__':main()
