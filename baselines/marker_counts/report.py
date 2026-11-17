"""Frozen-phenotype/CK count reporting; no SPECTRA refit or new detector scoring."""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json,shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import ROOT as MAIN,read,write,sha,utc
ROOT=REPO/'runs/bluev2_ctc_comparison_20261110_v1b';OUT=REPO/'analyses/bluev2_ctc_comparison_20261110_v1b'
CLASSES=['Large Nuclei','imCTCs','CTCs','pcCTCs','CECs','Megakaryocytes','Fibroblasts','Lymph','Gran','Mono']


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 receipt=read(ROOT/'counts_complete.json')
 for p,h in receipt['outputs_sha256'].items():assert sha(p)==h
 counts=pd.read_csv(ROOT/'slide_counts.csv',dtype={'slide_id':str})
 manifest=pd.read_csv(MAIN/'cohort/all_groups.csv',dtype={'slide_id':str})
 primary=manifest[manifest.primary_evaluation][['slide_id','donor_id']]
 assert primary.slide_id.is_unique and not counts.duplicated(['slide_id','pool']).any()
 assert len(counts)==manifest.slide_id.nunique()
 fields=['candidate_events','QC_rejected','assessed_events','CTC_like','CK_gate','CTC_like_and_CK',*CLASSES,*['CK_and_'+x for x in CLASSES]]
 assert counts[CLASSES].sum(1).eq(counts.assessed_events).all()
 assert counts[['imCTCs','CTCs','pcCTCs']].sum(1).eq(counts.CTC_like).all()
 joint=primary.merge(counts,on='slide_id',validate='one_to_many')
 patients=joint.groupby(['donor_id','pool'],as_index=False).agg(slides=('slide_id','nunique'),**{k:(k,'sum') for k in fields})
 for c in ['CTC_like','CK_gate',*CLASSES]:patients[c+'_per_slide']=patients[c]/patients.slides
 ledger=pd.read_csv(REPO/'cohort/reference_policy/population_patient_ledger.csv')
 patients=patients.merge(ledger,on='donor_id',validate='many_to_one')
 assert len(patients)==2653 and patients.slides.between(1,2).all()
 patients.to_csv(OUT/'private_primary_patient_counts.csv',index=False)
 rows=[];ops=[];curves=[];markercols=['CTC_like','CK_gate',*CLASSES]
 def evaluate(pool,split,name,cases,controls,per_cancer=True):
  assert cases.donor_id.is_unique and controls.donor_id.is_unique and not set(cases.donor_id)&set(controls.donor_id)
  if not len(cases) or not len(controls):return
  groups=[('Overall',cases)]
  if per_cancer:groups += list(cases.groupby('cancer_types'))
  for marker in markercols:
   for typ,ca in groups:
    y=np.r_[np.ones(len(ca)),np.zeros(len(controls))];scores=np.r_[ca[marker+'_per_slide'],controls[marker+'_per_slide']]
    auc=roc_auc_score(y,scores)
    rows.append(dict(pool=pool,partition=split,comparison=name,marker=marker,cancer_type=typ,cases=len(ca),controls=len(controls),AUROC=auc,primary_marker=marker=='CTC_like',score='Assigned events per unique assessed primary slide'))
   # Single control-derived ROC threshold per comparison, reused across case subgroups.
   fpr,tpr,threshold=roc_curve(np.r_[np.ones(len(cases)),np.zeros(len(controls))],np.r_[cases[marker+'_per_slide'],controls[marker+'_per_slide']],drop_intermediate=False)
   if marker in ['CTC_like','CK_gate']:
    curves.extend([dict(pool=pool,partition=split,comparison=name,marker=marker,FPR=float(f),TPR=float(t),threshold=float(h)) for f,t,h in zip(fpr,tpr,threshold)])
   for target in [.9,.95]:
    valid=np.flatnonzero(fpr<=1-target+1e-12);j=valid[np.argmax(tpr[valid])];h=threshold[j]
    for typ,ca in groups:
     tp=int(ca[marker+'_per_slide'].ge(h).sum());tn=int(controls[marker+'_per_slide'].lt(h).sum())
     ops.append(dict(pool=pool,partition=split,comparison=name,marker=marker,cancer_type=typ,target_specificity=target,threshold=h,true_positives=tp,cases=len(ca),sensitivity=tp/len(ca),true_negatives=tn,controls=len(controls),achieved_specificity=tn/len(controls)))
 for (pool,split),p in patients.groupby(['pool','split']):
  cases=p[p.label.eq(1)];controls=p[p.primary_reference_control]
  evaluate(pool,split,'Primary cancer vs reference',cases,controls)
  evaluate(pool,split,'Pancreatic vs PANC-Ctrl',cases[cases.cancer_types.eq('Pancreatic')],p[p.label.eq(0)&p.study.eq('PANC-Ctrl')],False)
  evaluate(pool,split,'PANCEUS vs PANC-Ctrl',cases[cases.study.eq('PANCEUS')],p[p.label.eq(0)&p.study.eq('PANC-Ctrl')],False)
  evaluate(pool,split,'Lung vs pooled lung-workup',cases[cases.cancer_types.eq('Lung')],p[p.label.eq(0)&p.study.isin(['SU','JN2','JN2-R'])],False)
  evaluate(pool,split,'Breast vs INTC09',cases[cases.cancer_types.eq('Breast')],p[p.label.eq(0)&p.study.eq('INTC09')],False)
 metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'marker_detection_AUROC.csv',index=False)
 pd.DataFrame(ops).to_csv(OUT/'marker_operating_points.csv',index=False);pd.DataFrame(curves).to_csv(OUT/'marker_ROC_curves.csv',index=False)
 # Macro includes all cancer types with at least10cases within stated partition, independent of scores.
 eligible=metrics[metrics.cancer_type.ne('Overall')&metrics.cases.ge(10)]
 macro=eligible.groupby(['pool','partition','comparison','marker'],as_index=False).agg(macro_AUROC=('AUROC','mean'),cancer_types=('cancer_type','nunique'));macro.to_csv(OUT/'marker_macro_AUROC.csv',index=False)
 distribution=patients.groupby(['pool','split','label','control_population'],as_index=False).agg(patients=('donor_id','nunique'),assessed_slides=('slides','sum'),**{c:(c,'sum') for c in ['assessed_events','CTC_like','CK_gate','CTC_like_and_CK',*CLASSES]})
 distribution.to_csv(OUT/'population_phenotype_totals.csv',index=False)
 # Predefined CK threshold robustness remains separate from primary40k/10k.
 grid=pd.read_csv(ROOT/'slide_CK_gate_robustness.csv',dtype={'slide_id':str})
 g=primary.merge(grid,on='slide_id',validate='one_to_many').groupby(['donor_id','CK_threshold','CY5_threshold'],as_index=False).agg(slides=('slide_id','nunique'),technical_QC_gate=('technical_QC_gate','sum'),model_input_gate=('model_input_gate','sum'))
 g=g.merge(ledger[['donor_id','label','split','primary_cancer_vs_reference_eligible']],on='donor_id',validate='many_to_one');robust=[]
 for (split,ck,cy),part in g[g.primary_cancer_vs_reference_eligible].groupby(['split','CK_threshold','CY5_threshold']):
  for pool in ['model_input']:
   robust.append(dict(partition=split,CK_threshold=ck,CY5_threshold=cy,pool=pool,primary_gate=ck==40000 and cy==10000,cases=int(part.label.eq(1).sum()),controls=int(part.label.eq(0).sum()),AUROC=roc_auc_score(part.label,part[pool+'_gate']/part.slides)))
 pd.DataFrame(robust).to_csv(OUT/'CK_gate_robustness_AUROC.csv',index=False)
 # Matched existing primary SPECTRA OOF, no new detector inference or patient fitting.
 oof_path=REPO/'analyses/reference_control_refit_20261110/private_OOF_predictions.csv';oof=pd.read_csv(oof_path)
 cv=[]
 for pool in ['model_input']:
  p=patients[patients.pool.eq(pool)&patients.split.eq('train')&patients.primary_cancer_vs_reference_eligible]
  d=p[['donor_id','label','CTC_like_per_slide','CK_gate_per_slide']].merge(oof[['donor_id','fold','label','probability']],on='donor_id',validate='one_to_one',suffixes=('','_model'))
  assert len(d)==1793 and d.label.eq(d.label_model).all()
  for fold,part in [*d.groupby('fold'),('pooled_OOF_secondary',d)]:
   for marker,col in [('CTC_like','CTC_like_per_slide'),('CK_gate','CK_gate_per_slide'),('SPECTRA','probability')]:cv.append(dict(pool=pool,fold=fold,marker=marker,AUROC=roc_auc_score(part.label,part[col]),cases=int(part.label.eq(1).sum()),controls=int(part.label.eq(0).sum())))
 pd.DataFrame(cv).to_csv(OUT/'training_CV_matched_comparison.csv',index=False)
 note='''# Frozen BLUEV2 CTC-like comparator

All6049selectedphysicalslides enumerated once from authenticated local raw512D/QC caches. Primary biomarker score is combined argmax classesCTCs/imCTCs/pcCTCs per unique assessed primaryslide only in the exact all3filter model-input pool. Cells failing artifactQC,leukocyte or brightness filters are not classified; raw per-patient counts and every ten-class count retained locally. CK comparator uses strict40kTRITC/10kCY5; ninegate robustness separate. Do not select/invert scores using outcomes. These are candidate-pool model-assigned phenotypes, not confirmed malignantcells or whole-blood prevalence.

Primary evaluation713cases/44referencecontrols, training1689/104. Referencecontrols:NBD,EPIC,INTC17-C,PANC-Ctrl; PANC-Ctrl is non-cancer pancreatic work-up. Lungworkup SU/JN2/JN2-R andINTC09BRCA-risk controls remain separately labeled. Separate diagnostic comparisons use their own ROC operating points; per-cancer subgroups against the common reference share the overall control threshold. Specificity is discrete with44controls; report actualdenominators/rates. No AUROC CIs/calibration. Patient score raw count/assessedslides, no nominalbloodvolume assumption.

Marker test reporting uses externally frozen phenotype classifier and CKgate; no new SPECTRA test scoring or parameter fitting. SPECTRA comparison is existing donor-exclusive primary trainingCV, shown as such, not its new test result. Scaler/classifier frozen; exact package prediction function reused,5source/raw row alignments and chunk equivalence checked;allslideinputindices/CKcounts reconciled with existing authenticated caches.

Configured BLUEencoder fingerprint matches phenotype package, but per-parquet generation identity/exact encoder training identities remain unverified. Referenceclassifier scaler used all referencecells before cell-level holdout;referencepatient overlap unverified. Its0.926cellholdoutaccuracy is not independentpatientvalidation. Original predictions are model assignments, not groundtruth. Molecularvalidation remainsfuturework.
'''
 (OUT/'README.md').write_text(note)
 outputs=[OUT/n for n in ['marker_detection_AUROC.csv','marker_operating_points.csv','marker_ROC_curves.csv','marker_macro_AUROC.csv','population_phenotype_totals.csv','CK_gate_robustness_AUROC.csv','training_CV_matched_comparison.csv','README.md']]
 write(OUT/'receipt.json',dict(completed_utc=utc(),registration_sha256=sha(ROOT/'registration.json'),counts_receipt_sha256=sha(ROOT/'counts_complete.json'),inputs_sha256={str(oof_path):sha(oof_path)},outputs_sha256={str(p):sha(p) for p in outputs},SPECTRA_test_inference=False,new_patient_fits=False))
 dst=Path(os.environ.get('SPECTRA_SHARED_OUT', 'shared_outputs')) / 'bluev2_ctc_comparison_20261110_v1b';dst.mkdir(parents=True,exist_ok=True);hashes={}
 for a in [*outputs,OUT/'receipt.json']:
  b=dst/a.name
  if b.exists():assert a.read_bytes()==b.read_bytes(),'Version conflicting shared output'
  else:shutil.copyfile(a,b)
  assert sha(a)==sha(b);hashes[a.name]=sha(b)
 write(OUT/'delivery_receipt.json',dict(destination=str(dst),outputs_sha256=hashes))
 print(metrics[metrics.partition.eq('test')&metrics.comparison.eq('Primary cancer vs reference')&metrics.cancer_type.eq('Overall')&metrics.marker.isin(['CTC_like','CK_gate'])].to_string(index=False))

if __name__=='__main__':main()
