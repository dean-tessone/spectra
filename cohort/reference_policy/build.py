"""Reference-control population; no labels inferred or rewritten."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
REPO=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
MAIN=REPO/'runs/paper_regeneration_20261109_v1a'
REFERENCE={'NBD','EPIC','INTC17-C','PANC-Ctrl'}
LUNG={'SU','JN2','JN2-R'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ledger_path=MAIN/'cohort/donor_ledger.csv'
 ledger=pd.read_csv(ledger_path)
 ledger['primary_reference_control']=ledger.label.eq(0)&ledger.study.isin(REFERENCE)
 ledger['primary_cancer_vs_reference_eligible']=ledger.label.eq(1)|ledger.primary_reference_control
 ledger['control_population']=np.select([ledger.label.eq(1),ledger.study.eq('PANC-Ctrl'),ledger.study.isin(['NBD','EPIC','INTC17-C']),ledger.study.isin(LUNG),ledger.study.eq('INTC09')],['Cancer','Non-cancer pancreatic workup, included in primary reference','Healthy reference','Lung diagnostic workup, separate comparison','BRCA-mutant breast risk, separate comparison'],default='Unclassified, not primary reference')
 assert ledger.loc[ledger.label.eq(0),'study'].isin(REFERENCE|LUNG|{'INTC09'}).all()
 ledger.to_csv(OUT/'population_patient_ledger.csv',index=False)
 counts=ledger.groupby(['split','label','control_population']).size().rename('patients').reset_index()
 counts.to_csv(OUT/'population_counts.csv',index=False)
 eligible=ledger[ledger.primary_cancer_vs_reference_eligible]
 assert eligible.groupby(['split','label']).size().to_dict()=={('test',0):44,('test',1):713,('train',0):104,('train',1):1689}
 eligible.to_csv(OUT/'primary_donor_ledger.csv',index=False)
 slides_path=MAIN/'cohort/all_groups.csv';slides=pd.read_csv(slides_path,dtype={'slide_id':str})
 slides=slides.merge(ledger[['donor_id','primary_reference_control','primary_cancer_vs_reference_eligible','control_population']],on='donor_id',validate='many_to_one')
 slides.to_csv(OUT/'all_population_slides.csv',index=False)
 slides[slides.primary_cancer_vs_reference_eligible].to_csv(OUT/'primary_population_slides.csv',index=False)
 A=REPO/'analyses/paper_regeneration_20261109_v1a/performance_driver_audit'
 old_path=A/'private_historical_primary_matched_predictions.csv';old=pd.read_csv(old_path)
 receipt=json.loads((A/'receipt.json').read_text());assert sha(old_path)==receipt['outputs_sha256'][old_path.name]
 new_path=MAIN/'detection/transformer_earliest_donor_predictions.csv';new=pd.read_csv(new_path)
 old=old.rename(columns={'revised_label':'label'})
 rows=[];ops=[];private=[]
 for model,frame in [('Historical frozen transformer',old),('Revised all-control transformer',new)]:
  d=frame.merge(ledger[['donor_id','primary_reference_control','control_population','split']],on='donor_id',validate='one_to_one',suffixes=('','_ledger'))
  assert len(d)==788 and d.donor_id.is_unique
  d['model']=model;private.append(d)
  cases=d[d.label.eq(1)];ref=d[d.primary_reference_control]
  comparisons=[('Overall cancer vs primary reference',cases,ref)]
  for typ,g in cases.groupby('cancer_types'):
   comparisons.append((f'{typ} vs primary reference',g,ref))
  comparisons += [('Pancreatic vs PANC-Ctrl',cases[cases.cancer_types.eq('Pancreatic')],d[(d.label==0)&d.study.eq('PANC-Ctrl')]),('PANCEUS vs PANC-Ctrl',cases[cases.study.eq('PANCEUS')],d[(d.label==0)&d.study.eq('PANC-Ctrl')]),('Lung vs pooled lung-workup controls',cases[cases.cancer_types.eq('Lung')],d[(d.label==0)&d.study.isin(LUNG)]),('Breast vs INTC09 BRCA-mutant controls',cases[cases.cancer_types.eq('Breast')],d[(d.label==0)&d.study.eq('INTC09')])]
  for study in sorted(LUNG):comparisons.append((f'Lung vs {study} controls',cases[cases.cancer_types.eq('Lung')],d[(d.label==0)&d.study.eq(study)]))
  for name,ca,co in comparisons:
   y=[1]*len(ca)+[0]*len(co);s=pd.concat([ca.probability,co.probability]);auc=roc_auc_score(y,s)
   rows.append(dict(model=model,comparison=name,cases=len(ca),controls=len(co),AUROC=auc))
   # Each explicitly distinct population has its own descriptive ROC operating point.
   # These are not prospectively calibrated cutoffs or thresholds transported from the mixed endpoint.
   fpr,tpr,th=roc_curve(y,s,drop_intermediate=False)
   for spec in [.9,.95]:
    good=np.flatnonzero(fpr<=1-spec+1e-12);j=good[np.argmax(tpr[good])]
    ops.append(dict(model=model,comparison=name,target_specificity=spec,threshold=th[j],achieved_specificity=1-fpr[j],sensitivity=tpr[j],true_positives=int(ca.probability.ge(th[j]).sum()),cases=len(ca),true_negatives=int(co.probability.lt(th[j]).sum()),controls=len(co),threshold_scope='Descriptive ROC within stated comparison'))
 r=pd.DataFrame(rows);r.to_csv(OUT/'frozen_model_population_AUROC.csv',index=False)
 pd.DataFrame(ops).to_csv(OUT/'frozen_model_population_operating_points.csv',index=False)
 # Macro uses the preexisting >=10 held-cases eligibility rule, not which types score well.
 macro=[]
 for model in r.model.unique():
  per=r[r.model.eq(model)&r.comparison.str.endswith(' vs primary reference')&~r.comparison.str.startswith('Overall')&r.cases.ge(10)]
  macro.append(dict(model=model,macro_AUROC=per.AUROC.mean(),eligible_cancer_types=len(per),definition='Unweighted mean over cancer types with at least10 retained final-test cases'))
 pd.DataFrame(macro).to_csv(OUT/'frozen_model_macro_AUROC.csv',index=False)
 pd.concat(private).to_csv(OUT/'private_frozen_predictions.csv',index=False)
 text='''# reference-control policy, 2026-11-10

Primary endpoint: all eligible retained cancer patients versus NBD, EPIC, INTC17-C and PANC-Ctrl controls. INTC09 and SU/JN2/JN2-R (lung-workup controls) are excluded from general-source control fitting and primary reference evaluation, but retained as distinct populations for specialized evaluation/adaptation. PANC-Ctrl is included while explicitly identified as non-cancer pancreatic-workup, rather than claiming all reference subjects are healthy volunteers.

Counts: training1689cancer/104reference controls; final713cancer/44reference controls. Primary population2550patients. Separate populations include72training/31final-test controls. Main case eligibility, donor folds, primary earliest-draw/tube/up-to2slide endpoint, nativefloat32 all3cell filters and technical source exclusions are unchanged. Additional reference control bags may be used for donor-balanced training, never increasing donor counts. Prior cohorts/results remain historical, not overwritten.

Frozen-model tables use cached old primary-group scores and new all-control-model primary scores on matched revised test donors. Old weights were trained under historical labels/eligibility; new all-control weights include the now-reserved controls in training. Neither is a fresh fit of this policy. In particular, removing patients from evaluation does not remove them from old training weights. New source fits must initialize freshly and recompute fitting-only QC-primary equal-donor normalizers. No reuse of old patient-model weights for replacement source fitting.

Report overall pooled AUROC, per-cancer AUROC against the common reference population, and unweighted macro over cancer types with>=10held cases. Report pancreatic versus PANC-Ctrl, lung versus pooled/per-study lung-workup controls, and breast versus INTC09 separately. Available MM precursor/state strata remain distinct, frozen case labels unchanged; no excluded MGUS donor is automatically reinstated. Cases compared against specialized controls from different studies are explicitly cross-cohort. Matched PANCEUS versus PANC-Ctrl is separately shown.

Each distinct diagnostic comparison can have its own ROC-derived90/95%specificity operating points, with achieved specificity and numerators/denominators. These are descriptive test ROC estimates, not train-CV calibration. Stage/state case subgroups within one comparison must use that comparison's single common control threshold. No threshold shared numerically across old/new models. No AUROC CIs requested.

The broad713/75mixed-control result remains preserved. No causal effect, confirmed malignant-cell identity, or prognostic equivalence is established by discrimination or attribution.
'''
 (OUT/'POLICY.md').write_text(text)
 inputs=[ledger_path,slides_path,old_path,new_path,Path(__file__)]
 (OUT/'receipt.json').write_text(json.dumps(dict(inputs_sha256={str(p):sha(p) for p in inputs},primary_control_studies=sorted(REFERENCE),reserved_lung_controls=sorted(LUNG),reserved_breast_controls=['INTC09'],frozen_labels_modified=False),indent=2)+'\n')
 print(counts.to_string(index=False));print(r[r.comparison.isin(['Overall cancer vs primary reference','Pancreatic vs PANC-Ctrl','Lung vs pooled lung-workup controls','Breast vs INTC09 BRCA-mutant controls'])].to_string(index=False));print(pd.DataFrame(macro).to_string(index=False))

if __name__=='__main__':main()
