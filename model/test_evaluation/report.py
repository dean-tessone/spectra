"""Aggregate fixed, sealed test predictions; no fitting or retuning."""
import os
from pathlib import Path
import sys,shutil
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import read,write,sha,utc
from report_detection import diagnostic,operating
ROOT=REPO/'runs/reference_control_test_20261110_v1a';OUT=REPO/'analyses/reference_control_test_20261110_v1a'
PARAMETERS=4870913


def pairwise_auc(d):
 a=d[d.label==1].probability.to_numpy();b=d[d.label==0].probability.to_numpy()
 return float(((a[:,None]>b[None,:])+.5*(a[:,None]==b[None,:])).mean())

def main():
 OUT.mkdir(parents=True,exist_ok=True);receipt=read(ROOT/'inference_complete.json');reg=read(ROOT/'registration.json')
 for p,h in receipt['outputs_sha256'].items():assert sha(p)==h
 for p,h in reg['inputs_sha256'].items():assert sha(p)==h
 ledger=pd.read_csv(REPO/'cohort/reference_policy/population_patient_ledger.csv')
 clinical_path=REPO/'analyses/clinical_cohort_annotations_20261110/patient_clinical_ledger.csv';clinical=pd.read_csv(clinical_path,usecols=['donor_id','breast_early_late','breast_TNM','breast_T','breast_N','breast_M','mm_state'])
 metrics=[];summaries=[];curves=[];operating_points=[];stage_metrics=[];stage_ops=[];primary=None;verification=[]
 for endpoint in ['primary','strict_same_draw_tube_supplemental','all_eligible_groups_supplemental']:
  d=pd.read_csv(ROOT/f'private_{endpoint}_donor_predictions.csv')
  d=d.merge(ledger[['donor_id','primary_cancer_vs_reference_eligible','primary_reference_control','control_population']],on='donor_id',validate='one_to_one')
  assert len(d)==788 and d.donor_id.is_unique
  reference=d[d.primary_cancer_vs_reference_eligible].copy();assert len(reference)==757 and reference.label.eq(1).sum()==713 and reference.label.eq(0).sum()==44
  if endpoint=='primary':primary=reference.copy()
  cases=d[d.label==1]
  comparisons=[('Cancer vs primary reference',reference),('Broad mixed controls supplemental',d),('Pancreatic vs PANC-Ctrl',d[(d.label.eq(1)&d.cancer_types.eq('Pancreatic'))|(d.label.eq(0)&d.study.eq('PANC-Ctrl'))]),('PANCEUS vs PANC-Ctrl',d[(d.label.eq(1)&d.study.eq('PANCEUS'))|(d.label.eq(0)&d.study.eq('PANC-Ctrl'))]),('Lung vs pooled lung-workup',d[(d.label.eq(1)&d.cancer_types.eq('Lung'))|(d.label.eq(0)&d.study.isin(['SU','JN2','JN2-R']))]),('Breast vs INTC09',d[(d.label.eq(1)&d.cancer_types.eq('Breast'))|(d.label.eq(0)&d.study.eq('INTC09'))])]
  for study in ['SU','JN2','JN2-R']:comparisons.append((f'Lung vs {study}',d[(d.label.eq(1)&d.cancer_types.eq('Lung'))|(d.label.eq(0)&d.study.eq(study))]))
  for name,sub in comparisons:
   assert sub.label.nunique()==2
   detail,summary=diagnostic(sub,'SPECTRA',endpoint,PARAMETERS)
   exact=pairwise_auc(sub);assert abs(exact-summary['pooled_auroc'])<1e-12
   verification.append(dict(endpoint=endpoint,comparison=name,independent_pairwise_AUROC=exact,passed=True))
   for row in detail:row['comparison']=name
   summary['comparison']=name;metrics.extend(detail);summaries.append(summary)
   roc,op=operating(sub,'SPECTRA',endpoint);roc['comparison']=name;op['comparison']=name;curves.append(roc);operating_points.append(op)
   if name=='Cancer vs primary reference':
    annotated=sub.merge(clinical,on='donor_id',how='left',validate='one_to_one');ctrl=annotated[annotated.label==0]
    fields=[('Breast','breast_early_late'),('Breast','breast_TNM'),('Breast','breast_T'),('Breast','breast_N'),('Breast','breast_M'),('Multiple Myeloma','mm_state')]
    for cancer,field in fields:
     ca=annotated[(annotated.label==1)&annotated.cancer_types.eq(cancer)].copy();ca[field]=ca[field].fillna('Unknown')
     for value,g in ca.groupby(field,dropna=False):
      combined=pd.concat([g,ctrl]);auc=roc_auc_score(combined.label,combined.probability)
      stage_metrics.append(dict(endpoint=endpoint,stratification=field,stratum=value,cancer_type=cancer,cases=len(g),controls=len(ctrl),AUROC=auc))
      for point in op[op.cancer_type.eq('Overall')].itertuples():
       tp=int(g.probability.ge(point.threshold).sum());tn=int(ctrl.probability.lt(point.threshold).sum())
       stage_ops.append(dict(endpoint=endpoint,stratification=field,stratum=value,cancer_type=cancer,target_specificity=point.target_specificity,threshold=point.threshold,true_positives=tp,cases=len(g),sensitivity=tp/len(g),true_negatives=tn,controls=len(ctrl),achieved_specificity=tn/len(ctrl)))
 pd.DataFrame(metrics).to_csv(OUT/'per_cancer_AUROC.csv',index=False);summary=pd.DataFrame(summaries);summary.to_csv(OUT/'endpoint_summary.csv',index=False)
 pd.concat(curves,ignore_index=True).to_csv(OUT/'ROC_curves.csv',index=False);op=pd.concat(operating_points,ignore_index=True);op.to_csv(OUT/'operating_points.csv',index=False)
 pd.DataFrame(stage_metrics).to_csv(OUT/'clinical_stratum_AUROC.csv',index=False);pd.DataFrame(stage_ops).to_csv(OUT/'clinical_stratum_operating_points.csv',index=False)
 # Paired primary marker comparison: no phenotype/gate refitting, same757donors.
 marker_path=REPO/'analyses/bluev2_ctc_comparison_20261110_v1b/private_primary_patient_counts.csv';marker=pd.read_csv(marker_path);marker=marker[marker.pool.eq('model_input')&marker.split.eq('test')&marker.primary_cancer_vs_reference_eligible]
 joined=primary[['donor_id','label','probability']].merge(marker[['donor_id','label','CTC_like_per_slide','CK_gate_per_slide']],on='donor_id',validate='one_to_one',suffixes=('','_marker'))
 assert len(joined)==757 and joined.label.eq(joined.label_marker).all()
 markers=[]
 for name,column in [('SPECTRA','probability'),('BLUEV2 CTC-like burden','CTC_like_per_slide'),('CK40k/CY510k burden','CK_gate_per_slide')]:
  values=joined[['donor_id','label',column]].rename(columns={column:'probability'});auc=roc_auc_score(values.label,values.probability);assert abs(auc-pairwise_auc(values))<1e-12
  markers.append(dict(model=name,cases=713,controls=44,pooled_AUROC=auc,endpoint='primary',new_model_fitting=False))
 pd.DataFrame(markers).to_csv(OUT/'paired_primary_marker_comparison.csv',index=False)
 main=summary[summary.endpoint.eq('primary')&summary.comparison.eq('Cancer vs primary reference')].iloc[0]
 primary_op=op[op.endpoint.eq('primary')&op.comparison.eq('Cancer vs primary reference')&op.cancer_type.eq('Overall')]
 (OUT/'RESULTS.md').write_text(f'''# Sealed SPECTRA reference-policy test result

Selected six-layer15epoch ensemble,seeds17/29/43. Full fitting completed and sealed before this evaluation; no duplicate refit or test-based tuning. Primary713cancer/44reference controls,one predefined earliest eligible primary group per patient. PooledAUROC{main.pooled_auroc:.6f};macroAUROC{main.macro_auroc:.6f} across{int(main.macro_cancer_types)}cancer types with>=10testcases.

{primary_op.to_csv(index=False)}

Reference controls are NBD,EPIC,INTC17-C,PANC-Ctrl;PANC-Ctrl; PANC-Ctrl is non-cancer pancreatic work-up. Reserved lungworkup/BRCA-risk controls appear only in separately labeled diagnostic comparisons or broadmixed supplemental endpoint. Strictdraw/tube versus all-eligible-group averaging are separate supplemental endpoints; differing sampling effort remainsexplicit. Each comparison has its own descriptive ROC90/95%specificity operating points; cancer/stage subgroups within a comparison share the overall control threshold. No AUROC CIs or trainCV calibration. Breast/MMan notation coverage incomplete, clinical timing unconfirmed; no inferredM0/AJCCstage.

Fixed-budget matched MILCV completed before test; those baselines were not automatically refit/tested and cannot be compared as new testmodels. Marker comparison uses frozen all3filter counts on identical757donors. No gate/class/score inversion selected from results. Cell assignments not confirmedmalignantcells.

Frozen upstream encoder/QC have documented historical exposure, per-parquet encoder identity/exact historical fitting donor identities unverified. This is patient-model held-out evaluation under the recorded protocol. Source lock remains immutable; its test_evaluatedfalse field records the seal-time state, while this run receipt records subsequent authorized testing.
''')
 write(OUT/'independent_verification.json',dict(pairwise_auc_checks=verification,primary_cases=713,primary_controls=44,all_test_donors=788,marker_pairing_passed=True,seed_replay=receipt['primary_replay'],model_lock_unchanged=True))
 public=[OUT/n for n in ['per_cancer_AUROC.csv','endpoint_summary.csv','ROC_curves.csv','operating_points.csv','clinical_stratum_AUROC.csv','clinical_stratum_operating_points.csv','paired_primary_marker_comparison.csv','RESULTS.md','independent_verification.json']]
 write(OUT/'receipt.json',dict(completed_utc=utc(),registration_sha256=sha(ROOT/'registration.json'),inference_receipt_sha256=sha(ROOT/'inference_complete.json'),outputs_sha256={str(p):sha(p) for p in public},model_lock_sha256=receipt['model_lock_sha256'],new_fitting=False,test_evaluated=True))
 dst=Path(os.environ.get('SPECTRA_SHARED_OUT', 'shared_outputs')) / 'reference_control_test_20261110_v1a';dst.mkdir(parents=True,exist_ok=True);hashes={}
 for a in [*public,OUT/'receipt.json']:
  b=dst/a.name
  if b.exists():assert a.read_bytes()==b.read_bytes(),'Version conflicting shared output'
  else:shutil.copyfile(a,b)
  assert sha(a)==sha(b);hashes[a.name]=sha(b)
 write(OUT/'delivery_receipt.json',dict(destination=str(dst),outputs_sha256=hashes))
 print(summary.to_string(index=False));print(primary_op.to_string(index=False))

if __name__=='__main__':main()
