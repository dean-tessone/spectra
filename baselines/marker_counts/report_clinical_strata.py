"""Secondary breast clinical strata for frozen CTC-like/CK burden scores."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from common import read,write,sha,utc
OUT=REPO/'analyses/bluev2_ctc_comparison_20261110_v1b'
CLIN=REPO/'analyses/clinical_stratification_20261110/test_patient_clinical_ledger.csv'

def main():
 countpath=OUT/'private_primary_patient_counts.csv';op_path=OUT/'marker_operating_points.csv'
 p=pd.read_csv(countpath);p=p[p.pool.eq('model_input')&p.split.eq('test')]
 fields=['breast_early_late','breast_TNM','breast_T','breast_N','breast_M']
 clinical=pd.read_csv(CLIN,usecols=['donor_id',*fields])
 assert clinical.donor_id.is_unique and p.donor_id.is_unique
 p=p.merge(clinical,on='donor_id',validate='one_to_one',how='left')
 for field in fields:p[field]=p[field].fillna('Unknown')
 breast=p[p.label.eq(1)&p.cancer_types.eq('Breast')];controls=p[p.primary_reference_control]
 assert len(breast)==218 and len(controls)==44
 ops=pd.read_csv(op_path);ops=ops[ops.partition.eq('test')&ops.pool.eq('model_input')&ops.comparison.eq('Primary cancer vs reference')&ops.cancer_type.eq('Overall')&ops.marker.isin(['CTC_like','CK_gate'])]
 assert len(ops)==4
 rows=[];operating=[]
 for field in fields:
  for value,ca in breast.groupby(field,dropna=False):
   for marker in ['CTC_like','CK_gate']:
    scores=np.r_[ca[marker+'_per_slide'],controls[marker+'_per_slide']];labels=np.r_[np.ones(len(ca)),np.zeros(len(controls))]
    auc=roc_auc_score(labels,scores)
    rows.append(dict(stratification=field,stratum=value,marker=marker,cases=len(ca),controls=len(controls),AUROC=auc,score='Assigned events per unique assessed primary slide'))
    for op in ops[ops.marker.eq(marker)].itertuples():
     tp=int(ca[marker+'_per_slide'].ge(op.threshold).sum());tn=int(controls[marker+'_per_slide'].lt(op.threshold).sum())
     operating.append(dict(stratification=field,stratum=value,marker=marker,target_specificity=op.target_specificity,threshold=op.threshold,true_positives=tp,cases=len(ca),sensitivity=tp/len(ca),true_negatives=tn,controls=len(controls),achieved_specificity=tn/len(controls),threshold_scope='Frozen overall cancer vs reference threshold; no stratum retuning'))
 r=pd.DataFrame(rows);o=pd.DataFrame(operating)
 r.to_csv(OUT/'breast_stage_CTC_vs_CK_AUROC.csv',index=False);o.to_csv(OUT/'breast_stage_CTC_vs_CK_operating_points.csv',index=False)
 table=r[r.stratification.eq('breast_early_late')].pivot(index=['stratum','cases','controls'],columns='marker',values='AUROC').reset_index().rename(columns={'CTC_like':'CTC_like_AUROC','CK_gate':'CK_gate_AUROC'})
 for marker in ['CTC_like','CK_gate']:
  for target in [.9,.95]:
   d=o[o.stratification.eq('breast_early_late')&o.marker.eq(marker)&o.target_specificity.eq(target)][['stratum','sensitivity','true_positives','achieved_specificity']].rename(columns={c:f'{marker}_{c}_at_{int(target*100)}spec_target' for c in ['sensitivity','true_positives','achieved_specificity']})
   table=table.merge(d,on='stratum',validate='one_to_one')
 assert table.cases.sum()==218
 table.to_csv(OUT/'breast_early_late_CTC_vs_CK_test.csv',index=False)
 coverage=breast.groupby('breast_early_late').size().rename('test_patients').reset_index();coverage.to_csv(OUT/'breast_stage_annotation_coverage.csv',index=False)
 (OUT/'BREAST_STAGE_NOTES.md').write_text('Secondary breast strata use supplied Early/Late labels and literal recorded TNM/T/N/M; no AJCC inference or missing-M0 assumption. Same218held breastcases,44referencecontrols,onlyall3filtercells. Every stratum uses the original overall marker thresholds for90/95%specificity; no stage-specific threshold tuning.50Early,10Late,1Conflicting,157Unknown. Only60/218have unambiguous Early/Late annotations; coverage is incomplete/cohort-dependent, small subgroups descriptive. Clinical timing unconfirmed. Missing/Conflicting retained as explicit rows, not inferred from scores or study identity. CTC-like=model-assignedCTCs+imCTCs+pcCTCs,not molecularlyconfirmed malignantcells. MarkerAUROC assesses diagnosis,not prognostic separation. No new SPECTRA test inference or fitting.\n')
 write(OUT/'breast_stage_receipt.json',dict(completed_utc=utc(),inputs_sha256={str(q):sha(q) for q in [countpath,op_path,CLIN,Path(__file__)]},test_breast_cases=218,reference_controls=44,clinical_thresholds_retuned=False))
 print(table[['stratum','cases','CTC_like_AUROC','CK_gate_AUROC']].to_string(index=False))

if __name__=='__main__':main()
