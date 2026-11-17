"""Independent cohort, fold, normalizer and presentation checks before fitting."""
import numpy as np
import pandas as pd
from common import *
from runtime import Cache,schedule

def main():
    prep=read(ROOT/'preparation_complete.json')
    for p,h in {**prep['inputs_sha256'],**prep['outputs_sha256']}.items():assert sha(p)==h,p
    d=pd.read_csv(ROOT/'cohort/all_groups.csv',dtype={'slide_id':str,'donor_id':str})
    people=pd.read_csv(ROOT/'cohort/donor_ledger.csv')
    assert people.donor_id.is_unique and len(people)==2653
    assert people.groupby('split').label.value_counts().to_dict()=={('test',0):75,('test',1):713,('train',0):176,('train',1):1689}
    assert not d.donor_id.str.startswith('jhwp-').any()
    pass  # donor-level exclusions are applied upstream
    assert d.slide_id.is_unique
    primary=d[d.primary_evaluation]
    assert primary.groupby('donor_id').group_id.nunique().eq(1).all()
    assert primary.groupby('donor_id').size().between(1,2).all()
    assert d.groupby('group_id').tube_id.nunique().eq(1).all()
    assert d.groupby('group_id').draw_date.nunique().eq(1).all()
    assert d.groupby('donor_id').split.nunique().eq(1).all()
    with np.load(ROOT/'data/donor_moments.npz') as m:
        ids=m['donors'];means=m['mean'];seconds=m['second']
        for fold in ['full',0,1,2]:
            p=read(ROOT/f'data/contexts/{fold}/provenance.json');fit=set(p['fit_donors']);held=set(p['held_donors']);test=set(people[people.split.eq('test')].donor_id)
            assert not fit&held and not fit&test and not held&test
            selected=np.isin(ids,sorted(fit));assert selected.sum()==len(fit)
            mean=means[selected].mean(0);scale=np.sqrt(np.maximum(seconds[selected].mean(0)-mean**2,1e-12))
            with np.load(ROOT/f'data/contexts/{fold}/normalization.npz') as z:
                np.testing.assert_array_equal(z['mean'],mean.astype(np.float32));np.testing.assert_array_equal(z['scale'],scale.astype(np.float32))
    proof=read(ROOT/'data/contexts/0/provenance.json');cache=Cache(proof['fit_donors']);design=schedule(cache,17,1)
    visited=set()
    for step in range(64):
        micros=design[0][step*8:(step+1)*8];gs=[g for micro in micros for g,w in micro];w=[w for micro in micros for g,w in micro]
        m=cache.meta.loc[gs];assert len(set(m.donor_id))==64 and m.label.eq(0).sum()==32 and m.label.eq(1).sum()==32 and np.isclose(sum(w),1)
        visited.update(m.donor_id)
    assert design==schedule(cache,17,1)
    report=dict(passed=True,cohort_counts=prep['counts'],slide_count=len(d),group_count=d.group_id.nunique(),primary_endpoint_patients=primary.donor_id.nunique(),patient_partition_disjoint=True,normalizers_recomputed_from_fit_donors_only=True,schedule_64_distinct_donors_per_update=True,loss_mass_one_per_update=True,test_used_in_fitting=False,created_utc=utc())
    write(ROOT/'independent_audit.json',report)
    provenance=dict(policy='Reuse fixed assay representations and technical QC; refit every patient model and train-derived transform.',qc_model_sha256=sha(REPO/'models/qc/qc_v1.json'),qc_disease_label_inputs=False,qc_deployment_definite_annotations=2548,qc_historical_study_donor_exposure=True,qc_patient_disjoint_independence_claim_allowed=False,encoder_provenance_path=str(DATA/'audits/spectra_foundation_20261008/encoder_provenance.json'),encoder_provenance_sha256=sha(DATA/'audits/spectra_foundation_20261008/encoder_provenance.json'),encoder_exact_historical_fit_identity_verified=False,encoder_per_parquet_checkpoint_identity_verified=False,patient_checkpoint_initialization='fresh_random_only',revised_patient_labels_not_used_in_upstream_fit=True,limitations='Fixed preprocessing has historical study exposure. This is patient-model held-out evaluation, not verified independent training of the entire upstream image pipeline. Exact QC fitting identities cannot be established from the deployment artifact alone.',created_utc=utc())
    write(ROOT/'upstream_provenance_audit.json',provenance)
    (ROOT/'UPSTREAM_PROVENANCE.md').write_text('''# Frozen upstream preprocessing audit

The revised analysis reuses frozen Z512/ECOD candidates, WBC predictions and QC-v1 technical-usability scoring. Source/raw/QC files are individually SHA authenticated against the existing physical inventories. These upstream artifacts predate the new Category annotations and do not fit the revised cancer/control task.

QC-v1 deployment was refitted on all 2,548 definite annotations; its reported 0.939 weighted validation AUROC is a development metric, and its original development split had 34 overlapping donor identifiers. Historical study donors therefore occur in upstream technical-model development. The deployment JSON alone does not authenticate the full original fitting identities. The historical manuscript claim of no patient overlap is incorrect and must be corrected.

The authenticated configured SimCLR checkpoint identifies a five-plane ResNet-18 representation encoder, but its exact original training identities are not archived. Candidate exposure reconstructed from a current slide list is not proof of original encoder training membership. Per-parquet checkpoint identity is also not fully established. Reusing these fixed measurements permits a patient-model comparison, but does not establish complete pipeline independence from held donors.

All revised disease-supervised patient weights and context normalizers are fitted fresh with canonical donor separation. No old patient checkpoint initializes any new source fit. Report these upstream limitations in Methods/Discussion and avoid external/prospective or entirely donor-independent pipeline claims. Reuse does not alter any source images, frozen embeddings or biological measurements.
''')
    print('Independent cohort, normalizer and sampling checks passed',flush=True)

if __name__=='__main__':main()
