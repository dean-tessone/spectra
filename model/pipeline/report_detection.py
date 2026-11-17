"""Score only sealed ensembles and report new-cohort detection without tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression
from runtime import *
from train import setup,verify_registration


def operating(d,model,endpoint):
    fpr,tpr,threshold=roc_curve(d.label,d.probability,drop_intermediate=False)
    roc=pd.DataFrame(dict(false_positive_rate=fpr,true_positive_rate=tpr,threshold=threshold,model=model,endpoint=endpoint))
    rows=[]
    for spec in [.90,.95]:
        allowed=np.flatnonzero(fpr<=1-spec+1e-12)
        best=allowed[np.argmax(tpr[allowed])];cut=float(threshold[best]);pred=d.probability.ge(cut)
        for typ in ['Overall',*sorted(d.loc[d.label.eq(1),'cancer_types'].unique())]:
            cases=d.label.eq(1)&(True if typ=='Overall' else d.cancer_types.eq(typ));control=d.label.eq(0)
            tp=int((pred&cases).sum());tn=int((~pred&control).sum())
            rows.append(dict(model=model,endpoint=endpoint,cancer_type=typ,target_specificity=spec,threshold=cut,true_positives=tp,cases=int(cases.sum()),sensitivity=tp/int(cases.sum()),true_negatives=tn,controls=int(control.sum()),achieved_specificity=tn/int(control.sum())))
    return roc,pd.DataFrame(rows)


def diagnostic(d,model,endpoint,parameters):
    rows=[];controls=d[d.label.eq(0)];types=sorted(d.loc[d.label.eq(1),'cancer_types'].unique());macro=[]
    for typ in ['Overall',*types]:
        x=d if typ=='Overall' else pd.concat([controls,d[d.label.eq(1)&d.cancer_types.eq(typ)]])
        auc=float(roc_auc_score(x.label,x.probability));n=int(x.label.sum())
        if typ!='Overall' and n>=10:macro.append(auc)
        rows.append(dict(model=model,endpoint=endpoint,cancer_type=typ,cases=n,controls=len(controls),auroc=auc,parameters=parameters,macro_eligible=typ!='Overall' and n>=10))
    return rows,dict(model=model,endpoint=endpoint,parameters=parameters,pooled_auroc=rows[0]['auroc'],macro_auroc=float(np.mean(macro)),macro_cancer_types=len(macro))


def main():
    verify_registration();setup(17)
    selected=read(ROOT/'selection.json')['selected'];recipe=selected['recipe']
    families=['transformer','abmil','additive','mean','clam_sb','clam_mb']
    if (ROOT/'locks/transformer_fixed15.json').exists():families.append('transformer_fixed15')
    for family in families:
        lock=read(ROOT/f'locks/{family}.json');assert lock['registration_sha256']==sha(ROOT/'registration.json')
        assert sha(lock['normalizer'])==lock['normalizer_sha256']
        for s,r in lock['models'].items():assert sha(r['checkpoint'])==r['sha256']
    ledger=pd.read_csv(ROOT/'cohort/donor_ledger.csv');test=set(ledger[ledger.split.eq('test')].donor_id);train=set(ledger[ledger.split.eq('train')].donor_id);assert not test&train
    cache=Cache(test)
    with np.load(ROOT/'data/contexts/full/normalization.npz') as z:mean,scale=[torch.tensor(z[k],device='cuda') for k in ['mean','scale']]
    out=ROOT/'detection';out.mkdir(exist_ok=True);tables=[];comparisons=[];rocs=[];ops=[]
    for family in families:
        lock=read(ROOT/f'locks/{family}.json');ds=[]
        for seed,r in lock['models'].items():
            model_family='transformer' if family=='transformer_fixed15' else family
            model=model_for(model_family,recipe).cuda();state=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
            assert state['epoch']==lock['epochs'];model.load_state_dict(state['model'],strict=True)
            d=score(model,model_family,cache,mean,scale,primary=False);d['seed']=seed;ds.append(d);d.to_csv(out/f'{family}_seed_{seed}_group_predictions.csv',index=False)
            parameters=sum(p.numel() for p in model.parameters());del model;torch.cuda.empty_cache()
        all_seed=pd.concat(ds,ignore_index=True);assert all_seed.groupby('group_id').size().eq(3).all()
        avg=all_seed.groupby(['group_id','donor_id','label','cancer_types','study','primary_evaluation'],as_index=False).agg(probability=('probability','mean'),retained_cells=('retained_cells','first'))
        primary=avg[avg.primary_evaluation].copy();assert set(primary.donor_id)==test and primary.donor_id.is_unique
        supplemental=avg.groupby(['donor_id','label','cancer_types','study'],as_index=False).probability.mean()
        for name,d in [('earliest',primary),('all_control_groups',supplemental)]:
            d.to_csv(out/f'{family}_{name}_donor_predictions.csv',index=False)
            rows,comparison=diagnostic(d,family,name,parameters);tables.extend(rows);comparisons.append(comparison)
            roc,op=operating(d,family,name);rocs.append(roc);ops.append(op)
        write(ROOT/'status.json',dict(phase='sealed_ensemble_test_reporting',completed_family=family,updated_utc=utc()))
    # Fitting-only count model, five cubic-spline coefficients plus intercept.
    receipt=read(ROOT/'data/cache_receipt.json')['slides'];manifest=pd.read_csv(ROOT/'cohort/all_groups.csv',dtype={'slide_id':str})
    counts=manifest[manifest.primary_evaluation].copy();counts['count']=counts.slide_id.map(lambda s:receipt[s]['retained_cells'])
    counts=counts.groupby(['donor_id','label','split','cancer_types','study'],as_index=False)['count'].sum()
    fit=counts[counts.split.eq('train')];held=counts[counts.split.eq('test')].copy()
    model=make_pipeline(SplineTransformer(n_knots=3,degree=3,include_bias=True),LogisticRegression(class_weight='balanced',max_iter=10000,random_state=17))
    model.fit(np.log1p(fit[['count']]),fit.label);assert model[-1].coef_.shape==(1,5)
    held['probability']=model.predict_proba(np.log1p(held[['count']]))[:,1]
    held.to_csv(out/'count_earliest_donor_predictions.csv',index=False)
    rows,comparison=diagnostic(held,'count','earliest',6);tables.extend(rows);comparisons.append(comparison);roc,op=operating(held,'count','earliest');rocs.append(roc);ops.append(op)
    write(out/'count_model.json',dict(coefficients=model[-1].coef_.tolist(),intercept=model[-1].intercept_.tolist(),spline_knots=model[0].bsplines_[0].t.tolist(),fit_donors=sorted(train),test_used_for_fitting=False))
    pd.DataFrame(tables).to_csv(out/'per_cancer_auroc.csv',index=False);comparison_table=pd.DataFrame(comparisons);comparison_table['epochs']=comparison_table.model.map(lambda m:selected['epoch'] if m=='transformer' else 0 if m=='count' else 15);comparison_table.to_csv(out/'model_comparison.csv',index=False);pd.concat(rocs,ignore_index=True).to_csv(out/'roc_curves.csv',index=False);pd.concat(ops,ignore_index=True).to_csv(out/'operating_points.csv',index=False)
    primary=next(r for r in comparisons if r['model']=='transformer' and r['endpoint']=='earliest')
    overall=pd.concat(ops);overall=overall[(overall.model=='transformer')&(overall.endpoint=='earliest')&(overall.cancer_type=='Overall')]
    (out/'RESULTS.md').write_text(f'''# Revised-cohort detection

Final held-out endpoint: {len(test)} independent patients, 713 cancer and75 controls. Matched earliest-draw/tube/slides, same policy for both classes. All-control-group averaging is supplementary and separately labeled.

SPECTRA pooled AUROC {primary['pooled_auroc']:.6f}, macro AUROC {primary['macro_auroc']:.6f} across {primary['macro_cancer_types']} cancer types with at least ten held cases. Per-cancer results, ROC coordinates and sensitivity at nominal90%/95% specificity appear in the adjacent CSVs. Operating thresholds are descriptive final-test ROC operating points, not deployment-calibrated cutoffs. No AUROC confidence intervals or test-based model/gate selection.

{overall.to_csv(index=False)}

Patient-model fitting was separated from final evaluation. Frozen upstream encoder/QC exposure limitations remain as documented in UPSTREAM_PROVENANCE.md; this is not external/prospective validation.
''')
    write(out/'complete.json',dict(completed_utc=utc(),registration_sha256=sha(ROOT/'registration.json'),outputs_sha256={str(p):sha(p) for p in out.glob('*') if p.is_file() and p.name!='complete.json'},test_patient_count=len(test),families_sealed_before_test=True))

if __name__=='__main__':main()
