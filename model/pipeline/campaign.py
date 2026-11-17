"""Serial authorized main-model rebuild with sealed final-test boundary."""
import fcntl
import os
import subprocess
import sys
import traceback
import numpy as np
import pandas as pd
from runtime import *

SEEDS=(17,29,43)
FAMILIES=['transformer','abmil','additive','mean','clam_sb','clam_mb']

def register():
    inputs=[*sorted(Path(__file__).parent.glob('*.py')),ROOT/'preparation_complete.json',ROOT/'independent_audit.json',ROOT/'upstream_provenance_audit.json',ROOT/'gpu_preflight.json',ROOT/'code_validation.json',REPO/'models/qc/qc_v1.json']
    inputs += [Path(p) for p in read(ROOT/'preparation_complete.json')['outputs_sha256']]
    inputs += sorted((Path(__file__).parent/'tests').glob('*.py'))
    inputs += [Path(m.__file__) for m in [MODEL,MIL,MEAN,SB,MB]]
    record=dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in inputs},recipes=RECIPES,cv_seeds=[17,29],refit_seeds=list(SEEDS),cv_folds=3,recipe_endpoints=[8,16,30],duration_endpoints=[15,20,25],retain_previously_eligible_recipe_endpoint=True,test_retuning=False,cohort_revision='revision_20261109',historical_patient_weights_reused=False,model_selection_endpoint='matched earliest donor',families=FAMILIES)
    if (ROOT/'registration.json').exists():
        old=read(ROOT/'registration.json');assert old['inputs_sha256']==record['inputs_sha256'];return old
    assert read(ROOT/'independent_audit.json')['passed'] and read(ROOT/'gpu_preflight.json')['passed']
    write(ROOT/'registration.json',record);return record


def launch(family,recipe,fold,seed,epochs,role):
    dest=ROOT/f'{role}/{family}/{recipe}/fold_{fold}/seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(Path(__file__).parent/'train.py'),'fit','--family',family,'--recipe',recipe,'--fold',str(fold),'--seed',str(seed),'--epochs',str(epochs),'--role',role]
    with (dest/'driver.log').open('a') as log:
        child=subprocess.Popen(cmd,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        write(ROOT/'status.json',dict(phase=role,family=family,recipe=recipe,fold=str(fold),seed=seed,target_epochs=epochs,controller_pid=os.getpid(),active_pid=child.pid,log=str(dest/'driver.log'),updated_utc=utc()))
        if child.wait():raise RuntimeError(f'Fit failed: {dest}; inspect driver.log')


def endpoint(recipe,epoch,role='selection_cv',seeds=(17,29)):
    folds=[];all_predictions=[]
    for fold in range(3):
        ds=[]
        for seed in seeds:
            p=ROOT/f'{role}/transformer/{recipe}/fold_{fold}/seed_{seed}/validation_epoch_{epoch}.csv'
            d=pd.read_csv(p);d['seed']=seed;ds.append(d)
        stacked=pd.concat(ds,ignore_index=True)
        assert stacked.groupby('donor_id').size().eq(len(seeds)).all()
        averaged=stacked.groupby(['donor_id','label','cancer_types','study'],as_index=False).probability.mean()
        folds.append(metrics(averaged));averaged['fold']=fold;all_predictions.append(averaged)
    return dict(recipe=recipe,epoch=epoch,macro=float(np.mean([m['macro'] for m in folds])),balanced_log_loss=float(np.mean([m['balanced_log_loss'] for m in folds])),mean_fold_auroc=float(np.mean([m['pooled_auroc'] for m in folds])),pooled_oof_auroc=metrics(pd.concat(all_predictions))['pooled_auroc'],fold_metrics=folds)


def main():
    ROOT.mkdir(parents=True,exist_ok=True);handle=(ROOT/'controller.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    register()
    if not (ROOT/'selection.json').exists():
        order=['d6_do04_lr2e4']+[k for k in RECIPES if k!='d6_do04_lr2e4']
        for recipe in order:
            for fold in range(3):
                for seed in (17,29):launch('transformer',recipe,fold,seed,30,'selection_cv')
        candidates=[endpoint(recipe,epoch) for recipe in RECIPES for epoch in [8,16,30]]
        finalists=[min([r for r in candidates if r['recipe']==recipe],key=lambda r:(-r['macro'],r['balanced_log_loss'],r['epoch'])) for recipe in RECIPES]
        peak=max(r['macro'] for r in finalists)
        choice=min([r for r in finalists if r['macro']>=peak-.005],key=lambda r:(r['balanced_log_loss'],RECIPES[r['recipe']]['depth'],r['epoch'],RECIPES[r['recipe']]['dropout'],r['recipe']))
        duration=[endpoint(choice['recipe'],epoch) for epoch in sorted({choice['epoch'],15,20,25})]
        selected=min(duration,key=lambda r:(-r['macro'],r['balanced_log_loss'],r['epoch']))
        write(ROOT/'selection.json',dict(selected=selected,recipe_choice=choice,recipe_candidates=candidates,duration_candidates=duration,selection_used_test=False,registration_sha256=sha(ROOT/'registration.json')))
    selected=read(ROOT/'selection.json')['selected'];recipe=selected['recipe'];epochs=selected['epoch']
    # Fresh fixed-duration CV checkpoints supply donor-excluded attribution and
    # third-seed confirmation. No test scoring or new selection occurs here.
    for fold in range(3):
        for seed in SEEDS:launch('transformer',recipe,fold,seed,epochs,'selected_cv')
    write(ROOT/'selected_cv_complete.json',dict(metrics=endpoint(recipe,epochs,'selected_cv',SEEDS),duration_fixed_before_confirmation=True,completed_utc=utc()))
    for family in FAMILIES:
        for seed in SEEDS:launch(family,recipe,'full',seed,epochs if family=='transformer' else 15,'full_refit')
        files=[ROOT/f'full_refit/{family}/{recipe}/fold_full/seed_{s}/best.pt' for s in SEEDS]
        write(ROOT/f'locks/{family}.json',dict(family=family,recipe=recipe,epochs=epochs if family=='transformer' else 15,models={str(s):dict(checkpoint=str(p),sha256=sha(p)) for s,p in zip(SEEDS,files)},normalizer=str(ROOT/'data/contexts/full/normalization.npz'),normalizer_sha256=sha(ROOT/'data/contexts/full/normalization.npz'),registration_sha256=sha(ROOT/'registration.json'),sealed_utc=utc(),test_used=False))
    if epochs!=15:
        for seed in SEEDS:launch('transformer',recipe,'full',seed,15,'matched15_reference')
        files=[ROOT/f'matched15_reference/transformer/{recipe}/fold_full/seed_{s}/best.pt' for s in SEEDS]
        write(ROOT/'locks/transformer_fixed15.json',dict(family='transformer',display_name='transformer_fixed15',recipe=recipe,epochs=15,models={str(s):dict(checkpoint=str(p),sha256=sha(p)) for s,p in zip(SEEDS,files)},normalizer=str(ROOT/'data/contexts/full/normalization.npz'),normalizer_sha256=sha(ROOT/'data/contexts/full/normalization.npz'),registration_sha256=sha(ROOT/'registration.json'),sealed_utc=utc(),test_used=False))
    # Only the new versioned evaluator is allowed past this barrier.
    subprocess.run([sys.executable,'-u',str(Path(__file__).parent/'report_detection.py')],cwd=REPO,check=True)
    write(ROOT/'main_detection_complete.json',dict(completed_utc=utc(),remaining=['attribution_method_confirmation_and_atlas','CK_removal_and_robustness','target_excluded_transfer_and_finetuning','supplementary_survival','figures_and_manuscript_numerical_audit']))
    write(ROOT/'status.json',dict(phase='main_detection_complete_downstream_pending',updated_utc=utc()))

if __name__=='__main__':
    try:main()
    except Exception:
        write(ROOT/'status.json',dict(phase='failed',error=traceback.format_exc(),updated_utc=utc()));raise
