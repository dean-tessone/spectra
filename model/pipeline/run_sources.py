"""Fresh target-excluded sources over authenticated revised main assay data."""
from pathlib import Path
import sys,os,random,time,subprocess,fcntl,argparse,traceback,shutil
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'model/pipeline'))
from runtime import *
from common import ROOT as MAIN
ROOT=REPO/'runs/paper_downstream_20261109_v1a/transfer_fits'
PREP=REPO/'runs/paper_downstream_20261109_v1a'
SEEDS=(17,29)

def verify_registration():
    r=read(ROOT/'registration.json')
    for p,h in r['inputs_sha256'].items():
        if sha(p)!=h:raise ValueError(f'Transfer source input changed:{p}')
    return r

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'registration.json').exists():return verify_registration()
    assert (MAIN/'main_detection_complete.json').exists() and (PREP/'attribution/complete.json').exists()
    upstream=read(MAIN/'registration.json')
    for p,h in upstream['inputs_sha256'].items():assert sha(p)==h
    receipt=read(PREP/'preparation.json')
    for section in ['source_inputs_sha256','outputs_sha256']:
        for p,h in receipt[section].items():assert sha(p)==h
    choice=read(MAIN/'selection.json')['selected'];files=[Path(__file__),MAIN/'selection.json',MAIN/'registration.json',MAIN/'preparation_complete.json',PREP/'preparation.json',PREP/'attribution/complete.json']
    ledger=pd.read_csv(MAIN/'cohort/donor_ledger.csv');train=set(ledger[ledger.split.eq('train')].donor_id);test=set(ledger[ledger.split.eq('test')].donor_id)
    for target in ['Pancreatic','Breast','Lung']:
        targets=set(ledger[(ledger.split=='train')&(ledger.label==1)&ledger.cancer_types.eq(target)].donor_id)
        for fold in range(3):
            source=PREP/f'transfer/{target}/fold_{fold}';p=read(source/'proof_17.json');fit=set(p['source_fit_donors']);held=set(p['held_donors'])
            assert fit<=train and held<=train and not (fit|held)&test and not fit&held and not fit&targets
            folder=ROOT/f'data/contexts/{target}_{fold}';folder.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source/'normalization.npz',folder/'normalization.npz')
            write(folder/'provenance.json',dict(fit_donors=sorted(fit),held_donors=sorted(held),target=target,fold=fold,source_model_initialization='random',normalization_sha256=sha(folder/'normalization.npz'),target_cases_excluded=True,main_test_used=False))
            files += [source/'proof_17.json',source/'proof_29.json',folder/'normalization.npz',folder/'provenance.json']
    write(ROOT/'registration.json',dict(created_utc=utc(),inputs_sha256={str(p):sha(p) for p in files},source_recipe=choice['recipe'],source_epochs=choice['epoch'],source_architecture=RECIPES[choice['recipe']],source_fits=18,folds=3,seeds=list(SEEDS),test_used=False,source_initialization='fresh_random',adaptation_settings='fixed5epochs320updates; separate adaptation registration'))
    return verify_registration()


def save(path,state):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial');torch.save(state,tmp);tmp.replace(p)

def setup(seed):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    free,total=torch.cuda.mem_get_info()
    if free<.20*total+8*1024**3:raise RuntimeError('Insufficient free memory for allocator20% plus8GiB reserve; leave other users alone')
    torch.cuda.set_per_process_memory_fraction(.20)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)

def context(fold):
    proof=read(ROOT/f'data/contexts/{fold}/provenance.json')
    cache=Cache(proof['fit_donors'])
    held=Cache(proof['held_donors']) if proof['held_donors'] else None
    with np.load(ROOT/f'data/contexts/{fold}/normalization.npz') as z:mean,scale=[torch.tensor(z[k],device='cuda') for k in ['mean','scale']]
    return cache,held,mean,scale

def fit(family,recipe,fold,seed,epochs,role):
    record=verify_registration();setup(seed)
    folder=ROOT/f'{role}/{family}/{recipe}/fold_{fold}/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    expected=dict(registration_sha256=sha(ROOT/'registration.json'),family=family,recipe=recipe,fold=str(fold),seed=seed,target_epochs=epochs,role=role,normalization_sha256=sha(ROOT/f'data/contexts/{fold}/normalization.npz'))
    if (folder/'complete.json').exists():
        done=read(folder/'complete.json');assert done['identity']==expected
        for p,h in done['outputs_sha256'].items():assert sha(p)==h
        print('Already complete',folder,flush=True);return
    cache,held,mean,scale=context(fold)
    design=schedule(cache,seed,epochs);write(folder/'schedule.json',design)
    model=model_for(family,recipe).cuda();lr=RECIPES[recipe]['lr'] if family=='transformer' else 2e-4
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lr_factor)
    start=1;history=[];best_score=-float('inf')
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False);assert state['identity']==expected
        model.load_state_dict(state['model'],strict=True);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng_cpu']);torch.cuda.set_rng_state_all(state['rng_cuda']);np.random.set_state(state['rng_numpy']);random.setstate(state['rng_python'])
        start=state['epoch']+1;history=state['history'];best_score=state['best_score']
    milestones={8,15,16,20,25,30,epochs} if role=='selection_cv' else {epochs}
    for epoch in range(start,epochs+1):
        begun=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
        if hasattr(model,'set_training_epoch'):model.set_training_epoch(epoch)
        dl=loader(cache,batches=design[epoch-1]);total_loss=0.;updates=0
        for i,b in enumerate(dl):
            out,mask=forward(model,family,b,mean,scale);labels=b['label'].cuda(non_blocking=True)
            losses=(SB.training_losses(model,out,labels,mask) if family=='clam_sb' else MB.training_losses(model,out,labels,mask) if family=='clam_mb' else F.binary_cross_entropy_with_logits(out['logits'],labels,reduction='none'))
            loss=(losses*b['sample_weight'].cuda(non_blocking=True)).sum()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite weighted loss')
            loss.backward();total_loss+=float(loss.detach())
            if (i+1)%8==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();scheduler.step();opt.zero_grad(set_to_none=True);updates+=1
        del dl
        assert updates==64
        history.append(dict(epoch=epoch,training_loss=total_loss/64,elapsed_seconds=time.monotonic()-begun))
        # Validation never consumes fitting RNG or changes future training draws.
        rngs=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())
        keep=(epoch==epochs)
        if held is not None and epoch in milestones:
            d=score(model,family,held,mean,scale);d.to_csv(folder/f'validation_epoch_{epoch}.csv',index=False)
            m=metrics(d);write(folder/f'metrics_epoch_{epoch}.json',m)
            if m['macro']>best_score:best_score=m['macro'];keep=True
        torch.set_rng_state(rngs[0]);torch.cuda.set_rng_state_all(rngs[1]);np.random.set_state(rngs[2]);random.setstate(rngs[3])
        state=dict(identity=expected,model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,history=history,best_score=best_score,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),rng_numpy=np.random.get_state(),rng_python=random.getstate())
        save(folder/'last.pt',state)
        if keep:save(folder/'best.pt',dict(identity=expected,model=model.state_dict(),epoch=epoch))
        write(folder/'history.json',history);write(folder/'status.json',dict(phase='fitting',epoch=epoch,target_epochs=epochs,updated_utc=utc()))
        print(family,recipe,fold,seed,epoch,round(total_loss/64,6),round(history[-1]['elapsed_seconds'],1),'seconds',flush=True)
    outputs=[folder/'best.pt',folder/'last.pt',folder/'history.json',folder/'schedule.json',*folder.glob('validation_epoch_*.csv'),*folder.glob('metrics_epoch_*.json')]
    write(folder/'complete.json',dict(identity=expected,outputs_sha256={str(p):sha(p) for p in outputs},completed_utc=utc(),test_features_accessed=False))
    write(folder/'status.json',dict(phase='complete',epoch=epochs,updated_utc=utc()))

def queue():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);record=prepare()
        for target in ['Pancreatic','Breast','Lung']:
            for fold in range(3):
                for seed in SEEDS:
                    dest=ROOT/f"source/transformer/{record['source_recipe']}/fold_{target}_{fold}/seed_{seed}";dest.mkdir(parents=True,exist_ok=True)
                    with (dest/'driver.log').open('a') as log:
                        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'fit','--target',target,'--fold',str(fold),'--seed',str(seed)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
                        write(ROOT/'status.json',dict(phase='source_fitting',target=target,fold=fold,seed=seed,controller_pid=os.getpid(),active_pid=child.pid,updated_utc=utc()))
                        if child.wait():raise RuntimeError(f'Target-excluded source failed:{target}/{fold}/{seed}')
        write(ROOT/'complete.json',dict(source_fits=18,registration_sha256=sha(ROOT/'registration.json'),completed_utc=utc()))
        write(ROOT/'status.json',dict(phase='sources_complete_adaptation_next',updated_utc=utc()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','queue','fit']);p.add_argument('--target');p.add_argument('--fold',type=int);p.add_argument('--seed',type=int);a=p.parse_args()
    try:
        if a.action=='prepare':prepare()
        elif a.action=='queue':queue()
        else:
            r=verify_registration();fit('transformer',r['source_recipe'],f'{a.target}_{a.fold}',a.seed,r['source_epochs'],'source')
    except Exception:
        write(ROOT/'status.json',dict(phase='failed',error=traceback.format_exc(),updated_utc=utc()));raise
