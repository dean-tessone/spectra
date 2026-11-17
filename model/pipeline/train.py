"""Fresh bounded fits, with complete resume state and planned CV predictions."""
import argparse
import random
import subprocess
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from runtime import *

def save(path,state):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial');torch.save(state,tmp);tmp.replace(p)

def verify_registration():
    record=read(ROOT/'registration.json')
    for p,h in record['inputs_sha256'].items():
        if sha(p)!=h:raise ValueError(f'Registered campaign input changed: {p}')
    return record

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


def preflight():
    setup(17);cache,held,mean,scale=context('0')
    ids=[g for label in [0,1] for g in sorted([g for g,e in cache.entries.items() if e['label']==label],key=lambda g:sum(cache.entries[g]['lengths']),reverse=True)[:4]]
    batches=[[(g,1/8) for g in ids]]
    batch=next(iter(loader(cache,batches=batches)))
    results={}
    for family in ['transformer','abmil','additive','mean','clam_sb','clam_mb']:
        model=model_for(family,'d6_do04_lr2e4').cuda();model.train()
        out,mask=forward(model,family,batch,mean,scale)
        labels=batch['label'].cuda()
        loss=(SB.training_losses(model,out,labels,mask).mean() if family=='clam_sb' else MB.training_losses(model,out,labels,mask).mean() if family=='clam_mb' else F.binary_cross_entropy_with_logits(out['logits'],labels))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        # Padding and cell order invariance in evaluation, using real cells.
        model.eval();b={**batch,'features':batch['features'][:2,:128].clone(),'attention_mask':batch['attention_mask'][:2,:128].clone()}
        with torch.no_grad():
            a,_=forward(model,family,b,mean,scale);b['features']=b['features'].flip(1);b['attention_mask']=b['attention_mask'].flip(1);z,_=forward(model,family,b,mean,scale)
        error=float((a['logits']-z['logits']).abs().max())
        assert error < (.02 if family=='transformer' else 5e-5)
        fp32_error=None
        if family=='transformer':
            x=((batch['features'][:2,:128].cuda().float()-mean)/scale).half().float();m=batch['attention_mask'][:2,:128].cuda()
            with torch.no_grad():
                one=model(x,m,fp32_attention=True)['logits'];two=model(x.flip(1),m.flip(1),fp32_attention=True)['logits']
                padded=model(torch.cat([x,torch.zeros_like(x[:,:13])],1),torch.cat([m,torch.zeros_like(m[:,:13])],1),fp32_attention=True)['logits']
            fp32_error=float((one-two).abs().max());padding_error=float((one-padded).abs().max())
            assert fp32_error<2e-5 and padding_error<2e-5
        results[family]=dict(fp32_permutation_error=fp32_error,parameters=sum(p.numel() for p in model.parameters()),finite_gradients=True,permutation_error=error)
        del model,out,loss;torch.cuda.empty_cache()
    write(ROOT/'gpu_preflight.json',dict(passed=True,models=results,largest_training_bag_cells=max(sum(cache.entries[g]['lengths']) for g in ids),test_access=False,allocator_fraction=.20))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','preflight']);p.add_argument('--family',default='transformer');p.add_argument('--recipe',default='d6_do04_lr2e4');p.add_argument('--fold',default='0');p.add_argument('--seed',type=int,default=17);p.add_argument('--epochs',type=int,default=30);p.add_argument('--role',default='selection_cv');a=p.parse_args()
    if a.action=='preflight':preflight()
    else:fit(a.family,a.recipe,a.fold,a.seed,a.epochs,a.role)
