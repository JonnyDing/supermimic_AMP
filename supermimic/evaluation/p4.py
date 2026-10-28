"""Independent native-load paired full-P4 evaluation; supports optimized checkpoints."""
import argparse,json
from pathlib import Path
import torch
from supermimic.core.policy import ROOT, load_policy
from supermimic.evaluation.scenarios import fixed_recovery_frames
from supermimic.evaluation.bptt import run_trial
from supermimic.evaluation.acceptance import json_safe_metrics
from supermimic.mdp.amp import DifferentiableAMP
from supermimic.evaluation.metrics import full_summary,acceptance


def evaluate_checkpoint(candidate,output,seeds=(82,83,84),seconds=20,device='cuda:0',push=True):
    if not seeds or seconds<1:raise ValueError('Need nonempty seeds and at least one second of evaluation')
    output=Path(output)
    if output.exists():raise ValueError('Use a new evaluation output to preserve previous results')
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_float32_matmul_precision('highest')
    source=ROOT/'logs/rsl_rl/g1_amp_locomotion/2026-10-24_13-24-33/model_66600.pt'
    original=torch.load(source,map_location=device,weights_only=False)
    baseline,bnorm=load_policy(source,device);policy,norm=load_policy(candidate,device)
    judge=DifferentiableAMP(original,device);_,data,frames=fixed_recovery_frames();tests=[]
    settings=[(seed,None) for seed in seeds]+([(seeds[0],250)] if push else [])
    if push and seconds<=5:raise ValueError('Push evaluation needs >5 seconds')
    for seed,push_step in settings:
        pair={'seed':seed,'push_step':push_step}
        for name,actor,normalizer in [('baseline',baseline,bnorm),('candidate',policy,norm)]:
            try:
                trial=run_trial(seed,actor,normalizer,data,frames,device,seconds,full_amp=judge,push_step=push_step)
                pair[name]={'summary':full_summary(trial),'trial':trial}
            except RuntimeError as error:
                if not str(error).startswith('Policy produced non-finite actions'):raise
                pair[name]={'error':str(error),'status':'invalid_numerical_trial'}
                print(json.dumps({'seed':seed,'policy':name,'error':str(error)}),flush=True)
        if any('error' in pair[name] for name in ('baseline','candidate')):
            pair['comparability']={'passed':False,'checks':{'valid_paired_trial':False}}
            tests.append(pair)
            clean,bad=json_safe_metrics(tests)
            output.with_suffix('.progress.json').write_text(json.dumps({'tests':clean,'nonfinite_metrics':bad},indent=2,allow_nan=False))
            continue
        for key in ('context_sha256','initial_state_sha256'):
            if pair['baseline']['trial'][key]!=pair['candidate']['trial'][key]:
                output.with_suffix('.unpaired.json').write_text(json.dumps(pair,indent=2))
                raise AssertionError(f'Unpaired {key}')
        pair['comparability']=acceptance(pair['baseline']['summary'],pair['candidate']['summary']);tests.append(pair)
        print(json.dumps({'seed':seed,'push_step':push_step,'comparability':pair['comparability']}),flush=True)
        clean,bad=json_safe_metrics(tests)
        output.with_suffix('.progress.json').write_text(json.dumps({'tests':clean,'nonfinite_metrics':bad},indent=2,allow_nan=False))
    groups={}
    for label,pairs in [('fixed',[p for p in tests if p['push_step'] is None]),('push',[p for p in tests if p['push_step'] is not None])]:
        if not pairs:continue
        invalid=[p for p in pairs if any('error' in p[n] for n in ('baseline','candidate'))]
        pairs=[p for p in pairs if all('summary' in p[n] for n in ('baseline','candidate'))]
        if not pairs:
            groups[label]={'aggregate':None,'invalid_pairs':len(invalid),'comparability':{'passed':False,'checks':{'valid_paired_trials':False}}}
            continue
        aggregate={n:{k:sum(p[n]['summary'][k] for p in pairs)/len(pairs) for k in pairs[0][n]['summary']} for n in ('baseline','candidate')}
        groups[label]={'aggregate':aggregate,'comparability':acceptance(aggregate['baseline'],aggregate['candidate']),'valid_pairs':len(pairs),'invalid_pairs':len(invalid)}
        if invalid:
            groups[label]['comparability']['passed']=False
            groups[label]['comparability']['checks']['valid_paired_trials']=False
    result={'candidate':str(candidate),'seeds':list(seeds),'seconds':seconds,'groups':groups,'tests':tests,
            'passed':all(g['comparability']['passed'] for g in groups.values())}
    result,bad=json_safe_metrics(result);result['nonfinite_metrics']=bad
    if bad:result['passed']=False
    output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'passed':result['passed'],'groups':groups}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[82,83,84]);p.add_argument('--seconds',type=int,default=20);p.add_argument('--device',default='cuda:0');p.add_argument('--no-push',action='store_true')
    a=p.parse_args();evaluate_checkpoint(a.candidate,a.output,a.seeds,a.seconds,a.device,not a.no_push)


if __name__=='__main__':main()
