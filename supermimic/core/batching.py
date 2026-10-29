"""Gradient accumulation keeps reference effective batches within GPU memory."""
import math
import torch


def learning_rate(base, update, decay_updates=250000):
    t=min(max(update,0),decay_updates)/decay_updates
    return base*(.03+.97*.5*(1+math.cos(math.pi*t)))


def world_model_update(model, optimizer, window, loss_fn, microbatch):
    batch=window['action'].shape[0]
    optimizer.zero_grad(set_to_none=True)
    losses=[]
    for start in range(0,batch,microbatch):
        end=min(start+microbatch,batch)
        part={k:v[start:end] for k,v in window.items()}
        loss=loss_fn(part)[0]
        (loss*((end-start)/batch)).backward()
        losses.append(loss.detach()*((end-start)/batch))
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return torch.stack(losses).sum()


def accumulate_imagination(engine, starts, actor, microbatch):
    batch=starts['state'].shape[0]
    pieces=[]
    for start in range(0,batch,microbatch):
        end=min(start+microbatch,batch)
        r=engine({k:v[start:end] for k,v in starts.items()})
        (-r['objective']*((end-start)/batch)).backward()
        pieces.append((end-start,{k:v.detach() for k,v in r.items()}))
    out={}
    for key in ('critic_inputs','rewards','continuations','imagined_actions'):
        out[key]=torch.cat([r[key] for _,r in pieces],dim=1)
    for key in ('objective','task_return','recovery_mode_fraction','style_score'):
        out[key]=sum(count*r[key] for count,r in pieces)/batch
    return out
