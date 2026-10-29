"""Atomic full-state checkpoints; resume starts a new simulator episode boundary."""
import hashlib,os,random
from pathlib import Path
import numpy as np
import torch


def rng_state():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
            'numpy':np.random.get_state(),'python':random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    np.random.set_state(state['numpy']);random.setstate(state['python'])


def atomic_save(value,path):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
