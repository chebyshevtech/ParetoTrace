"""Development-selected, shared attention arithmetic for matched comparisons.

This changes forward probabilities as well as backward rounding. It is applied
to every arm, and its forward/backward costs remain inside timed operations.
Only the inspected GPTNeoX eager implementation is supported.
"""
from contextlib import contextmanager
import hashlib
import inspect

@contextmanager
def shared_attention_precision(enabled=True):
    from transformers.models.gpt_neox import modeling_gpt_neox as neox
    original=neox.eager_attention_forward
    source=inspect.getsource(original)
    old='dtype=torch.float32).to(query.dtype)'
    if source.count(old)!=1:
        raise RuntimeError('Unrecognized attention implementation; inspect before patching')
    modified=source.replace(old,'dtype=torch.float64).to(query.dtype)')
    namespace=dict(neox.__dict__)
    exec(compile(modified,'<shared_softmax64>','exec'),namespace)
    chosen=namespace['eager_attention_forward'] if enabled else original
    metadata={'softmax_dtype':'float64' if enabled else 'float32',
        'original_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'executed_source_sha256':hashlib.sha256((modified if enabled else source).encode()).hexdigest(),
        'calls':0,'scope':'shared by all methods; forward and backward arithmetic change'}
    def counted(*args,**kwargs):
        metadata['calls']+=1
        return chosen(*args,**kwargs)
    neox.eager_attention_forward=counted
    try:yield metadata
    finally:neox.eager_attention_forward=original
