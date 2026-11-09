"""Validate the token vocabulary shared by pretokenized data and target model.

Padding uses the tokenizer that actually produced the frozen dataset. Some
Pythia checkpoints omit the Python tokenizer EOS attribute while retaining
the same backend vocabulary and declaring EOS in the model configuration.
No IDs are guessed or silently retokenized.
"""
import hashlib,json
def load_tokenizer(model_name,revision,manifest):
    from transformers import AutoConfig,AutoTokenizer
    source=AutoTokenizer.from_pretrained(manifest['tokenizer_model'],revision=manifest['tokenizer_revision'],token=False)
    target=AutoTokenizer.from_pretrained(model_name,revision=revision,token=False)
    cfg=AutoConfig.from_pretrained(model_name,revision=revision,token=False)
    a=json.loads(source.backend_tokenizer.to_str());b=json.loads(target.backend_tokenizer.to_str())
    keys=('model','normalizer','pre_tokenizer','decoder','added_tokens')
    left={k:a.get(k) for k in keys};right={k:b.get(k) for k in keys}
    if left!=right or source.get_vocab()!=target.get_vocab():raise ValueError('Target tokenizer is incompatible with frozen token IDs')
    eos=source.eos_token_id;declared=cfg.eos_token_id
    allowed=declared if isinstance(declared,list) else [declared]
    if not isinstance(eos,int) or eos<0 or source.convert_ids_to_tokens(eos)!=target.convert_ids_to_tokens(eos):
        raise ValueError('Data padding token mapping is not shared by target')
    if max(source.get_vocab().values())>=cfg.vocab_size:raise ValueError('Data tokenizer exceeds model vocabulary')
    metadata={'data_tokenizer_model':manifest['tokenizer_model'],'data_tokenizer_revision':manifest['tokenizer_revision'],
        'target_tokenizer_model':model_name,'target_revision':revision,'data_padding_eos_id':eos,
        'target_tokenizer_eos_attribute':target.eos_token_id,'model_config_eos':declared,
        'generation_eos_differs_from_data_eos':eos not in allowed,
        'generation_used':False,
        'encoding_backend_sha256':hashlib.sha256(json.dumps(left,sort_keys=True).encode()).hexdigest(),
        'scope':'identical vocabulary, merges, added tokens, normalizer, pre-tokenizer and decoder; frozen data IDs unchanged; generation EOS metadata is unused by teacher-forced loss and all padding/EOS labels are ignored'}
    return source,metadata
