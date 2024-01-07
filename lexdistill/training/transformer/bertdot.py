from pyterrier_dr import HgfBiEncoder, BiScorer
from transformers import AutoTokenizer
from typing import Any 

def make_scorer(model : Any, batch_size : int = 64, verbose : bool = False):
    tokenizer = AutoTokenizer.from_pretrained('google/electra-base-discriminator')
    val_backbone = HgfBiEncoder(model.encoder, tokenizer, {}, device=model.device)
    return BiScorer(val_backbone, batch_size=batch_size, verbose=verbose)