from pyterrier_dr import HgfBiEncoder, BiScorer
from typing import Any 

def make_scorer(model : Any, batch_size : int = 64, verbose : bool = False):
    val_backbone = HgfBiEncoder(model, model.tokenizer, {}, device=model.device)
    return BiScorer(val_backbone, batch_size=batch_size, verbose=verbose)