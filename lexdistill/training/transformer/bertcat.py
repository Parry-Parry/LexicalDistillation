from pyterrier_dr import ElectraScorer
from typing import Any 

def make_scorer(model : Any, batch_size : int = 64, verbose : bool = False):
    scorer = ElectraScorer(batch_size=batch_size, verbose=verbose, device=model.device)
    # transfer state dict
    scorer.model.load_state_dict(model.state_dict())
    return scorer