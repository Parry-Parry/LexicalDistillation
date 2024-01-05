from pyterrier_dr import ElectraScorer

def make_scorer(model):
    scorer = ElectraScorer(batch_size=64, verbose=False, device=model.device)
    # transfer state dict
    scorer.model.load_state_dict(model.state_dict())
    return scorer