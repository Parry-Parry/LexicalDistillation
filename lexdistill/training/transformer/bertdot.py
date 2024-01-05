from pyterrier_dr import HgfBiEncoder, BiScorer

def make_scorer(model):
    val_backbone = HgfBiEncoder(val_backbone, model.tokenizer, {}, device=model.device)
    return BiScorer(val_backbone, batch_size=64, verbose=False)