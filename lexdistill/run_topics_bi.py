import pyterrier as pt
pt.init()
import fire 
import os
from os.path import join
from pyterrier_dr import HgfBiEncoder, BiScorer
from transformers import ElectraModel, ElectraTokenizer

def create_bi_encoder(model_path : str):
    val_backbone = ElectraModel.from_pretrained(model_path)
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    val_backbone = HgfBiEncoder(val_backbone, tokenizer, {}, device='cuda')
    val_model = BiScorer(val_backbone, batch_size=64, verbose=False)
    return val_model

def main(eval : str, run_dir : str, out_dir : str, baseline : str = None, model : str = None, index : str = 'msmarco_passage', dataset : str = 'irds:msmarco-passage/train/triples-small'):  
    dataset = pt.get_dataset(dataset)
    bm25 = pt.BatchRetrieve(pt.get_dataset(index).get_index("terrier_stemmed_text"), wmodel="BM25")
    eval = pt.get_dataset(eval)
    os.makedirs(out_dir, exist_ok=True)
    if baseline : 
        baseline = bm25 >> pt.text.get_text(dataset, "text") >> create_bi_encoder(join(baseline, 'model'))
        if not os.path.exists(join(out_dir, "baseline_run.gz")):
            res = baseline.transform(eval.get_topics())
            pt.io.write_results(res, join(out_dir, "baseline_run.gz"))
    if not model:
        dirs = [f for f in os.listdir(run_dir) if os.path.isdir(join(run_dir, f)) and 'baseline' not in f]
        for _, store in enumerate(dirs):
            if os.path.exists(join(out_dir, f"{store}_run.gz")):
                continue
            try:
                _model = bm25 >> pt.text.get_text(dataset, "text") >> create_bi_encoder(join(run_dir, store, 'model'))
            except OSError:
                print(f"Failed to load {store}")
                continue
            res = _model.transform(eval.get_topics())
            pt.io.write_results(res, join(out_dir, f"{store}_run.gz"))
            del _model
    else:
        _model = bm25 >> pt.text.get_text(dataset, "text") >> create_bi_encoder(model_name=join(run_dir, model, 'model'))
        res = _model.transform(eval.get_topics())
        pt.io.write_results(res, join(out_dir, f"{model}_run.gz"))
    return "Success!"

if __name__ == '__main__':
    fire.Fire(main) 