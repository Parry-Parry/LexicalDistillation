import pyterrier as pt
pt.init()
import fire 
import os
from os.path import join
from lexdistill.lsr.transformer import LSR
from lexdistill.lsr.models import TransformerMLMSparseEncoder, DualSparseEncoder
from lexdistill.lsr.models.mlm import TransformerMLMConfig
from lexdistill.lsr.tokenizer import HFTokenizer

def make_lsr(model_path : str):
    config = TransformerMLMConfig(tf_base_model_name_or_dir=model_path)
    model = DualSparseEncoder(query_encoder=TransformerMLMSparseEncoder(config), doc_encoder=TransformerMLMSparseEncoder(config))
    tokenizer = HFTokenizer.from_pretrained('google/electra-base-discriminator')
    return LSR(model, tokenizer, device='cuda', batch_size=64, verbose=True)

def main(eval : str, run_dir : str, out_dir : str, baseline : str = None, model : str = None, index : str = 'msmarco_passage', dataset : str = 'irds:msmarco-passage/train/triples-small'):  
    dataset = pt.get_dataset(dataset)
    bm25 = pt.BatchRetrieve(pt.get_dataset(index).get_index("terrier_stemmed_text"), wmodel="BM25")
    eval = pt.get_dataset(eval)
    os.makedirs(out_dir, exist_ok=True)
    if baseline : 
        baseline = bm25 >> pt.text.get_text(dataset, "text") >> make_lsr(join(baseline, 'model'))
        if not os.path.exists(join(out_dir, "baseline_run.gz")):
            res = baseline.transform(eval.get_topics())
            pt.io.write_results(res, join(out_dir, "baseline_run.gz"))
    if not model:
        dirs = [f for f in os.listdir(run_dir) if os.path.isdir(join(run_dir, f)) and 'baseline' not in f]
        for _, store in enumerate(dirs):
            if os.path.exists(join(out_dir, f"{store}_run.gz")):
                continue
            try:
                _model = bm25 >> pt.text.get_text(dataset, "text") >> make_lsr(join(run_dir, store, 'model'))
            except:
                print(f"Failed to load {store}")
                continue
            res = _model.transform(eval.get_topics())
            pt.io.write_results(res, join(out_dir, f"{store}_run.gz"))
            del _model
    else:
        _model = bm25 >> pt.text.get_text(dataset, "text") >> make_lsr(join(run_dir, model, 'model'))
        res = _model.transform(eval.get_topics())
        pt.io.write_results(res, join(out_dir, f"{model}_run.gz"))
    return "Success!"

if __name__ == '__main__':
    fire.Fire(main) 