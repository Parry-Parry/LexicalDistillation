import pyterrier as pt
pt.init()
import fire 
import os
from os.path import join
from pyterrier_t5 import MonoT5ReRanker
import pandas as pd


def main(model_dir : str, out : str, eval_name : str, baseline : str, model : str = None):  
    dataset = pt.get_dataset("irds:msmarco-passage/train/triples-small")
    bm25 = pt.BatchRetrieve(pt.get_dataset("msmarco_passage").get_index("terrier_stemmed_text"), wmodel="BM25")
    eval = pt.get_dataset(eval_name)
    baseline = bm25 >> pt.text.get_text(dataset, "text") >> MonoT5ReRanker(model=join(baseline, 'model'))
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(join(out, "baseline_results.csv")):
        res = baseline.transform(eval.get_topics())
        res.to_csv(join(out, "baseline_run.tsv"), sep="\t", index=False)
    if not model:
        dirs = [f for f in os.listdir(model_dir) if os.path.isdir(join(model_dir, f)) and 'baseline' not in f]
        for _, store in enumerate(dirs):
            if os.path.exists(join(out, f"{store}_results.csv")):
                continue
            _model = bm25 >> pt.text.get_text(dataset, "text") >> MonoT5ReRanker(model=join(model_dir, store, 'model'))
            res = _model.transform(eval.get_topics())
            pt.io.write_results(res, join(out, f"{store}_run.gz"))
            del _model
    else:
        _model = bm25 >> pt.text.get_text(dataset, "text") >> MonoT5ReRanker(model=join(model_dir, model, 'model'))
        res = _model.transform(eval.get_topics())
        pt.io.write_results(res, join(out, f"{model}_run.gz"))
    return "Success!"

if __name__ == '__main__':
    fire.Fire(main) 