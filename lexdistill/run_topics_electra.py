import pyterrier as pt
pt.init()
import fire 
import os
from os.path import join
from pyterrier_dr import ElectraScorer

def main(eval : str, run_dir : str, out_dir : str, baseline : str = None, model : str = None):  
    dataset = pt.get_dataset("irds:msmarco-passage/train/triples-small")
    bm25 = pt.BatchRetrieve(pt.get_dataset("msmarco_passage").get_index("terrier_stemmed_text"), wmodel="BM25")
    eval = pt.get_dataset(eval)
    if baseline : baseline = bm25 >> pt.text.get_text(dataset, "text") >> ElectraScorer(model_name=join(baseline, 'model'))
    else: baseline = bm25 >> pt.text.get_text(dataset, "text") >> ElectraScorer()
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(join(out_dir, "baseline_run.gz")):
        res = baseline.transform(eval.get_topics())
        pt.io.write_results(res, join(out_dir, "baseline_run.gz"))
    if not model:
        dirs = [f for f in os.listdir(run_dir) if os.path.isdir(join(run_dir, f)) and 'baseline' not in f]
        for _, store in enumerate(dirs):
            if os.path.exists(join(out_dir, f"{store}_run.gz")):
                continue
            _model = bm25 >> pt.text.get_text(dataset, "text") >> ElectraScorer(model_name=join(run_dir, store, 'model'))
            res = _model.transform(eval.get_topics())
            pt.io.write_results(res, join(out_dir, f"{store}_run.gz"))
            del _model
    else:
        _model = bm25 >> pt.text.get_text(dataset, "text") >> ElectraScorer(model_name=join(run_dir, model, 'model'))
        res = _model.transform(eval.get_topics())
        pt.io.write_results(res, join(out_dir, f"{model}_run.gz"))
    return "Success!"

if __name__ == '__main__':
    fire.Fire(main) 