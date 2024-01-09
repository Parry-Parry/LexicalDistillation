import pyterrier as pt
pt.init()
import fire 
import os
from os.path import join
from pyterrier_dr import ElectraScorer

def filename(*args, old=False):
    if old: return join(*args, 'model')
    return join(*args, 'model', 'classifier')

def main(eval : str, 
         run_dir : str, 
         out_dir : str, 
         baseline : str = None, 
         model : str = None, 
         index : str = 'msmarco_passage', 
         dataset : str = 'irds:msmarco-passage/train/triples-small',
         old : bool = False,
         batch_size : int = 128):  
    dataset = pt.get_dataset(dataset)
    bm25 = pt.BatchRetrieve(pt.get_dataset(index).get_index("terrier_stemmed_text"), wmodel="BM25")
    eval = pt.get_dataset(eval)
    os.makedirs(out_dir, exist_ok=True)
    if baseline : 
        baseline = bm25 >> pt.text.get_text(dataset, "text") >> ElectraScorer(model_name=filename(baseline, old=old), batch_size=batch_size)
        if not os.path.exists(join(out_dir, "baseline_run.gz")):
            res = baseline.transform(eval.get_topics())
            pt.io.write_results(res, join(out_dir, "baseline_run.gz"))
    if not model:
        dirs = [f for f in os.listdir(run_dir) if os.path.isdir(join(run_dir, f)) and 'baseline' not in f]
        for _, store in enumerate(dirs):
            if os.path.exists(join(out_dir, f"{store}_run.gz")):
                continue
            try:
                _model = bm25 >> pt.text.get_text(dataset, "text") >> ElectraScorer(model_name=filename(run_dir, store, old=old), batch_size=batch_size)
            except OSError:
                print(f"Failed to load {store}")
                continue
            res = _model.transform(eval.get_topics())
            pt.io.write_results(res, join(out_dir, f"{store}_run.gz"))
            del _model
    else:
        _model = bm25 >> pt.text.get_text(dataset, "text") >> ElectraScorer(model_name=filename(run_dir, model, old=old), batch_size=batch_size)
        res = _model.transform(eval.get_topics())
        pt.io.write_results(res, join(out_dir, f"{model}_run.gz"))
    return "Success!"

if __name__ == '__main__':
    fire.Fire(main) 