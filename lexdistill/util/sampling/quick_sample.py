import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
from fire import Fire
import pandas as pd
import logging
import ir_datasets as irds
import re

def convert_to_dict(result):
    result = result.groupby('qid').apply(lambda x: dict(zip(x['docno'], zip(x['score'], x['rank'])))).to_dict()
    return result

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def main(out_path : str, triples_path : str = None, subset : int = 100000):
    dataset = irds.load("msmarco-passage/train/triples-small")
    if triples_path:
        train = pd.read_csv(triples_path, sep='\t', dtype={'qid':str, 'doc_id_a':str, 'doc_id_b': str}, index_col=False)
    else:
        train = pd.DataFrame(dataset.docpairs_iter()).rename(columns={'query_id': 'qid',})
    train = train.sample(n=subset) 

    train.to_csv(out_path, sep='\t', index=False)

    return "Done!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)