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

def main(out_dir : str, subset : int = 100000):
    dataset = irds.load("msmarco-passage/train/triples-small")
    train = pd.DataFrame(dataset.docpairs_iter()).drop(['doc_id_b'], axis=1).rename(columns={'query_id': 'qid',})
    train = train.sample(n=subset) 

    train.to_csv(out_dir, sep='\t', index=False)

    return "Done!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)