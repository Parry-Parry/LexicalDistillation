import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
from pyterrier.model import split_df
from fire import Fire
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import ir_datasets as irds
import gc
import re
from pyterrier_pisa import PisaIndex

def convert_to_dict(result):
    result = result.groupby('qid').apply(lambda x: dict(zip(x['docno'], zip(x['score'], x['rank'])))).to_dict()
    return result

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def main(out_dir : str, subset : int = 100000, budget : int = 1000, batch_size : int = 1000, num_threads : int = 8):
    index = PisaIndex.from_dataset("msmarco_passage", threads=num_threads)
    
    dataset = irds.load("msmarco-passage/train/triples-small")
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id')['text'].to_dict()
    train = pd.DataFrame(dataset.docpairs_iter()).drop(['doc_id_b'], axis=1).rename(columns={'query_id': 'qid',})
    train = train.sample(n=subset) 

    def get_query_text(x):
        x['query'] = x['qid'].apply(lambda qid : clean(queries[qid]))
        return x
    bm25 = pt.apply.generic(lambda x : get_query_text(x)) >> index.bm25(num_results=budget)

    new_set = []

    for subset in tqdm(split_df(train, batch_size), desc="Total Batched Iter"):
        new = subset.copy()
        topics = subset[['qid']].drop_duplicates()
        res = bm25.transform(topics).drop(['score', 'rank'], axis=1)

        def get_sample(qid):
            return res[res.qid==qid].iloc[:1000].sample(n=1)['docno'].iloc[0]

        new['doc_id_b'] = new.apply(lambda x : get_sample(), axis=1)
        new_set.append(new)

    new_set = pd.concat(new_set)
    new_set.to_csv(out_dir, sep='\t', index=False)

    return "Done!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)