from collections import defaultdict
import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
from pyterrier.model import split_df
from fire import Fire
import pandas as pd
from tqdm import tqdm
import logging
import json
import re
from math import ceil
import ir_datasets as irds
from pyterrier_pisa import PisaIndex

def convert_to_dict(result):
    result = result.groupby('qid').apply(lambda x: dict(zip(x['docno'], zip(x['score'], x['rank'])))).to_dict()
    return result

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def main(triples_path : str,
         out_path : str,
         batch_size : int = 1000) -> str:
    
    triples = pd.read_csv(triples_path, sep="\t", index_col=False).rename(columns={'query_id': 'qid'})
    queries = pd.DataFrame(irds.load("msmarco-passage/train/triples-small").queries_iter()).set_index('query_id')['text'].to_dict()

    def get_query_text(x):
        x['query'] = x.apply(lambda qid : clean(queries[str(qid)]))
        return x

    index = PisaIndex.from_dataset("msmarco_passage", threads=8)
    bm25 = pt.apply.generic(lambda x : get_query_text(x)) >> pt.text.get_text(pt.get_dataset("irds:msmarco-passage/train/triples-small"), 'text') >> index.bm25(num_results=5000)

    def pivot_batch(batch):
        records = []
        for row in batch.itertuples():
            records.extend([{
                'qid': row.qid,
                'docno': row.doc_id_a,
                },
                {
                'qid': row.qid,
                'docno': row.doc_id_b,
                }
                ])
        return pd.DataFrame.from_records(records)

    def convert_to_dict(result):
        lookup = defaultdict(lambda : defaultdict(lambda : 0))
        for row in result.itertuples():
            lookup[row.qid][row.docno] = row.score
        return lookup
    
    def score(batch, model, norm=False):
        rez = model.transform(batch)
        if norm:
            # minmax norm over each query score set 
            rez['score'] = rez.groupby('qid', group_keys=False)['score'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return rez
   
    main_lookup = {}

    for subset in tqdm(split_df(triples, ceil(len(triples) / batch_size)), desc="Total Batched Iter"):
        new = pivot_batch(subset.copy())
        topics = subset['qid'].drop_duplicates()
        res = score(topics, bm25, norm=True)
        # create default dict of results with key qid, docno
        results_lookup = convert_to_dict(res)
        new['score'] = new.apply(lambda x : results_lookup[x.qid][x.docno], axis=1)
        main_lookup.update(convert_to_dict(new))

    with open(out_path, 'w') as f:
        json.dump(main_lookup, f)

    return "Done!" 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)