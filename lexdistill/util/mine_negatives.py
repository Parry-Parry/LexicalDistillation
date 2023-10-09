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
import os

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def sample_negs(x, num_negs):
    if len(x) < num_negs:
        return x.sample(n=num_negs, replace=True)
    else:
        return x.sample(n=num_negs)
    

def main(triples_path : str,
         out_path : str,
         batch_size : int = 1000,
         num_negs : int = 32) -> str:
    
    triples = pd.read_csv(triples_path, sep="\t", index_col=False).rename(columns={'query_id': 'qid'})
    dataset = irds.load("msmarco-passage/train/triples-small")
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id')['text'].to_dict()
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id')['text'].to_dict()

    def get_query_text(x):
        df = pd.DataFrame({'qid' : x.values, 'query' : x.apply(lambda qid : clean(queries[str(qid)]))})
        return df

    pt_index = pt.get_dataset("msmarco_passage").get_index("terrier_stemmed")
    pt_index = pt.IndexFactory.of(pt_index, memory=True)
    bm25_scorer = pt.text.scorer(body_attr="text", wmodel="BM25", background_index=pt_index)
    index = PisaIndex.from_dataset("msmarco_passage", threads=8)
    bm25 = pt.apply.generic(lambda x : get_query_text(x)) >> index.bm25(num_results=1000) >> pt.text.get_text(pt.get_dataset('irds:msmarco-passage/train/triples-small'), 'text')

    def pivot_batch(batch):
        records = []
        for row in batch.itertuples():
            records.extend([{
                'qid': str(row.qid),
                'docno': str(row.doc_id_a),
                },
                ])
        return pd.DataFrame.from_records(records)

    def convert_to_dict(result):
        lookup = defaultdict(lambda : defaultdict(lambda : 0.))
        for row in result.itertuples():
            lookup[str(row.qid)][str(row.docno)] = float(row.score)
        return lookup
    
    def score(batch, norm=False):
        new = pivot_batch(batch.copy())
        topics = new['qid'].drop_duplicates()
        # score with bm25 over all topics and if any (qid docno) pair from new is missing, rsecore missing records with bm25 scorer 
        rez = bm25.transform(topics)

        new['query'] = new['qid'].apply(lambda qid : clean(queries[str(qid)]))
        new['text'] = new['docno'].apply(lambda qid : clean(docs[str(qid)]))
        
        batch_score = bm25_scorer.transform(new)
        rez = rez.append(batch_score).drop_duplicates(['qid', 'docno']).reset_index(drop=True)

        if norm:
            # minmax norm over each query score set 
            rez['score'] = rez.groupby('qid', group_keys=False)['score'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return rez
    
    main_lookup = {}
    new_triples = []

    for subset in tqdm(split_df(triples, ceil(len(triples) / batch_size)), desc="Total Batched Iter"):
        new = subset.copy()
        new = pivot_batch(new)
        new_triple = new[['qid', 'docno']].copy()
        res : pd.DataFrame = score(subset, norm=True)

        # remove all qid, docid combos from neg_pool which are in res
        neg_pool = res.copy()
        neg_pool = neg_pool[neg_pool['docno'].isin(new['docno']) == False]

        # randomly sample num_neg docs res groupby qid
        negs = neg_pool.groupby('qid').apply(lambda x : sample_negs(x, num_negs)).reset_index(drop=True)
        logging.info(negs.head())
        # create dict of qid to list of docids in negs
        negs = negs.groupby('qid')['docno'].apply(list).to_dict()

        new_triple['doc_id_b'] = new_triple['qid'].apply(lambda x : negs[str(x)])

        results_lookup = convert_to_dict(res)
        new['score'] = new.apply(lambda x : results_lookup[str(x.qid)][str(x.docno)], axis=1)
        main_lookup.update(convert_to_dict(new))
        new_triples.append(new_triple[['qid', 'docno', 'doc_id_b']].rename(columns={'doc_id': 'doc_id_a'}))

    with open(out_path, 'w') as f:
        json.dump(main_lookup, f)
    
    new_triples = pd.concat(new_triples)
    new_triples_path = os.path.dirname(triples_path) + '/triples_with_negs.tsv.gz'
    new_triples.to_csv(new_triples_path, sep='\t', index=False)

    return "Done!" 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)