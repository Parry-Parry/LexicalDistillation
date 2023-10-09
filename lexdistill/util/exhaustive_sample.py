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

def convert_to_dict(result):
    result = result.groupby('qid').apply(lambda x: dict(zip(x['docno'], zip(x['score'], x['rank'])))).to_dict()
    return result

def pivot_batch(batch):
        records = []
        pos_list = batch.apply(lambda x : (str(x.qid), str(x.doc_id_a)), axis=1).tolist()
        for row in batch.itertuples():
            records.extend([{
                'qid': str(row.qid),
                'docno': str(row.doc_id_a),
                },
                ])
        return pd.DataFrame.from_records(records), pos_list

def convert_to_dict(result):
        result.drop_duplicates(['qid', 'docno'], inplace=True)
        lookup = defaultdict(lambda : defaultdict(int))
        for row in result.itertuples():
            lookup[str(row.qid)][str(row.docno)] = float(row.score)
        return lookup

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def main(lookup_path : str, triples_path : str, subset : int = 100000, num_negs : int = 32, batch_size : int = 1000):
    dataset = irds.load("msmarco-passage/train/triples-small")
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id')['text'].to_dict()
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id')['text'].to_dict()

    pt_index = pt.get_dataset("msmarco_passage").get_index("terrier_stemmed")
    pt_index = pt.IndexFactory.of(pt_index, memory=True)
    bm25_scorer = pt.text.scorer(body_attr="text", wmodel="BM25", background_index=pt_index)
    index = PisaIndex.from_dataset("msmarco_passage", threads=8)

    def get_query_text(x):
        df = pd.DataFrame({'qid' : x.values, 'query' : x.apply(lambda qid : clean(queries[str(qid)]))})
        return df

    bm25 = pt.apply.generic(lambda x : get_query_text(x)) >> index.bm25(k1=1.2, b=0.75, num_results=1000) >> pt.text.get_text(pt.get_dataset('irds:msmarco-passage/train/triples-small'), 'text')
    
    def score(batch, norm=False):
        new, _ = pivot_batch(batch.copy())
        topics = new['qid'].drop_duplicates()
        # score with bm25 over all topics and if any (qid docno) pair from new is missing, rsecore missing records with bm25 scorer 
        logging.info('initial scoring...')
        res = bm25.transform(topics)[['qid', 'docno', 'score']]

        new['query'] = new['qid'].apply(lambda qid : clean(queries[str(qid)]))
        new['text'] = new['docno'].apply(lambda qid : clean(docs[str(qid)]))
        logging.info('rescoring...')
        batch_score = bm25_scorer.transform(new)[['qid', 'docno', 'score']]
        res = pd.concat([res, batch_score]).drop_duplicates(['qid', 'docno']).reset_index(drop=True)

        if norm:
            # minmax norm over each query score set 
            res['score'] = res.groupby('qid', group_keys=False)['score'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return res

    train = pd.DataFrame(dataset.docpairs_iter()).rename(columns={'query_id': 'qid',})

    to_retrieve = subset 
    main_lookup = {}
    new_set = []

    while to_retrieve > 0: 
        logging.info(f"Retrieving {to_retrieve} more triples")
        sub = train.sample(n=to_retrieve).rename(columns={'doc_id_b': 'doc_id_b_0',})
        logging.info('batching...')
        for _sub in tqdm(split_df(sub, ceil(len(sub) / batch_size)), desc="Total Batched Iter"):
            _triples = _sub.copy()
            new, pos_list = pivot_batch(_triples)
            logging.info('scoring...')
            res : pd.DataFrame = score(_sub, norm=True)

            # filter res by qids that have more than num_neg results 
            logging.info('filtering...')
            res = res.groupby('qid').filter(lambda x : len(x) >= num_negs)
            _triples = _triples[_triples['qid'].isin(res['qid'].unique())]
            to_retrieve -= len(_triples)
            neg_pool = res.copy()
            neg_pool = neg_pool[~neg_pool.set_index(['qid', 'docno']).index.isin(new.set_index(['qid', 'docno']).index)].reset_index(drop=True)
            logging.info('sampling...')
            # randomly sample num_neg docs res groupby qid
            negs = neg_pool.groupby('qid').apply(lambda x : x.sample(n=num_negs)).reset_index(drop=True)[['qid', 'docno']]
            new = pd.concat([new, negs])
            # create dict of qid to list of docids in negs
            negs = negs.groupby('qid')['docno'].apply(list).to_dict()

            _triples['doc_id_b'] = _triples['qid'].apply(lambda x : negs[str(x)])

            results_lookup = convert_to_dict(res)

            def lookup(x):
                try:
                    return results_lookup[str(x.qid)][str(x.docno)]
                except KeyError:
                    if (str(x.qid), str(x.docno)) in pos_list: return 1.
                    return 0.
            logging.info('updating...')
            new['score'] = new.apply(lambda x : lookup(x), axis=1)
            main_lookup.update(convert_to_dict(new))
            new_set.append(_triples[['qid', 'doc_id_a', 'doc_id_b']])

    with open(lookup_path, 'w') as f:
        json.dump(main_lookup, f)
    
    new_triples = pd.concat(new_triples)
    new_triples.to_csv(triples_path, sep='\t', index=False)


    return "Done!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)