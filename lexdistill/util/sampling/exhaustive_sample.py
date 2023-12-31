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
from os.path import join

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

def sample_neg(neg_pool, num_negs):
    if len(neg_pool) < num_negs:
        logging.info(f'not enough negs, sampling with replacement {len(neg_pool)} < {num_negs}')
        return neg_pool.sample(n=num_negs, replace=True)
    else:
        return neg_pool.sample(n=num_negs)

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def main(out_path : str, 
         subset : int = 100000, 
         num_negs : int = 32, 
         batch_size : int = 1000, 
         data_split : str = 'train/triples-small',
         docpairs_file : str = None,
         val_split : int = None):
    
    dataset = irds.load(f"msmarco-passage/{data_split}")
    queries = pd.DataFrame(dataset.queries_iter()).set_index('query_id')['text'].to_dict()
    docs = pd.DataFrame(dataset.docs_iter()).set_index('doc_id')['text'].to_dict()

    pt_index = pt.get_dataset("msmarco_passage").get_index("terrier_stemmed")
    pt_index = pt.IndexFactory.of(pt_index, memory=False)
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
        res = bm25.transform(topics)[['qid', 'docno', 'score']]

        new['query'] = new['qid'].apply(lambda qid : clean(queries[str(qid)]))
        new['text'] = new['docno'].apply(lambda qid : clean(docs[str(qid)]))
        batch_score = bm25_scorer.transform(new)[['qid', 'docno', 'score']]
        res = pd.concat([res, batch_score]).drop_duplicates(['qid', 'docno']).reset_index(drop=True)

        if norm:
            # minmax norm over each query score set 
            res['score'] = res.groupby('qid', group_keys=False)['score'].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return res

    if docpairs_file:
        train = pd.read_csv(docpairs_file, sep='\t', index_col=False)
    else:    
        train = pd.DataFrame(dataset.docpairs_iter()).rename(columns={'query_id': 'qid',})
    
    qrels = pd.DataFrame(dataset.qrels_iter()).rename(columns={'query_id': 'qid'})

    if val_split:
        val = pd.DataFrame({'qid' : [], 'doc_id_a': []})
        val_to_retrieve = val_split
        while val_to_retrieve > 0:
            tmp = train.drop_duplicates('qid').sample(n=val_to_retrieve)
            tmp = tmp[tmp['qid'].isin(qrels['qid'].unique()) & ~tmp['qid'].isin(val['qid'].unique())]
            train = train[~train['qid'].isin(tmp['qid'])]
            val_to_retrieve -= len(tmp)
            val = pd.concat([val, tmp[['qid', 'doc_id_a']]])
        val.rename(columns={'doc_id_a': 'docno'}, inplace=True)

        # get top 100 by score by sorting by qid and score then take top 100 grouped by qid
        ranks = bm25.transform(val['qid'].drop_duplicates())[['qid', 'docno', 'score']].sort_values(['qid', 'score'], ascending=[True, False]).groupby('qid').head(100)[['qid', 'docno']]
        val = pd.concat([val, ranks]).drop_duplicates(['qid', 'docno'])
        val['score'] = 0.
        val.to_csv(join(out_path, f'triples.{num_negs}.val.tsv.gz'), sep='\t', index=False)

    to_retrieve = subset 
    main_lookup = {}
    new_set = []

    while to_retrieve > 0: 
        sub = train.sample(n=to_retrieve).rename(columns={'doc_id_b': 'doc_id_b_0',})
        for _sub in tqdm(split_df(sub, ceil(len(sub) / batch_size)), desc="Total Batched Iter"):
            _triples = _sub.copy()
            new, pos_list = pivot_batch(_triples)
            res = score(_sub, norm=True).groupby('qid').filter(lambda x : len(x) >= num_negs)
            _triples = _triples[_triples['qid'].isin(res['qid'].unique())]
            to_retrieve -= len(_triples)
            neg_pool = res.copy()
            neg_pool = neg_pool[~neg_pool[['qid', 'docno']].isin(pos_list)].reset_index(drop=True)
            negs = neg_pool.groupby('qid').apply(lambda x : sample_neg(x, num_negs)).reset_index(drop=True)[['qid', 'docno']]
            new = pd.concat([new, negs])
            negs = negs.groupby('qid')['docno'].apply(list).to_dict()

            _triples['doc_id_b'] = _triples['qid'].apply(lambda x : negs[str(x)])

            results_lookup = convert_to_dict(res)

            def lookup(x):
                try:
                    return results_lookup[str(x.qid)][str(x.docno)]
                except KeyError:
                    if (str(x.qid), str(x.docno)) in pos_list: return 1.
                    return 0.
            new['score'] = new.apply(lambda x : lookup(x), axis=1)
            main_lookup.update(convert_to_dict(new))
            new_set.append(_triples[['qid', 'doc_id_a', 'doc_id_b']])

    with open(join(out_path, f'lookup.{num_negs}.json'), 'w') as f:
        json.dump(main_lookup, f)
    
    new_set = pd.concat(new_set)
    new_set.to_csv(join(out_path, f'triples.{num_negs}.train.tsv.gz'), sep='\t', index=False)


    return "Done!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)