from fire import Fire
import pandas as pd
import logging
import ir_datasets as irds

def main(triples_path : str, out_path : str):
    triples = pd.read_csv(triples_path, sep='\t', converters={'doc_id_d' : pd.eval}, dtype={'qid':str, 'doc_id_a':str, 'doc_id_b': str}, index_col=False).drop(columns=['doc_id_b'])
    dataset = irds.load("msmarco-passage/train/triples-small")
    train = pd.DataFrame(dataset.docpairs_iter()).rename(columns={'query_id': 'qid',}).set_index(['qid', 'doc_id_a']).doc_id_b.to_dict()

    triples['doc_id_b'] = triples.apply(lambda x : train[(x.qid, x.doc_id_a)], axis=1)

    triples.to_csv(out_path, sep='\t', index=False)
    return "Done!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)