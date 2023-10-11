from fire import Fire
import pandas as pd
import logging
import ir_datasets as irds
import re

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def main(main : str, sub : str, out_dir : str):
    main_ds = irds.load(main)
    sub_ds = irds.load(sub)

    main_queries = pd.DataFrame(main_ds.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})
    sub_queries = pd.DataFrame(sub_ds.queries_iter()).rename(columns={'query_id': 'qid', 'text': 'query'})

    # get all sub not in main 

    sub_queries = sub_queries[~sub_queries.qid.isin(main_queries.qid)]
    logging.info('Sub queries not in main: %d', len(sub_queries))
    sub_queries.to_csv(out_dir, sep='\t', index=False)

    return "Done!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)