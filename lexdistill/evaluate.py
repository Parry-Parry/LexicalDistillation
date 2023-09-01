from fire import Fire 
import os 
from os.path import join
from ir_measures import evaluator, read_trec_run 
from ir_measures import *
import ir_datasets as irds
import pandas as pd


def main(eval :str, run_dir : str, out_dir : str):
    files = [f for f in os.listdir(run_dir) if os.path.isfile(join(run_dir, f))]
    ds = irds.load(eval)
    qrels = ds.qrels_iter()
    metrics = [AP(rel=2), NDCG(cutoff=10), R(rel=2)@1000, P(rel=2)@10, P(rel=2)@1000, RR(rel=2)]
    evaluate = evaluator(metrics, qrels)
    df = []
    for file in files:
        if file.endswith(".gz"):
            name = file.split(".")[0]
            run = read_trec_run(join(run_dir, file))
            res = evaluate.calc_aggregate(run)
            res = {str(k) : v for k, v in res.items()}
            res['name'] = name 
            df.append(res)
            
            per_query = []
            for q_res in evaluate.iter_calc(run):
                q_res = {str(k) : v for k, v in q_res._asdict().items()}
                q_res['name'] = name
                print(q_res)
                per_query.append(q_res)
            
            per_query = pd.DataFrame.from_records(per_query)
            # pivot metric per query_id 
            per_query = per_query.pivot(index='query_id', columns='measure', values='value')
            print(per_query.head())
            per_query.to_csv(join(out_dir, f"{name}_per_query.tsv"), index=False, sep='\t')
    
    df = pd.DataFrame.from_records(df)
    df.to_csv(join(out_dir, 'metrics.tsv'), sep='\t', index=False)

    return "Success!"

if __name__ == '__main__':
    Fire(main)