import json
from torch.utils.data import Dataset
import pandas as pd
import ast
import torch
from typing import Any
import ir_datasets
from tqdm import tqdm
import gzip
import pickle
import random

from lexdistill.lsr.utils.dataset_utils import (
    read_collection,
    read_queries,
    read_qrels,
    read_ce_score,
)


class TripletDistilDataset(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.triplets = []
        with open(data_path, "r") as f:
            for line in tqdm(f, desc="Loading dataset"):
                cols = line.split("\t")
                assert len(cols) == 5, "wrong format"
                self.triplets.append(
                    (cols[0], cols[1], cols[2], float(cols[3]), float(cols[4]))
                )

    def __getitem__(self, idx):
        return self.triplets[idx]

    def __len__(self):
        return len(self.triplets)


class TripletIDDistilDataset(Dataset):
    """
    Dataset with teacher's scores for distillation
    """

    def __init__(self, 
                 teacher_file : str, 
                 triples_file : str, 
                 corpus : Any,
                 num_negatives : int = 1,
                 shuffle : bool = False) -> None:
        super().__init__()
        self.teacher_file = teacher_file
        self.triples_file = triples_file
        self.corpus = corpus

        self.num_negatives = num_negatives
        self.shuffle = shuffle

        with open(self.teacher_file, 'r') as f:
            self.teacher = json.load(f)
        self.triples = pd.read_csv(self.triples_file, sep='\t', converters={'doc_id_pd' : pd.eval}, dtype={'qid':str, 'doc_id_a':str, 'doc_id_b': str}, index_col=False)
        print(f'Loaded {len(self.triples)} triples with {self.num_negatives} negatives each')
        self.triples['doc_id_b'] = self.triples['doc_id_b'].apply(lambda x : ast.literal_eval(x)[:self.num_negatives])
        if self.shuffle: self.triples = self.triples.sample(frac=1).reset_index(drop=True)
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()
    
    def get_teacher_scores(self, qid, doc_id, neg=False): 
        if neg == False: return [1.]
        try: score = self.teacher[str(qid)][str(doc_id)]
        except KeyError: score = 0. 
        return [score]

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        print(idx)
        item = self.triples.iloc[idx]
        q, d = [self.queries[item['qid']]], [self.docs[item['doc_id_a']]]
        y = [self.get_teacher_scores(item['qid'], item['doc_id_a'], neg=False)]
        for neg_item in item['doc_id_b']:
            neg_score = self.get_teacher_scores(item['qid'], neg_item, neg=True)
            d.append(self.docs[neg_item])
            y.append(neg_score)
        
        return (q, d, y)