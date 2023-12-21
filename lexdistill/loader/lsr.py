from datasets import Dataset
from typing import Any
import json
import pandas as pd
import ast
import torch

class TripletIDDistilDataset(Dataset):
    """
    Dataset with teacher's scores for distillation
    """

    def __init__(self, 
                 teacher_file : str, 
                 triples_file : str, 
                 corpus : Any,
                 tokenizer : Any,
                 batch_size : int = 16,
                 num_negatives : int = 1,
                 shuffle : bool = False,
                 tokenizer_kwargs : dict = None) -> None:
        super().__init__()
        self.teacher_file = teacher_file
        self.triples_file = triples_file
        self.tokenizer = tokenizer
        self.corpus = corpus

        if tokenizer_kwargs is not None: self.tokenizer_kwargs.update(tokenizer_kwargs)

        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.shuffle = shuffle

        with open(self.teacher_file, 'r') as f:
            self.teacher = json.load(f)
        self.triples = pd.read_csv(self.triples_file, sep='\t', converters={'doc_id_pd' : pd.eval}, dtype={'qid':str, 'doc_id_a':str, 'doc_id_b': str}, index_col=False)
        self.triples['doc_id_b'] = self.triples['doc_id_b'].apply(lambda x : ast.literal_eval(x)[:self.num_negatives])
        if self.shuffle: self.triples = self.triples.sample(frac=1).reset_index(drop=True)
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()
    
    def get_teacher_scores_one_sided(self, qid, doc_id, neg=False): 
        if neg == False: return [1.]
        try:
            score = self.teacher[str(qid)][str(doc_id)]
        except KeyError:
            score = 0. 

        return [score]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        q, d = [self.queries[item['qid']]], [self.docs[item['doc_id_a']]]
        y = [self.get_teacher_scores(item['qid'], item['doc_id_a'], neg=False)]
        for neg_item in item['doc_id_b']:
            neg_score = self.get_teacher_scores(item['qid'], neg_item, neg=True)
            d.append(self.docs[neg_item])
            y.append(neg_score)
        
        return (q, d, y)

class DistillDataCollator:
    "Tokenize and batch of (query, pos, neg, pos_score, neg_score)"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_queries = []
        pos_docs = []
        neg_docs = []
        batch_scores = []
        for (query, doc_group, *args) in batch:
            batch_queries.append(query)
            pos_docs.append(doc_group[0])
            neg_docs.append(doc_group[1])
            if len(args) == 0:
                continue
            batch_scores.append(args[0])
        tokenized_queries = self.tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokenized_pos_docs = self.tokenizer(
            pos_docs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokenized_neg_docs = self.tokenizer(
            neg_docs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        return {
            "queries": dict(tokenized_queries),
            "pos_docs": dict(tokenized_pos_docs),
            "neg_docs": dict(tokenized_neg_docs),
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }
    
