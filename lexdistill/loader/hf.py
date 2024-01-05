from itertools import chain
from torch.utils.data import Dataset
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
                 batch_size : int = 16,
                 num_negatives : int = 1,
                 shuffle : bool = False) -> None:
        super().__init__()
        self.teacher_file = teacher_file
        self.triples_file = triples_file
        self.corpus = corpus

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
    
    def get_teacher_scores(self, qid, doc_id, neg=False): 
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

class PerfectMarginDataset(TripletIDDistilDataset):
    def __init__(self, teacher_file: str, triples_file: str, corpus: Any, batch_size: int = 16, num_negatives: int = 1, shuffle: bool = False) -> None:
        super().__init__(teacher_file, triples_file, corpus, batch_size, num_negatives, shuffle)
    
    def get_teacher_scores(self, qid, doc_id, neg=False):
        if neg == False: return [1.]
        else: return [0.]

class StandardMarginDataset(TripletIDDistilDataset):
    def __init__(self, teacher_file: str, triples_file: str, corpus: Any, tokenizer: Any, batch_size: int = 16, num_negatives: int = 1, shuffle: bool = False) -> None:
        super().__init__(teacher_file, triples_file, corpus, tokenizer, batch_size, num_negatives, shuffle)
    def get_teacher_scores(self, qid, doc_id, neg=False):
        try:
            score = self.teacher[str(qid)][str(doc_id)]
        except KeyError:
            score = 0. if neg else 1.
        return [score]

class MultiMarginDataset(TripletIDDistilDataset):
    def __init__(self, teacher_file: str, triples_file: str, corpus: Any, batch_size: int = 16, num_negatives: int = 1, shuffle: bool = False) -> None:
        super().__init__(teacher_file, triples_file, corpus, batch_size, num_negatives, shuffle)
    def get_teacher_scores(self, qid, doc_id, neg=False):
        try:
            score = self.teacher[str(qid)][str(doc_id)]
            additional = 0. if neg else 1.
            score = (score + additional) / 2
        except KeyError:
            score = 0. if neg else 1.
        return [score]


class DotDataCollator:
    "Tokenize and batch of (query, pos, neg, pos_score, neg_score)"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.q_max_length = 30
        self.d_max_length = 200

    def __call__(self, batch):
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for (q, dx, *args) in batch:
            batch_queries.append(q)
            batch_docs.extend(dx)
            if len(args) == 0:
                continue
            batch_scores.extend(args[0])
        # flatten all lists 
        batch_queries = list(chain.from_iterable(batch_queries))
        batch_scores = list(chain.from_iterable(batch_scores))

        tokenized_queries = self.tokenizer(
            batch_queries,
            padding=True,
            truncation=True,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        tokenized_docs = self.tokenizer(
            batch_docs,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        return {
            "queries": dict(tokenized_queries),
            "docs_batch": dict(tokenized_docs),
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }
    
class CatDataCollator:
    "Tokenize and batch of (query, pos, neg, pos_score, neg_score)"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.q_max_length = 30
        self.d_max_length = 200

    def __call__(self, batch):
        batch_queries = []
        batch_docs = []
        batch_scores = []
        for (q, dx, *args) in batch:
            batch_queries.extend([q]*len(dx))
            batch_docs.extend(dx)
            if len(args) == 0:
                continue
            batch_scores.extend(args[0])
        # flatten lists 
        batch_scores = list(chain.from_iterable(batch_scores))

        tokenized_sequences = self.tokenizer(
            batch_queries,
            batch_docs,
            padding=True,
            truncation=True,
            max_length=self.q_max_length + self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        return {
            "sequences": dict(tokenized_sequences),
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }