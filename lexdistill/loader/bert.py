import pandas as pd
import json
import torch
import os
from typing import Any

class BERTTeacherLoader:
    teacher = {} 
    triples = None
    tokenizer_kwargs = {'padding' : 'longest', 'truncation' : True, 'return_tensors' : 'pt'}
    def __init__(self, 
                 teacher_dir : str, 
                 triples_file : str, 
                 corpus : Any,
                 tokenizer : Any,
                 mode = 'std',
                 batch_size : int = 16,
                 shuffle : bool = False,
                 tokenizer_kwargs : dict = None,
                 aggr_func : callable = lambda x : x) -> None:
        self.teacher_dir = teacher_dir
        self.triples_file = triples_file
        self.tokenizer = tokenizer
        self.corpus = corpus

        if tokenizer_kwargs is not None: self.tokenizer_kwargs.update(tokenizer_kwargs)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.initialized = False

        self.aggr_func = aggr_func

        if mode == 'std': 
            self.get_teacher_scores = self.get_teacher_scores_std
        elif mode == 'one_sided':
            self.get_teacher_scores = self.get_teacher_scores_one_sided
        elif mode == 'perfect':
            self.get_teacher_scores = self.get_teacher_scores_perfect
        elif mode == 'perfect_one_sided':
            self.get_teacher_scores = self.get_teacher_scores_perfect_one_sided

    def setup(self) -> None:
        for i, file in enumerate(os.listdir(self.teacher_dir)):
            with open(os.path.join(self.teacher_dir, file), 'r') as f:
                self.teacher[i] = json.load(f)
        self.triples = pd.read_csv(self.triples_file, sep='\t', dtype={'qid':str, 'doc_id_a':str, 'doc_id_b':str}, index_col=False)
        if self.shuffle: self.triples = self.triples.sample(frac=1).reset_index(drop=True)
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()

        self.initialized = True

    def get_teacher_scores_std(self, qid, doc_id, neg=False) -> torch.Tensor:
        sample = []
        for _, _teacher in self.teacher.items():
            try:
                score = _teacher[str(qid)][str(doc_id)]
            except KeyError:
                score = 0. if neg else 1.
            sample.append(score)

        return self.aggr_func(sample)
    
    def get_teacher_scores_one_sided(self, qid, doc_id, neg=False) -> torch.Tensor:
        sample = []
        for _, _teacher in self.teacher.items():
            if neg == False: 
                sample.append(1.)
                continue
            try:
                score = _teacher[str(qid)][str(doc_id)]
            except KeyError:
                score = 0.
            sample.append(score)

        return self.aggr_func(sample)
    
    def get_teacher_scores_perfect(self, qid, doc_id, neg=False) -> torch.Tensor:
        sample = []
        for _, _teacher in self.teacher.items():
            try:
                score = _teacher[str(qid)][str(doc_id)]
            except KeyError:
                score = 0. if neg else 1.
            sample.append(score)
        
        sample.append(0. if neg else 1.)

        return self.aggr_func(sample)
    
    def get_teacher_scores_perfect_one_sided(self, qid, doc_id, neg=False) -> torch.Tensor:
        sample = []
        for _, _teacher in self.teacher.items():
            if neg == False: 
                sample.append(1.)
                continue
            try:
                score = _teacher[str(qid)][str(doc_id)]
            except KeyError:
                score = 0.
            sample.append(score)
        
        sample.append(0. if neg else 1.)

        return self.aggr_func(sample)

    def format(self, q, d):
        return 'Query: ' + q + ' Document: ' + d + ' Relevant:'
    
    def tokenize(self, q, d):
        return self.tokenizer(q, d, **self.tokenizer_kwargs)
    
    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        q, d = [self.queries[item['qid']]]*2, [self.docs[item['doc_id_a']], self.docs[item['doc_id_b']]] 
        y = [self.get_teacher_scores(item['qid'], item['doc_id_a'], neg=False), self.get_teacher_scores(item['qid'], item['doc_id_b'], neg=True)]
        return q, d, y

    def get_batch(self, idx):
        q, d = [], []
        ys = []
        for i in range(idx, min(len(self.triples), idx + self.batch_size)):
            _q, _d, y = self[i]
            q.extend(_q)
            d.extend(_d)
            ys.extend(y)
        return self.tokenize(q, d), torch.tensor(ys, dtype=torch.float32)

class BERTSingleTeacherLoader:
    teacher = None 
    triples = None
    tokenizer_kwargs = {'padding' : 'longest', 'truncation' : True, 'return_tensors' : 'pt'}
    def __init__(self, 
                 teacher_file : str, 
                 triples_file : str, 
                 corpus : Any,
                 tokenizer : Any,
                 mode = 'std',
                 batch_size : int = 16,
                 shuffle : bool = False,
                 tokenizer_kwargs : dict = None) -> None:
        self.teacher_file = teacher_file
        self.triples_file = triples_file
        self.tokenizer = tokenizer
        self.corpus = corpus

        if tokenizer_kwargs is not None: self.tokenizer_kwargs.update(tokenizer_kwargs)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.initialized = False

        if mode == 'std': 
            self.get_teacher_scores = self.get_teacher_scores_std
        elif mode == 'one_sided':
            self.get_teacher_scores = self.get_teacher_scores_one_sided
        elif mode == 'reversed':
            self.get_teacher_scores = self.get_teacher_scores_reversed
        elif mode == 'perfect':
            self.get_teacher_scores = self.get_teacher_scores_perfect
        elif mode == 'perfect_one_sided':
            self.get_teacher_scores = self.get_teacher_scores_perfect_one_sided

    def setup(self) -> None:
        with open(self.teacher_file, 'r') as f:
            self.teacher = json.load(f)
        self.triples = pd.read_csv(self.triples_file, sep='\t', dtype={'qid':str, 'doc_id_a':str, 'doc_id_b':str}, index_col=False)
        if self.shuffle: self.triples = self.triples.sample(frac=1).reset_index(drop=True)
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()

        self.initialized = True

    def get_teacher_scores_std(self, qid, doc_id, neg=False) -> torch.Tensor: 
        try:
            score = self.teacher[str(qid)][str(doc_id)]
        except KeyError:
            score = 0. if neg else 1.

        return [score]

    def get_teacher_scores_one_sided(self, qid, doc_id, neg=False) -> torch.Tensor: 
        if neg == False: return [1.]
        try:
            score = self.teacher[str(qid)][str(doc_id)]
        except KeyError:
            score = 0. 

        return [score]

    def get_teacher_scores_reversed(self, qid, doc_id, neg=False) -> torch.Tensor:
        if neg == True: return [0.]
        try:
            score = self.teacher[str(qid)][str(doc_id)]
        except KeyError:
            score = 1. 

        return [score]

    def get_teacher_scores_perfect(self, qid, doc_id, neg=False) -> torch.Tensor:
        try:
            score = [self.teacher[str(qid)][str(doc_id)], 0. if neg else 1.]
        except KeyError:
            score = [0., 0.] if neg else [1., 1.]

        return score
    
    def get_teacher_scores_perfect_one_sided(self, qid, doc_id, neg=False) -> torch.Tensor:
        if neg == False: return [1., 1.]
        try:
            score = [self.teacher[str(qid)][str(doc_id)], 0. if neg else 1.]
        except KeyError:
            score = [0., 0.] 

        return score
    
    def tokenize(self, q, d):
        return self.tokenizer(q, d, **self.tokenizer_kwargs)
    
    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        q, d = [self.queries[item['qid']]]*2, [self.docs[item['doc_id_a']], self.docs[item['doc_id_b']]] 
        y = [self.get_teacher_scores(item['qid'], item['doc_id_a'], neg=False), self.get_teacher_scores(item['qid'], item['doc_id_b'], neg=True)]
        return q, d, y

    def get_batch(self, idx):
        q, d = [], []
        ys = []
        for i in range(idx, min(len(self.triples), idx + self.batch_size)):
            _q, _d, y = self[i]
            q.extend(_q)
            d.extend(_d)
            ys.extend(y)
        return self.tokenize(q, d), torch.tensor(ys)

class BERTStandardLoader:
    teacher = None 
    triples = None
    tokenizer_kwargs = {'padding' : 'longest', 'truncation' : True, 'return_tensors' : 'pt'}
    def __init__(self, 
                 triples_file : str, 
                 corpus : Any,
                 tokenizer : Any,
                 batch_size : int = 16,
                 shuffle : bool = False,
                 tokenizer_kwargs : dict = None) -> None:
        self.triples_file = triples_file
        self.tokenizer = tokenizer
        self.corpus = corpus

        if tokenizer_kwargs is not None: self.tokenizer_kwargs.update(tokenizer_kwargs)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.initialized = False

    def setup(self) -> None:
        self.triples = pd.read_csv(self.triples_file, sep='\t', dtype={'qid':str, 'doc_id_a':str, 'doc_id_b':str}, index_col=False)
        if self.shuffle: self.triples = self.triples.sample(frac=1).reset_index(drop=True)
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()

        self.initialized = True

    def format(self, q, d):
        return 'Query: ' + q + ' Document: ' + d + ' Relevant:'
    
    def tokenize(self, q, d):
        return self.tokenizer(q, d, **self.tokenizer_kwargs)
    
    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        return [self.queries[item['qid']]]*2, [self.docs[item['doc_id_a']], self.docs[item['doc_id_b']]] 

    def get_batch(self, idx):
        q, d = [], []
        for i in range(idx, min(len(self.triples), idx + self.batch_size)):
            _q, _d = self[i]
            q.extend(_q)
            d.extend(_d)
        y = torch.tensor([[0., 1.] if i % 2 == 0 else [1., 0.] for i in range(len(d))])
        return self.tokenize(q, d), y

class BERTPerfectLoader:
    teacher = None 
    triples = None
    tokenizer_kwargs = {'padding' : 'longest', 'truncation' : True, 'return_tensors' : 'pt'}
    def __init__(self, 
                 triples_file : str, 
                 corpus : Any,
                 tokenizer : Any,
                 batch_size : int = 16,
                 shuffle : bool = False,
                 tokenizer_kwargs : dict = None) -> None:
        self.triples_file = triples_file
        self.tokenizer = tokenizer
        self.corpus = corpus

        if tokenizer_kwargs is not None: self.tokenizer_kwargs.update(tokenizer_kwargs)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.initialized = False

    def setup(self) -> None:
        self.triples = pd.read_csv(self.triples_file, sep='\t', dtype={'qid':str, 'doc_id_a':str, 'doc_id_b':str}, index_col=False)
        if self.shuffle: self.triples = self.triples.sample(frac=1).reset_index(drop=True)
        self.docs = pd.DataFrame(self.corpus.docs_iter()).set_index("doc_id")["text"].to_dict()
        self.queries = pd.DataFrame(self.corpus.queries_iter()).set_index("query_id")["text"].to_dict()

        self.initialized = True
    
    def tokenize(self, q, d):
        return self.tokenizer(q, d, **self.tokenizer_kwargs)
    
    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        return [self.queries[item['qid']]]*2, [self.docs[item['doc_id_a']], self.docs[item['doc_id_b']]] 

    def get_batch(self, idx):
        q, d = [], []
        for i in range(idx, min(len(self.triples), idx + self.batch_size)):
            _q, _d = self[i]
            q.extend(_q)
            d.extend(_d)
        y = torch.tensor([[1.] if i % 2 == 0 else [0.] for i in range(len(d))])
        return self.tokenize(q, d), y