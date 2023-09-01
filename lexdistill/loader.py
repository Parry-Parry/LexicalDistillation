import pandas as pd
import json
import torch
import os
from typing import Any

class TeacherLoader:
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
                 tokenizer_kwargs : dict = None) -> None:
        self.teacher_dir = teacher_dir
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

        return sample
    
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

        return sample
    
    def get_teacher_scores_perfect(self, qid, doc_id, neg=False) -> torch.Tensor:
        sample = []
        for _, _teacher in self.teacher.items():
            try:
                score = [_teacher[str(qid)][str(doc_id)], 0. if neg else 1.]
            except KeyError:
                score = [0., 0.] if neg else [1., 1.]
            sample.append(score)

        return sample
    
    def get_teacher_scores_perfect_one_sided(self, qid, doc_id, neg=False) -> torch.Tensor:
        sample = []
        for _, _teacher in self.teacher.items():
            if neg == False: 
                sample.append([1., 1.])
                continue
            try:
                score = [_teacher[str(qid)][str(doc_id)], 0.]
            except KeyError:
                score = [0., 0.] 
            sample.append(score)

        return sample

    def format(self, q, d):
        return 'Query: ' + q + ' Document: ' + d + ' Relevant:'
    
    def tokenize(self, x):
        return self.tokenizer(x, **self.tokenizer_kwargs)
    
    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        x = [self.format(self.queries[item['qid']], self.docs[item['doc_id_a']]), self.format(self.queries[item['qid']], self.docs[item['doc_id_b']])]
        y = [self.get_teacher_scores(item['qid'], item['doc_id_a'], neg=False), self.get_teacher_scores(item['qid'], item['doc_id_b'], neg=True)]
        return x, y

    def get_batch(self, idx):
        xs = []
        ys = []
        for i in range(idx, min(len(self.triples), idx + self.batch_size)):
            x, y = self[i]
            xs.extend(x)
            ys.extend(y)
        return self.tokenize(xs), torch.tensor(ys)

class SingleTeacherLoader:
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
    
    def format(self, q, d):
        return 'Query: ' + q + ' Document: ' + d + ' Relevant:'
    
    def tokenize(self, x):
        return self.tokenizer(x, **self.tokenizer_kwargs)
    
    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        x = [self.format(self.queries[item['qid']], self.docs[item['doc_id_a']]), self.format(self.queries[item['qid']], self.docs[item['doc_id_b']])]
        y = [self.get_teacher_scores(item['qid'], item['doc_id_a'], neg=False), self.get_teacher_scores(item['qid'], item['doc_id_b'], neg=True)]
        return x, y

    def get_batch(self, idx):
        xs = []
        ys = []
        for i in range(idx, min(len(self.triples), idx + self.batch_size)):
            x, y = self[i]
            xs.extend(x)
            ys.extend(y)
        return self.tokenize(xs), torch.tensor(ys)

class StandardLoader:
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
    
    def tokenize(self, x):
        return self.tokenizer(x, **self.tokenizer_kwargs)
    
    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        x = [self.format(self.queries[item['qid']], self.docs[item['doc_id_a']]), self.format(self.queries[item['qid']], self.docs[item['doc_id_b']])]
        return x

    def get_batch(self, idx):
        xs = []
        for i in range(idx, min(len(self.triples), idx + self.batch_size)):
            x = self[i]
            xs.extend(x)
        y = self.tokenize(['true' if i % 2 == 0 else 'false' for i in range(len(xs))]).input_ids
        return self.tokenize(xs), y

class PerfectLoader:
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
    
    def tokenize(self, x):
        return self.tokenizer(x, **self.tokenizer_kwargs)
    
    def __getitem__(self, idx):
        item = self.triples.iloc[idx]
        x = [self.format(self.queries[item['qid']], self.docs[item['doc_id_a']]), self.format(self.queries[item['qid']], self.docs[item['doc_id_b']])]
        return x

    def get_batch(self, idx):
        xs = []
        for i in range(idx, min(len(self.triples), idx + self.batch_size)):
            x = self[i]
            xs.extend(x)
        y = torch.tensor([[1.] if i % 2 == 0 else [0.] for i in range(len(xs))])
        return self.tokenize(xs), y