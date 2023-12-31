import ir_measures
import numpy as np
import pandas as pd
import ir_datasets as irds
from pyterrier_pisa import PisaIndex
from transformers import TrainerCallback
from lexdistill.training.transformer.lsr import LSR
from lexdistill.training.transformer.bertdot import make_scorer as make_dot_scorer
from lexdistill.training.transformer.bertcat import make_scorer as make_cat_scorer

# Adapted from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d

class EarlyStopping(object):
    def __init__(self, val_topics, metric, qrels, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

        self.val_topics = val_topics
        self.metric = ir_measures.parse_measure(metric)
        self.evaluator = ir_measures.evaluator([self.metric], qrels)

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
    
    def compute_metric(self, ranks):
        ranks = ranks.copy().rename(columns={'qid': 'query_id', 'docno': 'doc_id'})
        ranks['score'] = ranks['score'].astype(float)
        ranks['query_id'] = ranks['query_id'].astype(str)
        ranks['doc_id'] = ranks['doc_id'].astype(str)
        value = self.evaluator.calc_aggregate(ranks)
        return list(value.values())[0]
                
    def __call__(self, model):
        print('Running Validation')
        ranks = model.transform(self.val_topics)
        print('Metric Compute')
        value = self.compute_metric(ranks)
        print(f'Performance: {value}') 
        return self.step(value)

class SparseEarlyStoppingCallback(TrainerCallback):
    def __init__(self, 
                 tokenizer,
                 val_topics, 
                 ir_dataset,
                 metric, 
                 early_check = 4000,
                 min_train_steps = 100000,
                 mode='max', 
                 min_delta=0, 
                 patience=10, 
                 percentage=False) -> None:
        super().__init__()
        val_topics = pd.read_csv(val_topics, sep='\t', index_col=False)
        corpus = irds.load(ir_dataset)
        queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
        docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        qrels = corpus.qrels_iter()
        val_topics['query'] = val_topics['qid'].apply(lambda x: queries[str(x)])
        val_topics['text'] = val_topics['docno'].apply(lambda x: docs[str(x)])
        del queries
        del docs
        self.stopping = EarlyStopping(val_topics, metric, qrels, mode, min_delta, patience, percentage)
        self.tokenizer = tokenizer
        self.early_check = early_check
        self.min_train_steps = min_train_steps

    def on_step_end(self, args, state, control, **kwargs):
        global_step = state.global_step
        if (
            global_step % self.early_check == 0
            and global_step > self.min_train_steps
        ):
            val_model = LSR(kwargs['model'], self.tokenizer, fp16=True, batch_size=256) 
            
            if self.stopping(val_model):
                control.should_training_stop = True  # Stop training

class EncoderEarlyStoppingCallback(TrainerCallback):
    def __init__(self, 
                 tokenizer,
                 val_topics, 
                 ir_dataset,
                 metric, 
                 early_check = 4000,
                 min_train_steps = 100000,
                 mode='max', 
                 min_delta=0, 
                 patience=10, 
                 percentage=False,
                 model='cat') -> None:
        super().__init__()
        val_topics = pd.read_csv(val_topics, sep='\t', index_col=False)
        corpus = irds.load(ir_dataset)
        queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()
        docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
        qrels = corpus.qrels_iter()
        val_topics['query'] = val_topics['qid'].apply(lambda x: queries[str(x)])
        val_topics['text'] = val_topics['docno'].apply(lambda x: docs[str(x)])
        del queries
        del docs
        self.stopping = EarlyStopping(val_topics, metric, qrels, mode, min_delta, patience, percentage)
        self.tokenizer = tokenizer
        self.early_check = early_check
        self.min_train_steps = min_train_steps
        self.transformer = make_cat_scorer if model == 'cat' else make_dot_scorer

    def on_step_end(self, args, state, control, **kwargs):
        global_step = state.global_step
        if (
            global_step % self.early_check == 0
            and global_step > self.min_train_steps
        ):
            val_model = self.transformer(kwargs['model'])
            if self.stopping(val_model):
                control.should_training_stop = True  # Stop training
