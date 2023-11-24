import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer, ElectraForSequenceClassification, ElectraModel, AutoTokenizer

class MonoT5Model(nn.Module):
    def __init__(self, model, tokenizer, rank=None):
        super().__init__()
        self.device = torch.device('cuda', rank) if rank else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.rel = self.tokenizer.encode('true')[0]
        self.nrel = self.tokenizer.encode('false')[0]
    
    @staticmethod
    def init(rank=None):
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        return MonoT5Model(model, tokenizer, rank)
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
    
    def gen_labels(self, x):
        return self.tokenizer(['true' if i % 2 == 0 else 'false' for i in range(len(x))], return_tensors='pt', padding=True).input_ids.to(self.device)
    
    def train(self):
        self.model.train()
    
    def parameters(self):
        return self.model.parameters()
    
    def forward(self, x):
        x['labels'] = self.gen_labels(x['input_ids'])
        logits = self.model(**x).logits
        result = logits[:, 0, (self.rel, self.nrel)]
        return F.log_softmax(result, dim=1)[:, 0]

class BaselineT5(nn.Module):
    def __init__(self, model, tokenizer, rank=None):
        super().__init__()
        self.device = torch.device('cuda', rank) if rank else torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.rel = self.tokenizer.encode('true')[0]
        self.nrel = self.tokenizer.encode('false')[0]
    
    @staticmethod
    def init(rank=None):
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        return BaselineT5(model, tokenizer, rank)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
    
    def gen_labels(self, x):
        return self.tokenizer(['true' if i % 2 == 0 else 'false' for i in range(len(x))], return_tensors='pt', padding=True).input_ids.to(self.device)
    
    def train(self):
        self.model.train()
    
    def parameters(self):
        return self.model.parameters()
    
    def forward(self, x):
        x['labels'] = self.gen_labels(x['input_ids'])
        return self.model(**x)

class DuoMonoT5Model(nn.Module):
    def __init__(self, model, tokenizer, rank=None):
        super().__init__()
        self.device = torch.device('cuda', rank) if rank else torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.rel = self.tokenizer.encode('true')[0]
        self.nrel = self.tokenizer.encode('false')[0]
    
    @staticmethod
    def init(rank=None):
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        return DuoMonoT5Model(model, tokenizer, rank)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
    
    def gen_labels(self, x):
        return self.tokenizer(['true' if i % 2 == 0 else 'false' for i in range(len(x))], return_tensors='pt', padding=True).input_ids.to(self.device)
    
    def train(self):
        self.model.train()
    
    def parameters(self):
        return self.model.parameters()
    
    def forward(self, x):
        x['labels'] = self.gen_labels(x['input_ids'])
        output = self.model(**x)
        logits = output.logits
        result = logits[:, 0, (self.rel, self.nrel)]
        return F.log_softmax(result, dim=1)[:, 0], output.loss

class MonoBERTModel(nn.Module):
    def __init__(self, model, tokenizer, rank=None):
        super().__init__()
        self.device = torch.device('cuda', rank) if rank else torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
    
    @staticmethod
    def init(rank=None):
        model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator', num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained('google/electra-base-discriminator')
        return MonoBERTModel(model, tokenizer, rank)
    
    def transfer_state_dict(self, skeleton):
        skeleton.model.load_state_dict(self.model.state_dict())
        
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
    
    def gen_labels(self, x):
        return torch.tensor([[0., 1.] if i % 2 == 0 else [1., 0.] for i in range(len(x))]).to(self.device)
    
    def train(self):
        self.model.train()
    
    def parameters(self):
        return self.model.parameters()
    
    def forward(self, x):
        x['labels'] = self.gen_labels(x['input_ids'])
        logits = self.model(**x).logits
        return F.softmax(logits, dim=1)[:, 1]

class BERTDotModel(nn.Module):
    def __init__(self, model, tokenizer, rank=None, return_vecs=False):
        super().__init__()
        self.device = torch.device('cuda', rank) if rank else torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.return_vecs = return_vecs
    
    @staticmethod
    def init(rank=None, return_vecs=False):
        model = ElectraModel.from_pretrained('google/electra-base-discriminator')
        tokenizer = AutoTokenizer.from_pretrained('google/electra-base-discriminator')
        return BERTDotModel(model, tokenizer, rank, return_vecs)
    
    def transfer_state_dict(self, skeleton):
        skeleton.bi_encoder_model.model.load_state_dict(self.model.state_dict())
        
    def save_pretrained(self, path):
        self.model.save_pretrained(path)
    
    def train(self):
        self.model.train()
    
    def parameters(self):
        return self.model.parameters()
    
    def forward(self, x):
        query, docs, num_negatives = x 

        e_query = self.model(**query)[0][:, 0, :]
        e_docs = self.model(**docs)[0][:, 0, :]

        e_docs = e_docs.view(-1, num_negatives+1, e_query.shape[-1])
        
        score = torch.bmm(e_query.unsqueeze(1), e_docs.permute(0, 2, 1)).squeeze(1)

        if self.return_vecs:
            return (score, e_query, e_docs)
        return score, None, None

class BERTCatModel(nn.Module):
    def __init__(self, model, tokenizer, rank=None):
        super().__init__()
        self.device = torch.device('cuda', rank) if rank else torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
    
    @staticmethod
    def init(rank=None):
        model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator', num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained('google/electra-base-discriminator')
        return BERTCatModel(model, tokenizer, rank)

    def transfer_state_dict(self, skeleton):
        skeleton.model.load_state_dict(self.model.state_dict())

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
    
    def gen_labels(self, x):
        return torch.tensor([[0., 1.] if i % 2 == 0 else [1., 0.] for i in range(len(x))]).to(self.device)
    
    def train(self):
        self.model.train()
    
    def parameters(self):
        return self.model.parameters()
    
    def forward(self, x):
        x['labels'] = self.gen_labels(x['input_ids'])
        result = self.model(**x)
        return result.logits, result.loss

class DuoBERTModel(nn.Module):
    def __init__(self, model, tokenizer, rank=None):
        super().__init__()
        self.device = torch.device('cuda', rank) if rank else torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
    
    @staticmethod
    def init(rank=None):
        model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator', num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained('google/electra-base-discriminator')
        return DuoBERTModel(model, tokenizer, rank)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
    
    def gen_labels(self, x):
        return torch.tensor([[0., 1.] if i % 2 == 0 else [1., 0.] for i in range(len(x))]).to(self.device)
    
    def train(self):
        self.model.train()
    
    def parameters(self):
        return self.model.parameters()
    
    def forward(self, x):
        x['labels'] = self.gen_labels(x['input_ids'])
        output = self.model(**x)
        logits = output.logits
        result = F.softmax(logits, dim=1)[:, 1]
        return result, output.loss
        
class SPLADEModel(nn.Module):
    def __init__(self, model, tokenizer, rank=None):
        super().__init__()
        self.device = torch.device('cuda', rank) if rank else torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
    @staticmethod
    def init(rank=None):
        model = ElectraForSequenceClassification.from_pretrained('google/electra-base-discriminator', num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained('google/electra-base-discriminator')
        return DuoBERTModel(model, tokenizer, rank)



