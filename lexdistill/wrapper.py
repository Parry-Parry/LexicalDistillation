import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

class MonoT5Model(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.rel = self.tokenizer.encode('true')[0]
        self.nrel = self.tokenizer.encode('false')[0]
    
    @staticmethod
    def init():
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        return MonoT5Model(model, tokenizer)

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
        return F.log_softmax(result, dim=1)[:, 0].cpu()

class BaselineT5(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.rel = self.tokenizer.encode('true')[0]
        self.nrel = self.tokenizer.encode('false')[0]
    
    @staticmethod
    def init():
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        return BaselineT5(model, tokenizer)

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

class DualMonoT5Model(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.rel = self.tokenizer.encode('true')[0]
        self.nrel = self.tokenizer.encode('false')[0]
    
    @staticmethod
    def init():
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        return DualMonoT5Model(model, tokenizer)

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
        return F.log_softmax(result, dim=1)[:, 0].cpu(), output.loss

class MonoBERTModel(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
    
    @staticmethod
    def init():
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        return MonoBERTModel(model, tokenizer)

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
        return F.log_softmax(result, dim=1)[:, 0].cpu()

class BaselineBERT(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.rel = self.tokenizer.encode('true')[0]
        self.nrel = self.tokenizer.encode('false')[0]
    
    @staticmethod
    def init():
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        return BaselineBERT(model, tokenizer)

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
        


