from typing import Any
import torch
import torch.nn.functional as F
import numpy as np

# create margin function which works with multiple negatives (i.e. x.shape[-1] > 1)
def margin(x):
    return [x[::2, i] - x[1::2, i] for i in range(x.shape[-1])]

def aggregate(x):
    return [np.mean(x)]

def MarginMSELoss(x, y):
    student_margin = margin(x)
    losses = [F.mse_loss(student_margin, margin(y[:, i]))for i in range(y.shape[-1])]
    return torch.mean(torch.stack(losses))

class MarginMultiLoss:
    def __init__(self, batch_size : int, num_negatives : int = 1) -> None:
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        
    def __call__(self, x, y) -> Any:
        x = x.view(self.batch_size, -1)
        y = y.view(y.shape[-1], self.batch_size, x.shape[-1])

        x_pos = x[:, 0]
        x_neg = x[:, 1:]

        x_margins = [x_pos - x_neg[:, i] for i in range(x_neg.shape[-1])]

        loss = []
        for i in range(y.shape[0]):
            tmp_y = y[i]
            y_pos = tmp_y[:, 0]
            y_neg = tmp_y[:, 1:]
            y_margins = [y_pos - y_neg[:, j] for j in range(y_neg.shape[-1])]
            loss.append(torch.stack([F.mse_loss(x_margins[j], y_margins[j]) for j in range(len(x_margins))]))
        
        return torch.mean(torch.stack(loss))

class InBatchLoss:
    def __init__(self, batch_size : int, num_negatives : int = 1) -> None:
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        
    def __call__(self, query_vecs, doc_vecs) -> Any:
        query_vecs = query_vecs.view(self.batch_size, -1)
        doc_vecs = doc_vecs.view(self.batch_size, -1, query_vecs.shape[-1])
        pos_vecs = doc_vecs[:, 0]

        select  = torch.ones((pos_vecs.shape[0], pos_vecs.shape[0]),device=pos_vecs.device)
        select.fill_diagonal_(0)
        select = select.bool()

        scores = torch.mm(query_vecs, pos_vecs.transpose(-2,-1))[select]
        return torch.sum(scores)
    