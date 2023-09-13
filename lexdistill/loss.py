import torch
import torch.nn.functional as F
import numpy as np
margin = lambda x : x[::2] - x[1::2]

def aggregate(x):
    return [np.mean(x)]

def MarginMSELoss(x, y):
    student_margin = margin(x)
    losses = [F.mse_loss(student_margin, margin(y[:, i]))for i in range(y.shape[-1])]
    return torch.mean(torch.stack(losses))