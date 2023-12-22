import torch
from torch import nn
import torch.nn.functional as F
from lexdistill.lsr.losses import Loss, num_non_zero

class MarginMSELoss(Loss):
    """
    The class for the MarginMSE loss.
    The MarginMSE is used distillation from a teacher ranking model (T) to a student model (S).
    MarginMSE(q,d1,d2) = MSE(S(q,d1)-S(q,d2), T(q,d1) - T(q,d2))
    """

    def __init__(self, q_regularizer=None, d_regularizer=None, num_negatives=1) -> None:
        """
        Constructing MarginMSELoss
        Parameters
        ----------
        q_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the query side. If q_regularizer is None, no regularization is applied on the query side.
        d_regularizer: lsr.losses.regularizer.Regularizer
            The regularizer on the document side. If d_regularizer is None, no regularization is applied on the document side.
        """
        super(MarginMSELoss, self).__init__(q_regularizer, d_regularizer)
        self.mse = nn.MSELoss()
        self.num_negatives = num_negatives

    def forward(self, q_reps, d_reps, labels):
        """
        Calculating the MarginMSE over a batch of query and document
        Parameters
        ----------
        q_reps: torch.Tensor
            batch of query vectors (size: batch_size x vocab_size)
        d_reps: torch.Tensor
            batch of document vectors (size: batch_size*2 x vocab_size).
            The number of documents needed is twice the number of query as we need a pair of documents for each query to calculate the margin.
            Documents in even positions (0, 2, 4...) are positive (relevant) documents, documents in odd positions (1, 3, 5...) are negative (non-relvant) documents.
        labels: torch.Tensor
            Teacher's margin between positive and negative documents. labels[i] = teacher(q_reps[i], d_reps[i*2]) - teacher(q_reps[i], d_reps[i*2+1])
        Returns
        -------
        tuple (loss, q_reg, d_reg, log)
            a tuple of averaged loss, query regularization, doc regularization and log (for experiment tracking)
        """
        batch_size = q_reps.size(0)
        e_q = q_reps.view(batch_size, -1)
        e_d = d_reps.view(batch_size, self.num_negatives+1, -1)
        labels = labels.view(batch_size, self.num_negatives+1)

        scores = torch.bmm(e_q.unsqueeze(1), e_d.transpose(1, 2)).squeeze(1)

        pos_score = scores[:, 0]
        neg_score = scores[:, 1:]

        x_margins = [pos_score - neg_score[:, i] for i in range(neg_score.shape[-1])]
        y_pos = labels[:, 0]
        y_neg = labels[:, 1:]
        y_margins = [y_pos - y_neg[:, j] for j in range(y_neg.shape[-1])]
        loss = torch.stack([F.mse_loss(x_margins[j], y_margins[j]) for j in range(len(x_margins))])
        mse_loss = torch.mean(loss)

        reg_q_output = (
            torch.tensor(0.0, device=q_reps.device)
            if (self.q_regularizer is None)
            else self.q_regularizer(q_reps)
        )
        reg_d_output = (
            torch.tensor(0.0, device=d_reps.device)
            if (self.d_regularizer is None)
            else (self.d_regularizer(d_reps))
        )
        if not self.q_regularizer is None:
            self.q_regularizer.step()
        if not self.d_regularizer is None:
            self.d_regularizer.step()

        to_log = {
            "query reg": reg_q_output.detach(),
            "doc reg": reg_d_output.detach(),
            "query length": num_non_zero(q_reps),
            "doc length": num_non_zero(d_reps),
            "loss_no_reg": mse_loss.detach(),
        }
        return (
            mse_loss,
            reg_q_output,
            reg_d_output,
            to_log,
        )