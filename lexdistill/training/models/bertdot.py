from transformers import PreTrainedModel, PretrainedConfig, ElectraModel
from os.path import join

class dotConfig(PretrainedConfig):
    model_type = "Encoder"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BERTdot(PreTrainedModel):
    def __init__(
        self,
        encoder: PreTrainedModel,
        config: dotConfig = dotConfig(),
    ):
        super().__init__(config)
        self.encoder = encoder
    
    def encode(self, **text):
        return self.encoder(**text).last_hidden_states[:, 0, :]

    def forward(self, loss, queries, docs_batch, labels=None):
        """Compute the loss given (queries, docs, labels)"""
        q_reps = self.encode(**queries)
        docs_batch_rep = self.encode(**docs_batch)
        if labels is None:
            output = loss(q_reps, docs_batch_rep)
        else:
            output = loss(q_reps, docs_batch_rep, labels)
        return output

    def save_pretrained(self, model_dir):
        """Save both query and document encoder"""
        self.config.save_pretrained(model_dir)
        self.encoder.save_pretrained(model_dir)

    @classmethod
    def from_pretrained(cls, model_dir_or_name):
        """Load encoder from a directory"""
        config = dotConfig.from_pretrained(model_dir_or_name)
        encoder = ElectraModel.from_pretrained(model_dir_or_name)
        return cls(encoder, config)