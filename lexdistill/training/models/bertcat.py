from transformers import PreTrainedModel, PretrainedConfig, ElectraForSequenceClassification
from os.path import join

class catConfig(PretrainedConfig):
    model_type = "Classifier"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BERTcat(PreTrainedModel):
    def __init__(
        self,
        classifier: PreTrainedModel,
        config: catConfig = catConfig(),
    ):
        super().__init__(config)
        self.classifier = classifier

    def forward(self, loss, batch, labels=None):
        """Compute the loss given (pairs, labels)"""
        logits = self.classifier(**batch).logits
        if labels is None:
            output = loss(logits)
        else:
            output = loss(logits, labels)
        return output

    def save_pretrained(self, model_dir):
        """Save classifier"""
        self.config.save_pretrained(model_dir)
        self.classifier.save_pretrained(model_dir)

    @classmethod
    def from_pretrained(cls, model_dir_or_name):
        """Load classifier from a directory"""
        config = catConfig.from_pretrained(model_dir_or_name)
        classifier = ElectraForSequenceClassification.from_pretrained(model_dir_or_name, num_labels=2)
        return cls(classifier, config)