class MetricTracker:
    def __init__(self, metric, val_topics, qrels, patience=10):
        self.metric = metric
        self.val_topics = val_topics
        self.qrels = qrels
        self.patience = patience
        self.history = []

    def compute_metric(self, ranks):
        metric = None
        return metric
    
    def __call__(self, model, step):
        ranks = model.transform(self.val_topics)
        value = self.compute_metric(ranks)

        # TODO: implement early stopping

        if step % self.patience == 0: 
            self.history = [value]
        else: 
            self.history.append(value)
        return 0