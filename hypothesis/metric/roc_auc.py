import hypothesis
import numpy as np
import torch

from hypothesis.metric import BaseStateMetric
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
from sklearn.metrics import roc_curve as sklearn_roc_curve
from torch.utils.data import DataLoader



def roc_curve(predictions, targets):
    r""""""
    predictions = predictions.view(-1, 1).numpy()
    targets = targets.view(-1, 1).numpy()
    fpr, tpr, _ = sklearn_roc_curve(targets, predictions)

    return fpr, tpr



def roc_auc_score(predictions, targets):
    r""""""
    predictions = predictions.view(-1, 1).numpy()
    targets = targets.view(-1, 1).numpy()

    return sklearn_roc_auc_score(targets, predictions)



class AreaUnderCurveMetric(BaseStateMetric):
    r""""""

    def __init__(self, model, dataset, batch_size=32, workers=1):
        super(AreaUnderCurveMetric, self).__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.model = model
        self.workers = workers

    def update(self):
        predictions = []
        targets = []
        self.model.eval()
        data_loader = DataLoader(dataset,
            batch_size=self.batch_size,
            shuffle=True,
            workers=self.workers)
        for x, y in data_loader:
            x = x.to(hypothesis.accelerator, non_blocking=True)
            y = y.to(hypothesis.accelerator, non_blocking=True)
            y_hat = self.model(x)
            predictions.append(y_hat.cpu())
            targets.append(y.cpu())
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        auc = roc_auc_curve(predictions, targets)
        self.update(auc)
