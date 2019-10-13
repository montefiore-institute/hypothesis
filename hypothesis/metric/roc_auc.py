import numpy as np
import torch

from hypothesis.metric import BaseMetric
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
from sklearn.metrics import roc_curve as sklearn_roc_curve



def roc_auc_score(predictions, targets):
    predictions = predictions.view(-1, 1).numpy()
    targets = targets.view(-1, 1).numpy()

    return sklearn_roc_auc_score(targets, predictions)


def roc_curve(predictions, targets):
    predictions = predictions.view(-1, 1).numpy()
    targets = targets.view(-1, 1).numpy()
    fpr, tpr, _ = sklearn_roc_curve(targets, predictions)

    return fpr, tpr
