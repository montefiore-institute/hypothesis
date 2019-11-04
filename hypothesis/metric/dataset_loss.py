import hypothesis
import torch

from hypothesis.metric import BaseStateMetric
from torch.utils.data import DataLoader



def _default_batch_handler(model, criterion, batch):
    x, y = batch
    x = x.to(hypothesis.accelerator, non_blocking=True)
    y = y.to(hypothesis.accelerator, non_blocking=True)
    y_hat = model(x)
    loss = criterion(y, y_hat)

    return loss



class DatasetLossMetric(BaseStateMetric):
    r""""""

    def __init__(self, model, criterion, dataset,
        batch_size=32,
        workers=1,
        batch_handler=_default_batch_handler):
        super(DatasetLossMetric).__init__()
        self.batch_size = batch_size
        self.criterion = criterion
        self.dataset = dataset
        self.model = model
        self.workers = 1
        self.process_batch = _default_batch_handler

    def update(self):
        self.model.eval()
        losses = []

        with torch.no_grad():
            data_loader = DataLoader(dataset,
                batch_size=self.batch_size,
                shuffle=True,
                workers=self.workers)
            for batch in data_loader:
                loss = self.process_batch(batch)
                losses.append(loss.cpu())
            del data_loader
            losses = torch.cat(losses, dim=0)
            loss = losses.mean().item()
        self.update(loss)
