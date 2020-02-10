import hypothesis
import os
import torch

from hypothesis.engine import Procedure



class BaseTrainer(Procedure):

    def __init__(self,
        criterion,
        dataset_train,
        feeder,
        optimizer,
        batch_size=hypothesis.default.batch_size,
        checkpoint=None,
        dataset_test=None,
        epochs=hypothesis.default.epochs,
        lr_scheduler=None,
        workers=hypothesis.default.dataloader_workers):
        super(BaseTrainer, self).__init__()
        # Datasets
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        # Training hyperparameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_path = checkpoint
        self.dataloader_workers = workers
        # Trainer state
        self.criterion = criterion
        self.current_epoch = 0
        self.epochs_remaining = self.epochs
        self.estimator = criterion.estimator
        self.feeder = feeder
        self.losses_test = []
        self.losses_train = []
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.best_epoch = None
        self.best_loss = float("infinity")
        self.best_model = None

    def _checkpoint_store(self):
        raise NotImplementedError

    def _checkpoint_load(self):
        raise NotImplementedError

    def _allocate_data_loader(self, dataset):
        return Dataset(dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.dataloader_workers,
            pin_memory=True)

    def _register_events(self):
        raise NotImplementedError

    def _summarize(self):
        raise NotImplementedError

    def checkpoint(self):
        raise NotImplementedError

    def optimize(self):
        for epoch in range(self.epochs_remaining):
            self.current_epoch = epoch
            self.train()
            # Check if a testing dataset is available.
            if self.dataset_test is not None:
                self.test()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # self.checkpoint()

        return self._summarize()

    def test(self):
        self.estimator.test()
        loader = self._allocate_data_loader(self.dataset_test)
        total_loss = 0.0
        for batch in loader:
            loss = self.feeder(batch)
            total_loss += loss.item()
        total_loss /= len(dataset_test)
        if total_loss < self.loss_best:
            state_dict = self.estimator.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            self.best_loss = total_loss
            self.best_model = state_dict
            self.best_epoch = self.current_epoch

    def train(self):
        self.estimator.train()
        loader = self._allocate_data_loader(self.dataset_train)
        for batch in loader:
            loss = self.feeder(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses_train.append(loss.item())
