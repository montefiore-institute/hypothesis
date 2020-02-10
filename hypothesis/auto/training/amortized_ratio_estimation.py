import hypothesis
import os
import torch

from .base import BaseTrainer
from hypothesis.nn.amortized_ratio_estimation import LikelihoodToEvidenceCriterion



class BaseAmortizedRatioEstimatorTrainer(BaseTrainer):

    def __init__(self,
        criterion,
        estimator,
        feeder,
        optimizer,
        dataset_train,
        accelerator=hypothesis.accelerator,
        batch_size=hypothesis.default.batch_size,
        checkpoint=None,
        dataset_test=None,
        epochs=hypothesis.default.epochs,
        lr_scheduler=None,
        workers=hypothesis.default.dataloader_workers):
        super(BaseAmortizedRatioEstimatorTrainer, self).__init__(
            batch_size=batch_size,
            checkpoint=checkpoint,
            epochs=epochs,
            workers=workers)
        # Datasets
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        # Trainer state
        self.accelerator = accelerator
        self.criterion = criterion(estimator, self.batch_size)
        self.current_epoch = 0
        self.epochs_remaining = self.epochs
        self.estimator = estimator
        self.feeder = feeder
        self.losses_test = []
        self.losses_train = []
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.best_epoch = None
        self.best_loss = float("infinity")
        self.best_model = None
        # Move estimator and criterion to the specified accelerator.
        self.estimator = self.estimator.to(self.accelerator)
        self.criterion = self.criterion.to(self.accelerator)

    def _register_events(self):
        pass

    def _valid_checkpoint_path(self):
        return self.checkpoint_path is not None and len(checkpoint_path) > 0

    def _valid_checkpoint_path_and_exists(self):
        return self._valid_checkpoint_path() and os.path.exists(self.checkpoint_path)

    def _checkpoint_store(self):
        if self._valid_checkpoint_path():
            state = {}
            state["accelerator"] = self.accelerator
            state["current_epoch"] = self.current_epoch
            state["estimator"] = self._cpu_estimator_state_dict()
            state["epochs_remaining"] = self.epochs_remaining
            state["losses_test"] = self.losses_test
            state["losses_train"] = self.losses_train
            if self.lr_scheduler is not None:
                state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["optimizer"] = self.optimizer.state_dict()
            state["best_epoch"] = self.best_epoch
            state["best_loss"] = self.best_loss
            state["best_model"] = self.best_model
            torch.save(state, self.checkpoint_path)

    def _checkpoint_load(self):
        # Check if checkpoint path exists.
        if self._valid_checkpoint_path_and_exists():
            state = torch.load(self.checkpoint_path)
            self.accelerator = state["accelerator"]
            self.current_epoch = state["current_epoch"]
            self.estimator.load_state_dict(state["estimator"])
            self.epochs_remaining = state["epochs_remaining"]
            self.losses_test = state["losses_test"]
            self.losses_train = state["losses_train"]
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_epoch = state["best_epoch"]
            self.best_loss = state["best_loss"]
            self.best_model = state["best_model"]

    def _summarize(self):
        raise NotImplementedError

    def _cpu_estimator_state_dict(self):
        state_dict = self.estimator.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()

        return state_dict

    def checkpoint(self):
        self._checkpoint_store()

    def optimize(self):
        # Training procedure
        for epoch in range(self.epochs_remaining):
            self.current_epoch = epoch
            self.train()
            # Check if a testing dataset is available.
            if self.dataset_test is not None:
                self.test()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.checkpoint()
        # Remove the checkpoint.
        if self._valid_checkpoint_path_and_exists():
            os.remove(self.checkpoint_path)

        return self._summarize()

    def test(self):
        self.estimator.eval()
        loader = self._allocate_data_loader(self.dataset_test)
        total_loss = 0.0
        for batch in loader:
            loss = self.feeder(
                accelerator=self.accelerator,
                batch=batch,
                criterion=self.criterion)
            total_loss += loss.item()
        total_loss /= len(self.dataset_test)
        if total_loss < self.best_loss:
            state_dict = self._cpu_estimator_state_dict()
            self.best_loss = total_loss
            self.best_model = state_dict
            self.best_epoch = self.current_epoch

    def train(self):
        self.estimator.train()
        loader = self._allocate_data_loader(self.dataset_train)
        for batch in loader:
            loss = self.feeder(
                accelerator=self.accelerator,
                batch=batch,
                criterion=self.criterion)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.losses_train.append(loss.item())



class LikelihoodToEvidenceRatioEstimatorTrainer(BaseAmortizedRatioEstimatorTrainer):

    def __init__(self,
        estimator,
        optimizer,
        dataset_train,
        accelerator=hypothesis.accelerator,
        batch_size=hypothesis.default.batch_size,
        checkpoint=None,
        dataset_test=None,
        epochs=hypothesis.default.epochs,
        lr_scheduler=None,
        workers=hypothesis.default.dataloader_workers):
        feeder = LikelihoodToEvidenceRatioEstimatorTrainer.feeder
        criterion = LikelihoodToEvidenceCriterion
        super(LikelihoodToEvidenceRatioEstimatorTrainer, self).__init__(
            accelerator=accelerator,
            batch_size=batch_size,
            criterion=criterion,
            dataset_test=dataset_test,
            dataset_train=dataset_train,
            epochs=epochs,
            estimator=estimator,
            feeder=feeder,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            workers=workers)

    @staticmethod
    def feeder(batch, criterion, accelerator):
        inputs, outputs = batch
        inputs = inputs.to(accelerator, non_blocking=True)
        outputs = outputs.to(accelerator, non_blocking=True)

        return criterion(inputs=inputs, outputs=outputs)
