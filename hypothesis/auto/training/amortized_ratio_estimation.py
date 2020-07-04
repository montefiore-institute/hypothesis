import hypothesis
import numpy as np
import os
import torch

from .base import BaseTrainer
from hypothesis.nn.amortized_ratio_estimation import BaseCriterion
from hypothesis.nn.amortized_ratio_estimation import LikelihoodToEvidenceCriterion
from hypothesis.summary import TrainingSummary as Summary



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
        lr_scheduler_epoch=None,
        lr_scheduler_update=None,
        identifier=None,
        workers=hypothesis.default.dataloader_workers):
        super(BaseAmortizedRatioEstimatorTrainer, self).__init__(
            identifier=identifier,
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
        self.lr_scheduler_epoch = lr_scheduler_epoch
        self.lr_scheduler_update = lr_scheduler_update
        self.optimizer = optimizer
        self.best_epoch = None
        self.best_loss = float("infinity")
        self.best_model = None
        # Move estimator and criterion to the specified accelerator.
        self.estimator = self.estimator.to(self.accelerator)
        self.criterion = self.criterion.to(self.accelerator)

    def _register_events(self):
        self.register_event("epoch_start")
        self.register_event("epoch_complete")

    def _valid_checkpoint_path(self):
        return self.checkpoint_path is not None and len(self.checkpoint_path) > 0

    def _valid_checkpoint_path_and_exists(self):
        return self._valid_checkpoint_path() and os.path.exists(self.checkpoint_path)

    def _checkpoint_store(self):
        if self._valid_checkpoint_path():
            state = {}
            state["accelerator"] = self.accelerator
            state["current_epoch"] = self.current_epoch
            state["estimator"] = self._cpu_estimator_state_dict()
            state["epochs_remaining"] = self.epochs_remaining
            state["epochs"] = self.epochs
            state["losses_test"] = self.losses_test
            state["losses_train"] = self.losses_train
            if self.lr_scheduler_update is not None:
                state["lr_scheduler_update"] = self.lr_scheduler_update.state_dict()
            if self.lr_scheduler_epoch is not None:
                state["lr_scheduler_epoch"] = self.lr_scheduler_epoch.state_dict()
            state["optimizer"] = self.optimizer.state_dict()
            state["best_epoch"] = self.best_epoch
            state["best_loss"] = self.best_loss
            state["best_model"] = self.best_model
            torch.save(state, self.checkpoint_path)

    def _checkpoint_load(self):
        if self._valid_checkpoint_path_and_exists():
            raise NotImplementedError

    def _summarize(self):
        return Summary(
            identifier=self.identifier,
            model_best=self.best_model,
            model_final=self.estimator.cpu().state_dict(),
            epoch_best=self.best_epoch,
            epochs=self.epochs,
            losses_train=np.array(self.losses_train).reshape(-1),
            losses_test=np.array(self.losses_test).reshape(-1))

    def _cpu_estimator_state_dict(self):
        state_dict = self.estimator.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()

        return state_dict

    def checkpoint(self):
        self._checkpoint_store()

    def fit(self):
        # Training procedure
        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1
            self.call_event(self.events.epoch_start, trainer=self)
            self.train()
            # Check if a testing dataset is available.
            if self.dataset_test is not None:
                self.test()
            else:
                self.best_model = self._cpu_estimator_state_dict()
            # Check if a learning rate scheduler has been allocated.
            if self.lr_scheduler_update is not None:
                self.lr_scheduler_update.step()
            self.remaining_epochs -= 1
            self.checkpoint()
            self.call_event(self.events.epoch_complete, trainer=self)
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
        total_loss /= len(loader)
        self.losses_test.append(total_loss)
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
            if self.lr_scheduler_epoch is not None:
                self.lr_scheduler_epoch.step()
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
        lr_scheduler_epoch=None,
        lr_scheduler_update=None,
        identifier=None,
        workers=hypothesis.default.dataloader_workers):
        feeder = LikelihoodToEvidenceRatioEstimatorTrainer.feeder
        criterion = LikelihoodToEvidenceCriterion
        super(LikelihoodToEvidenceRatioEstimatorTrainer, self).__init__(
            accelerator=accelerator,
            batch_size=batch_size,
            checkpoint=checkpoint,
            criterion=criterion,
            dataset_test=dataset_test,
            dataset_train=dataset_train,
            epochs=epochs,
            estimator=estimator,
            feeder=feeder,
            identifier=identifier,
            lr_scheduler_epoch=lr_scheduler_epoch,
            lr_scheduler_update=lr_scheduler_update,
            optimizer=optimizer,
            workers=workers)

    @staticmethod
    def feeder(batch, criterion, accelerator):
        inputs, outputs = batch
        inputs = inputs.to(accelerator, non_blocking=True)
        outputs = outputs.to(accelerator, non_blocking=True)

        return criterion(inputs=inputs, outputs=outputs)



def create_trainer(denominator, feeder):
    # Create the criterion with the specified denominator.
    class Criterion(BaseCriterion):

        def __init__(self, estimator, batch_size=hypothesis.default.batch_size, logits=False):
            super(Criterion, self).__init__(
                batch_size=batch_size,
                denominator=denominator,
                estimator=estimator,
                logits=logits)
    # Create the trainer object with the desired criterion.
    class Trainer(BaseAmortizedRatioEstimatorTrainer):

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
            identifier=None,
            workers=hypothesis.default.dataloader_workers):
            criterion = Criterion
            super(Trainer, self).__init__(
                accelerator=accelerator,
                batch_size=batch_size,
                checkpoint=checkpoint,
                criterion=criterion,
                dataset_test=dataset_test,
                dataset_train=dataset_train,
                epochs=epochs,
                estimator=estimator,
                feeder=feeder,
                identifier=identifier,
                lr_scheduler=lr_scheduler,
                optimizer=optimizer,
                workers=workers)

    return Criterion, Trainer
