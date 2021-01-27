import hypothesis as h
import numpy as np
import torch

from .base import BaseTrainer
from hypothesis.nn.ratio_estimation import BaseCriterion as Criterion
from hypothesis.nn.ratio_estimation import ConservativeCriterion
from hypothesis.util.data import NamedDataset



class RatioEstimatorTrainer(BaseTrainer):

    def __init__(self,
        estimator,
        optimizer,
        accelerator=h.accelerator,
        batch_size=h.default.batch_size,
        conservativeness=0.0,
        dataset_test=None,
        dataset_train=None,
        dataset_validate=None,
        epochs=h.default.epochs,
        logits=False,
        pin_memory=True,
        shuffle=True,
        workers=h.default.dataloader_workers):
        super(RatioEstimatorTrainer, self).__init__(
            accelerator=accelerator,
            batch_size=batch_size,
            dataset_test=dataset_test,
            dataset_train=dataset_train,
            dataset_validate=dataset_validate,
            epochs=epochs,
            pin_memory=pin_memory,
            shuffle=shuffle,
            workers=workers)
        # Verify the properties of the datasets
        if dataset_train is not None and not isinstance(dataset_train, NamedDataset):
            raise ValueError("The training dataset is not of the type `NamedDataset`.")
        if dataset_validate is not None and not isinstance(dataset_train, NamedDataset):
            raise ValueError("The validation dataset is not of the type `NamedDataset`.")
        if dataset_test is not None and not isinstance(dataset_train, NamedDataset):
            raise ValueError("The test dataset is not of the type `NamedDataset`.")
        # Basic trainer properties
        self._conservativeness = conservativeness
        self._estimator = estimator
        self._optimizer = optimizer
        # Optimization monitoring
        self._state_dict_best = None
        # Criterion properties
        if conservativeness > 0.0:
            self._criterion = ConservativeCriterion(
                batch_size=batch_size,
                conservativeness=conservativeness,
                estimator=estimator,
                logits=logits)
        else:
            self._criterion = Criterion(
                batch_size=batch_size,
                estimator=estimator,
                logits=logits)
        # Move to the specified accelerator
        self._criterion = self._criterion.to(accelerator)
        # Capture the best estimator
        self.add_event_handler(self.events.new_best_test, self._save_best_estimator_weights)

    @torch.no_grad()
    def _save_best_estimator_weights(self, trainer, **kwargs):
        self._estimator.eval()
        self._state_dict_best = self._estimator.cpu().state_dict()

    def _register_events(self):
        super()._register_events()

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def estimator(self):
        self._estimator.eval()
        return self._estimator

    @property
    def state_dict():
        return self.estimator.cpu().state_dict()

    @property
    def best_state_dict(self):
        return self._state_dict_best

    @property
    def best_estimator(self):
        if self._state_dict_best is not None:
            self._estimator.eval()
            estimator = self._estimator.cpu()
            estimator.load_state_dict(self._state_dict_best)
        else:
            estimator = None

        return estimator

    @torch.no_grad()
    def _estimator_cpu_state_dict(self):
        # Check if we're training a Data Parallel model.
        self.estimator = self.estimator.cpu()
        if isinstance(self.estimator, torch.nn.DataParallel):
            state_dict = self.estimator.module.state_dict()
        else:
            state_dict = self.estimator.state_dict()
        self.estimator = self.estimator.to(hypothesis.accelerator)

        return state_dict

    @property
    def conservativeness(self):
        return self._conservativeness

    @conservativeness.setter
    def conservativeness(self, value):
        assert value >= 0 and value <= 1
        self._conservativeness = value

    def train(self):
        assert self._dataset_train is not None
        self._estimator.train()
        loader = self._allocate_train_loader()
        losses = []
        total_batches = len(loader)
        for index, sample_joint in enumerate(loader):
            self.call_event(self.events.batch_train_start,
                            batch_index=index,
                            total_batches=total_batches)
            self._optimizer.zero_grad()
            for k, v in sample_joint.items():
                sample_joint[k] = v.to(self._accelerator, non_blocking=True)
            loss = self._criterion(**sample_joint)
            loss.backward()
            self._optimizer.step()
            loss = loss.item()
            losses.append(loss)
            self.call_event(self.events.batch_train_complete,
                            batch_index=index,
                            total_batches=total_batches,
                            loss=loss)
        expected_loss = np.mean(losses)

        return expected_loss

    @torch.no_grad()
    def validate(self):
        assert self._dataset_validate is not None
        self._estimator.eval()
        loader = self._allocate_validate_loader()
        losses = []
        total_batches = len(loader)
        for index, sample_joint in enumerate(loader):
            self.call_event(self.events.batch_validate_start,
                            batch_index=index,
                            total_batches=total_batches)
            for k, v in sample_joint.items():
                sample_joint[k] = v.to(self._accelerator, non_blocking=True)
            loss = self._criterion(**sample_joint).item()
            losses.append(loss)
            self.call_event(self.events.batch_validate_complete,
                            batch_index=index,
                            total_batches=total_batches,
                            loss=loss)
        expected_loss = np.mean(losses)

        return expected_loss

    @torch.no_grad()
    def test(self):
        assert self._dataset_test is not None
        self._estimator.eval()
        loader = self._allocate_test_loader()
        losses = []
        total_batches = len(loader)
        for index, sample_joint in enumerate(loader):
            self.call_event(self.events.batch_test_start,
                            batch_index=index,
                            total_batches=total_batches)
            for k, v in sample_joint.items():
                sample_joint[k] = v.to(self._accelerator, non_blocking=True)
            loss = self._criterion(**sample_joint).item()
            losses.append(loss)
            self.call_event(self.events.batch_test_complete,
                            batch_index=index,
                            total_batches=total_batches,
                            loss=loss)
        expected_loss = np.mean(losses)

        return expected_loss
