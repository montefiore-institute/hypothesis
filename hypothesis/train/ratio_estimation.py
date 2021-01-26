import hypothesis as h
import torch

from .base import BaseTrainer
from hypothesis.nn.ratio_estimation import BaseCriterion



class RatioEstimatorTrainer(BaseTrainer):

    def __init__(self,
        estimator,
        optimizer,
        conservativeness=0.0,
        accelerator=h.accelerator,
        batch_size=h.default.batch_size,
        dataset_test=None,
        dataset_validate=None,
        dataset_train=None,
        epochs=h.default.epochs,
        shuffle=True,
        workers=h.default.dataloader_workers):
        super(RatioEstimatorTrainer, self).__init__(
            accelerator=accelerator,
            batch_size=batch_size,
            dataset_test=dataset_test,
            dataset_train=dataset_train,
            dataset_validate=dataset_validate,
            epochs=epochs,
            shuffle=shuffle,
            workers=workers)
        self._conservativeness = conservativeness
        self._estimator = estimator
        self._optimizer = optimizer

    @property
    def conservativeness(self):
        return self._conservativeness

    @conservativeness.setter
    def conservativeness(self, value):
        assert value >= 0 and value <= 1
        self._conservativeness = value

    def train(self):
        assert self._dataset_train is not None
        loader = self._allocate_train_loader()

    def validate(self):
        assert self._dataset_validate is not None
        loader = self._allocate_validate_loader()

    def test(self):
        assert self._dataset_test is not None
        loader = self._allocate_test_loader()
