import hypothesis as h
import numpy as np

from hypothesis.engine import Procedure
from torch.utils.data import DataLoader



class BaseTrainer(Procedure):

    def __init__(self,
        accelerator=h.accelerator,
        batch_size=h.default.batch_size,
        dataset_test=None,
        dataset_validate=None,
        dataset_train=None,
        epochs=h.default.epochs,
        pin_memory=True,
        shuffle=True,
        workers=h.default.dataloader_workers):
        super(BaseTrainer, self).__init__()
        self._accelerator = accelerator
        self._batch_size = batch_size
        self._current_epoch = None
        self._dataset_test = dataset_test
        self._dataset_train = dataset_train
        self._dataset_validate = dataset_validate
        self._epochs = epochs
        self._losses_test = []
        self._losses_train = []
        self._losses_validate = []
        self._pin_memory = pin_memory
        self._shuffle = shuffle
        self._workers = workers

    def _register_events(self):
        self.register_event("batch_test_complete")
        self.register_event("batch_test_start")
        self.register_event("batch_train_complete")
        self.register_event("batch_train_start")
        self.register_event("batch_validate_complete")
        self.register_event("batch_validate_start")
        self.register_event("epoch_complete")
        self.register_event("epoch_start")
        self.register_event("fit_complete")
        self.register_event("fit_start")
        self.register_event("new_best_test")
        self.register_event("new_best_train")
        self.register_event("new_best_validate")
        self.register_event("test_complete")
        self.register_event("test_start")
        self.register_event("train_complete")
        self.register_event("train_start")
        self.register_event("validate_complete")
        self.register_event("validate_start")

    def fit(self):
        self.call_event(self.events.fit_start)
        for epoch in range(1, self._epochs + 1, 1):
            self._current_epoch = epoch
            self.call_event(self.events.epoch_start, epoch=epoch)
            # Training procedure
            if self._dataset_train is not None:
                self.call_event(self.events.train_start)
                loss = self.train()
                if len(self._losses_train) == 0 or loss < np.min(self._losses_train):
                    self.call_event(self.events.new_best_train, loss=loss)
                self._losses_train.append(loss)
                self.call_event(self.events.train_complete)
            # Validation procedure
            if self._dataset_validate is not None:
                self.call_event(self.events.validate_start)
                loss = self.validate()
                if len(self._losses_validate) == 0 or loss < np.min(self._losses_validate):
                    self.call_event(self.events.new_best_validate, loss=loss)
                self._losses_validate.append(loss)
                self.call_event(self.events.validate_complete)
            # Testing procedure
            if self._dataset_test is not None:
                self.call_event(self.events.test_start)
                loss = self.test()
                if len(self._losses_test) == 0 or loss < np.min(self._losses_test):
                    self.call_event(self.events.new_best_test, loss=loss)
                self._losses_test.append(loss)
                self.call_event(self.events.test_complete)
            self.call_event(self.events.epoch_complete, epoch=epoch)
        self.call_event(self.events.fit_complete)

    def train(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def _allocate_train_loader(self):
        if self._dataset_train is not None:
            return self._allocate_data_loader(self._dataset_train)
        else:
            return None

    def _allocate_validate_loader(self):
        if self._dataset_validate is not None:
            return self._allocate_data_loader(self._dataset_validate)
        else:
            return None

    def _allocate_test_loader(self):
        if self._dataset_test is not None:
            return self._allocate_data_loader(self._dataset_test)
        else:
            return None

    def _allocate_data_loader(self, dataset):
        return DataLoader(dataset,
            batch_size=self._batch_size,
            drop_last=True,
            num_workers=self._workers,
            pin_memory=self._pin_memory,
            shuffle=self._shuffle)

    @property
    def losses_test(self):
        return np.array(self._losses_test)

    @property
    def losses_train(self):
        return np.array(self._losses_train)

    @property
    def losses_validate(self):
        return np.array(self._losses_validate)

    @property
    def current_epoch(self):
        return self._current_epoch
