import hypothesis as h

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
        self._shuffle = shuffle
        self._workers = workers

    def _register_events(self):
        self.register_event("batch_train_complete")
        self.register_event("batch_train_start")
        self.register_event("batch_validate_complete")
        self.register_event("batch_validate_start")
        self.register_event("batch_test_complete")
        self.register_event("batch_test_start")
        self.register_event("epoch_complete")
        self.register_event("epoch_start")
        self.register_event("test_complete")
        self.register_event("test_start")
        self.register_event("train_complete")
        self.register_event("train_start")
        self.register_event("validate_complete")
        self.register_event("validate_start")

    def fit(self):
        for epoch in range(self.epochs):
            self._current_epoch = epoch
            self.call_event(self.events.epoch_start)
            # Training procedure
            if self._dataset_train is not None:
                self.call_event(self.events.train_start)
                self.train()
                self.call_event(self.events.train_complete)
            # Validation procedure
            if self._dataset_validate is not None:
                self.call_event(self.events.validate_start)
                self.validate()
                self.call_event(self.events.validate_complete)
            # Testing procedure
            if self._dataset_test is not None:
                self.call_event(self.events.test_start)
                self.test()
                self.call_event(self.events.test_complete)
            self.call_event(self.events.epoch_end)

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
            shuffle=self._shuffle)
