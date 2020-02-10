import hypothesis

from hypothesis.engine import Procedure



class BaseTrainer(Procedure):

    def __init__(self,
        batch_size=hypothesis.default.batch_size,
        checkpoint=None,
        epochs=hypothesis.default.epochs,
        workers=hypothesis.default.dataloader_workers):
        super(BaseTrainer, self).__init__()
        # Training hyperparameters
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint
        self.dataloader_workers = workers
        self.epochs = epochs
        # Trainer state
        self.current_epoch = 0
        self.epochs_remaining = self.epochs
        # Load the previously saved state.
        if self.checkpoint_path is not None:
            self._checkpoint_load()

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
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
