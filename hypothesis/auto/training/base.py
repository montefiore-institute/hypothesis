import hypothesis

from hypothesis.engine import Procedure
from torch.utils.data import DataLoader



class BaseTrainer(Procedure):

    def __init__(self,
        batch_size=hypothesis.default.batch_size,
        checkpoint=None,
        epochs=hypothesis.default.epochs,
        identifier=None,
        workers=hypothesis.default.dataloader_workers):
        super(BaseTrainer, self).__init__()
        # Training hyperparameters
        self.identifier = identifier
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint
        self.dataloader_workers = workers
        self.epochs = epochs
        # Load the previously saved state.
        self._checkpoint_load()

    def _checkpoint_store(self):
        raise NotImplementedError

    def _checkpoint_load(self):
        raise NotImplementedError

    def _allocate_data_loader(self, dataset):
        return DataLoader(dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            shuffle=True)

    def _register_events(self):
        raise NotImplementedError

    def _summarize(self):
        raise NotImplementedError

    def checkpoint(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
