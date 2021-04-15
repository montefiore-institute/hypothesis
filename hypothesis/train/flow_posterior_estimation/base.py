import hypothesis as h
import numpy as np
import torch

from hypothesis.nn.ratio_estimation import FlowPosteriorCriterion
from hypothesis.nn.ratio_estimation import ConservativeCriterion
from hypothesis.train import BaseTrainer
from hypothesis.util.data import NamedDataset
from tqdm import tqdm



class FlowPosteriorEstimatorTrainer(BaseTrainer):

    def __init__(self,
        estimator,
        optimizer,
        accelerator=h.accelerator,
        batch_size=h.default.batch_size,
        calibrate=True,
        conservativeness=0.0,
        dataset_test=None,
        dataset_train=None,
        dataset_validate=None,
        epochs=h.default.epochs,
        gamma=25.0,
        logits=False,
        pin_memory=True,
        shuffle=True,
        show=False,
        workers=h.default.dataloader_workers):
        super(FlowPosteriorEstimatorTrainer, self).__init__(
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
        self._conservativenesses = []
        self._estimator = estimator
        self._optimizer = optimizer
        # Optimization monitoring
        self._state_dict_best = None
        # Criterion properties
        self._criterion = FlowPosteriorCriterion(batch_size=batch_size, estimator=estimator)
        '''self._criterion = ConservativeCriterion(
            batch_size=batch_size,
            calibrate=calibrate,
            conservativeness=conservativeness,
            estimator=estimator,
            gamma=gamma,
            logits=logits)'''
        # Move to the specified accelerator
        self._criterion = self._criterion.to(accelerator)
        # Capture the best estimator
        self.add_event_handler(self.events.new_best_test, self._save_best_estimator_weights)
        # Check if debugging information needs to be shown.
        if show:
            self._progress_top = tqdm()
            self._progress_bottom = tqdm()
            self._progress_bottom_prefix = None
            self._add_display_hooks()
        else:
            self._progress_top = None
            self._progress_bottom = None

    def _init_progress_bottom(self, prefix, total=None):
        if self._progress_bottom is not None:
            self._progress_bottom_prefix = prefix
            self._progress_bottom.set_description(prefix)
            self._progress_bottom.total = total
            self._progress_bottom.reset()
            self._progress_bottom.refresh()

    @torch.no_grad()
    def _add_display_hooks(self):
        # Initialize the top progress bar
        self._progress_top.set_description("Epochs")
        self._progress_top.total = self.epochs
        self._progress_top.reset()
        self._progress_top.refresh()
        # Define the init hooks
        def start_training(trainer, **kwargs):
            self._init_progress_bottom("Training")
        def start_testing(trainer, **kwargs):
            self._init_progress_bottom("Testing")
        def start_validation(trainer, **kwargs):
            self._init_progress_bottom("Validation")
        # Define the update hooks
        @torch.no_grad()
        def update_batch(trainer, loss, batch_index, total_batches, **kwargs):
            if batch_index == 0:
                self._progress_bottom.total = total_batches
            self._progress_bottom.set_description(self._progress_bottom_prefix + " ~ current loss {:.4f}".format(loss))
            self._progress_bottom.update()
        @torch.no_grad()
        def update_epoch(trainer, **kwargs):
            epoch = trainer.current_epoch
            if len(trainer.losses_test) > 0:
                best_loss = np.min(trainer.losses_test)
                self._progress_top.set_description("Epochs ~ best test loss: {:.4f}".format(best_loss))
            self._progress_top.update()
        # Register the hooks
        self.add_event_handler(self.events.batch_test_complete, update_batch)
        self.add_event_handler(self.events.batch_train_complete, update_batch)
        self.add_event_handler(self.events.batch_validate_complete, update_batch)
        self.add_event_handler(self.events.epoch_complete, update_epoch)
        self.add_event_handler(self.events.test_start, start_testing)
        self.add_event_handler(self.events.train_start, start_training)
        self.add_event_handler(self.events.validate_start, start_validation)

    @torch.no_grad()
    def _save_best_estimator_weights(self, trainer, **kwargs):
        self._state_dict_best = self._estimator_cpu_state_dict()

    def _register_events(self):
        super()._register_events()

    @property
    def conservativenesses(self):
        return self._conservativenesses

    @property
    def conservativeness(self):
        return self._criterion.conservativeness

    @conservativeness.setter
    def conservativeness(self, value):
        # Verify constraints
        if value > 1.0:
            value = 1.0
        elif value < 0:
            value = 0.0
        self._criterion.conservativeness = value

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def estimator(self):
        self._estimator.eval()
        return self._estimator

    @property
    def state_dict(self):
        return self._estimator_cpu_state_dict()

    @property
    def best_state_dict(self):
        return self._state_dict_best

    @property
    def best_estimator(self):
        if self._state_dict_best is not None:
            estimator = self._estimator.cpu()
            estimator.eval()
            estimator.load_state_dict(self._state_dict_best)
        else:
            estimator = None

        return estimator

    @torch.no_grad()
    def _estimator_cpu_state_dict(self):
        # Check if we're training a Data Parallel model.
        self._estimator.eval()
        self._estimator = self._estimator.cpu()
        if isinstance(self._estimator, torch.nn.DataParallel):
            state_dict = self._estimator.module.state_dict()
        else:
            state_dict = self._estimator.state_dict()
        # Move back to the original device
        self._estimator = self._estimator.to(self.accelerator)

        return state_dict

    def train(self):
        assert self._dataset_train is not None
        self._estimator.train()
        self._conservativenesses.append(self.conservativeness)
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
