import torch
import hypothesis
import os

from torch.utils.data import DataLoader



class Trainer:
    r"""Base ``Trainer`` interface."""

    def __init__(self, dataset, allocate_optimizer, epochs=1, data_workers=2,
                 batch_size=32, checkpoint=None, validate=None,
                 allocate_scheduler=None):
        self.allocate_optimizer = allocate_optimizer
        self.allocate_scheduler = allocate_scheduler
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.data_workers = data_workers
        self.dataset = dataset
        self.epochs = epochs
        self.lr_scheduler = None
        self.model = None
        self.validate = validate

    def _validate(self):
        if self.supports_validation():
            hypothesis.call_hooks(hypothesis.hooks.pre_validation, self)
            validation_result = self.validate()
            hypothesis.call_hooks(hypothesis.hooks.post_validation, self, validation_result=validation_result)

    def supports_checkpointing(self):
        return self.checkpoint is not None

    def supports_validation(self):
        return self.validate is not None

    def reset(self):
        if self.allocate_optimizer is not None:
            self.optimizer = self.allocate_optimizer(self.model)
        if self.allocate_scheduler is not None:
            self.lr_scheduler = self.allocate_scheduler(self.optimizer)

    def scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def dataset_iterations(self):
        return len(self.dataset)

    def epoch(self, epoch):
        # Perform an LR scheduling step.
        self.scheduler_step()
        loader = iter(DataLoader(
            self.dataset, num_workers=self.data_workers,
            batch_size=self.batch_size))
        num_iterations = self.dataset_iterations()
        for iteration in range(num_iterations):
            hypothesis.call_hooks(hypothesis.hooks.pre_checkpoint, self, iteration=iteration)
            loss = self.step(loader)
            hypothesis.call_hooks(hypothesis.hooks.post_step, self, iteration=iteration, loss=loss)
        if self.supports_checkpointing():
            hypothesis.call_hooks(hypothesis.hooks.pre_checkpoint, self)
            self.checkpoint(self.model, epoch)
            hypothesis.call_hooks(hypothesis.hooks.post_checkpoint, self)
        del loader

    def step(self, loader):
        raise NotImplementedError

    def train(self, model):
        # Initialize the training.
        hypothesis.call_hooks(hypothesis.hooks.pre_reset, self)
        self.model = model
        self.reset()
        hypothesis.call_hooks(hypothesis.hooks.post_reset, self)
        # Seed the initial validation score.
        self._validate()
        # Start the traning procedure.
        for epoch in range(self.epochs):
            hypothesis.call_hooks(hypothesis.hooks.pre_epoch, self, epoch=epoch)
            self.epoch(epoch)
            hypothesis.call_hooks(hypothesis.hooks.post_epoch, self, epoch=epoch)
            self._validate()
        # Call the final hook.
        hypothesis.call_hooks(hypothesis.hooks.end, self)
