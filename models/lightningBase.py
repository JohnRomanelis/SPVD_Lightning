import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from abc import ABC, abstractmethod

class Task(ABC):
    @abstractmethod
    def prep_data(self, batch):
        pass
    @abstractmethod
    def loss_fn(self, pred, target):
        pass

class SparseGeneration(Task):
    def prep_data(self, batch):
        noisy_data, t, noise = batch['input'], batch['t'], batch['noise']
        inp = (noisy_data, t)
        return inp, noise.F
    def loss_fn(self, preds, target):
        return F.mse_loss(preds, target)

class DiffusionBase(L.LightningModule):

    def __init__(self, model, task=SparseGeneration(), lr=0.0002):
        super().__init__()
        self.model = model
        self.task = task
        self.learning_rate = lr
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # get data from the batch
        inp, target = self.task.prep_data(batch)

        # activate the network for noise prediction
        preds = self(inp)

        # calculate the loss
        loss = self.task.loss_fn(preds, target)

        self.log('train_loss', loss, batch_size=self.tr_batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, target = self.task.prep_data(batch)
        preds = self(inp)
        loss = self.task.loss_fn(preds, target)
        self.log('val_loss', loss, batch_size=self.vl_batch_size)

    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.05)

        # Create a dummy scheduler (we will update `total_steps` later)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=1)

        # Return optimizer and scheduler (scheduler will be updated in `on_fit_start`)
        return [optimizer], [{'scheduler': self.lr_scheduler, 'interval': 'step'}]

    # Setting the OneCycle scheduler correct number of steps at the start of the fit loop, where the dataloaders are available.
    def on_train_start(self):
        # Access the dataloader and calculate total steps
        train_loader = self.trainer.train_dataloader  # Access the dataloader from the trainer
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.trainer.max_epochs
        
        # Update the scheduler's `total_steps` dynamically
        self.lr_scheduler.total_steps = total_steps

        # Read the batch size for logging
        self.tr_batch_size = self.trainer.train_dataloader.batch_size

    def on_validation_start(self):
        val_loader = self.trainer.val_dataloaders
        if val_loader:
            self.vl_batch_size = val_loader.batch_size

