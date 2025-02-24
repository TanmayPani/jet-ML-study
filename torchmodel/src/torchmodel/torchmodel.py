import os
from typing import Any, Callable, Iterable, List, Tuple, Union, Optional, TypeVar

import torch
from tensordict import TensorDict

from . import callbacks

_Loss = TypeVar("_Loss", torch.nn.Module, Callable)

class TensorDictModel():
    def __init__(self, model : torch.nn.Module, device : torch.device = None):
        super().__init__()
        self.model = model
        self.to_device(device)
    
    def to_device(self, device : torch.device = None):
       if device is None:
           self.device = torch.device(
                           "cuda" if torch.cuda.is_available() 
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu"
                       )
       else:
            self.device = device
       self.model.to(self.device)
    
    def compile(self, compile_mode : str = None, 
                criterion : _Loss = None, optimizer : torch.optim.Optimizer = None, 
                scheduler : torch.optim.lr_scheduler.LRScheduler =None,
                compile_backend = None):
        
        if isinstance(criterion, torch.nn.Module):
            self.criterion = criterion.to(self.device)
        else:
            self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if compile_mode is not None:
            self.compile_mode = compile_mode
            self.comple_backend = compile_backend if compile_backend is not None else "inductor"
            if self.compile_mode in ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]:
                self.model.compile(mode = self.compile_mode, backend = self.comple_backend)
            else:
                raise ValueError(f"compile_mode must be one of 'default', 'reduce-overhead', 'max-autotune' or 'max-autotune-no-cudagraphs', but got {compile_mode}!")
    
    def __call__(self):
        return self.model
    
    def save(self, checkpoint):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, checkpoint)
        
        self.last_checkpoint = checkpoint   
    
    def get_last_checkpoint(self):
        return self.last_checkpoint
            
    def load(self, checkpoint, **torch_load_kwargs):
        checkpoint_dict = torch.load(checkpoint, **torch_load_kwargs)
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        self.model.to(self.device)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
        self.last_checkpoint = checkpoint   
        return checkpoint_dict
    
    def load_last_checkpoint(self, **torch_load_kwargs):
        return self.load(self.last_checkpoint, **torch_load_kwargs)
    
    def apply(self, batch : TensorDict, in_keys : Iterable, verbose : bool = False) -> torch.Tensor:
        inputs = []
        for key in in_keys:
            if verbose:
                print(f"Applying key: {key}")
                print(batch.get(key))
            inputs.append(batch.get(key))
        return self.model(*inputs)
    
    def predict(self, loader : torch.utils.data.DataLoader, in_keys : Iterable, out_activation : Callable = None) -> torch.Tensor:
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for ibatch, batch in enumerate(loader):
                print(f"Predicting... Batch: [{ibatch+1}/{len(loader)}]", end = "\r")
                inputs = batch.data.to(self.device)
                outputs.append(self.apply(inputs, in_keys))
        #print("Done.")
        
        if out_activation is None:
            return torch.cat(outputs)
        
        return out_activation(torch.cat(outputs))
    
    def evaluate(self, batch : TensorDict, in_keys : Iterable) -> torch.Tensor:
        outputs = self.apply(batch, in_keys)
        sample_weights = batch["sample_weights"]
        targets = batch["targets"]
        #return torch.dot(sample_weights.squeeze_(), self.criterion(outputs, targets).squeeze_()).div_(sample_weights.sum())
        return torch.dot(sample_weights.squeeze_(), self.criterion(outputs, targets).squeeze_()).div_(targets.shape[0])
        
    def train_epoch(self, loader : torch.utils.data.DataLoader, in_keys : Iterable):
        self.model.train()
        epoch_loss = 0
        nbatches = len(loader)
        for ibatch, batch in enumerate(loader):
            input = batch.data.to(self.device)
            loss = self.evaluate(input, in_keys)
            for param in self.model.parameters():
                param.grad = None
            loss.backward()
            self.optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            print(f"----Batch: [{ibatch+1}/{nbatches}], Loss: {batch_loss:.4f}", end = "\r")
        return epoch_loss / nbatches
    
    def validate_epoch(self, loader : torch.utils.data.DataLoader, in_keys : Iterable):
        self.model.eval()
        epoch_loss = 0
        nbatches = len(loader)
        with torch.no_grad():
            for ibatch, batch in enumerate(loader):
                input = batch.data.to(self.device)
                loss = self.evaluate(input, in_keys)
                batch_loss = loss.item()
                epoch_loss += batch_loss
                print(f"----Batch: [{ibatch+1}/{nbatches}], Loss: {batch_loss:.4f}", end = "\r")
        return epoch_loss / nbatches                
    
    def fit(self, train_loader : torch.utils.data.DataLoader, val_loader : torch.utils.data.DataLoader, epochs : int, 
            in_keys : Iterable, early_stopping : callbacks.EarlyStopping = None):
        assert len(in_keys) > 0
        history = {"train_loss": [], "val_loss": []}
        self.last_checkpoint = None
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, in_keys)
            val_loss = self.validate_epoch(val_loader, in_keys)
            print(f"Epoch: [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            if early_stopping is not None:
                if early_stopping(epoch, val_loss, self.model, self.optimizer, self.scheduler):
                    checkpoint_dict = self.load(early_stopping.getLastCheckpoint(), weights_only = True)
                    print(f"Best model from epoch {checkpoint_dict['epoch']} with validation loss {checkpoint_dict['val_loss']}")
                    break
        return history