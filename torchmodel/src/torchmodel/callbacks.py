import os
import torch

class EarlyStopping:
    def __init__(self, patience=10, checkpointFolder = "checkpoints", checkPointFileName="model.pth"):
        self.patience = patience
        self.checkpointFolder = checkpointFolder
        
        self.bestEpoch = -1
        self.counter = 0
        self.bestLoss = float("inf")
        self.lastCheckpoint = None
        #if os.path.exists(self.checkpointFolder):
        #    shutil.rmtree(self.checkpointFolder)
        os.makedirs(self.checkpointFolder, exist_ok=True)
        self.checkPointFileName = checkPointFileName
        self.checkPointFilePath = os.path.join(self.checkpointFolder, self.checkPointFileName)
    def __call__(self, epoch, loss, model, optimizer, scheduler=None):
        if loss < self.bestLoss:
            self.lastCheckpoint = str.replace(self.checkPointFilePath, "[EPOCH]", str(epoch))
            print(f"Validation loss decreased from {self.bestLoss} to {loss}, saving checkpoint to {self.lastCheckpoint}")
            self.bestLoss = loss
            self.bestEpoch = epoch
            self.counter = 0
            state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": self.bestLoss
            }
            if scheduler is not None:
                state_dict["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(state_dict, self.lastCheckpoint)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            print(f"No improvement for {self.counter} epochs, early stopping!")
            return True
        return False
    def getLastCheckpoint(self):
        return self.lastCheckpoint