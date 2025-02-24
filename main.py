from datetime import datetime
from functools import wraps
import os
import sys
import errno

import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from dotenv import load_dotenv

import torch
import pyarrow as pa
from pyarrow import compute as pc

from torchmodel import archs
from torchmodel import torchmodel
from torchmodel import datasets
from torchmodel import callbacks

from load_data import *

def train(trainDataset, valDataset, model : torchmodel.TensorDictModel, in_keys, model_weights_file : str = None):
    batchSize = 5000

    trainSampler = torch.utils.data.RandomSampler(trainDataset)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, sampler=trainSampler, pin_memory=True, collate_fn=datasets.batch_collate)

    valSampler = torch.utils.data.RandomSampler(valDataset) 
    valLoader = torch.utils.data.DataLoader(valDataset, batch_size=batchSize, sampler=valSampler, pin_memory=True, collate_fn=datasets.batch_collate)

    early_stopping = callbacks.EarlyStopping(patience=20, checkpointFolder = f"outputs/checkpoints/{datetime.now().strftime("%Y%m%d_%H%M")}", 
                                             checkPointFileName="model_[EPOCH].pth")
    
    if model_weights_file is not None:
       model.load(model_weights_file, weights_only=True)

    return model.fit(trainLoader, valLoader, epochs=5, in_keys=in_keys, early_stopping=early_stopping)

def test(testDataset, model : torchmodel.TensorDictModel, in_keys, model_weights_file : str = None):
    testSampler = torch.utils.data.SequentialSampler(testDataset)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=50000, sampler=testSampler, pin_memory=True, collate_fn=datasets.batch_collate)

    #model = torchmodel.TensorDictModel(archs.JetNN(phi_layers, f_layers, jet_dnn_layers, output_dnn_layers, do_batch_norm=True, dropout_prob=0.1), device=device)
    #model.compile(compile_mode="default")

    if model_weights_file is not None:
        model.load(model_weights_file, weights_only=True)

    return model.predict(testLoader, in_keys, torch.nn.Sigmoid()).flatten().cpu().numpy()

def main(device = "cuda"):
    #load_dotenv() 

    ppDataBuffer = pa.memory_map("/home/tanmaypani/wsl-stuff/workspace/macros/output/jetTable-pp200GeV_dijet.arrow", 'rb')
    AuAuDataBuffer = pa.memory_map("/home/tanmaypani/wsl-stuff/workspace/macros/output/jetTable-0010_AuAu200GeV_dijet_withoutRec.arrow", 'rb')

    jetColumns = ["pt", "eta", "phi", "e", "nef", "ncon_charged", "ncon_neutral"]
    constitColumns = ["con_pt", "con_eta", "con_phi", "con_charge"]

    table, targets, sample_weights  = load_jet_data(ppDataBuffer, AuAuDataBuffer, 
                                                    sample_weights="wt", tableKeys=["pp", "AuAu"], 
                                                    columns=jetColumns+constitColumns, pool=pa.default_memory_pool())
    trainDataset, valDataset, testDataset = make_jet_datasets(table, targets, sample_weights, seed=0, max_num_constits=15)

    phi_layers = [len(constitColumns), 64, 128, 256]
    f_layers = [512, 256, 128]
    jet_dnn_layers = [len(jetColumns), 128, 128, 64]
    output_dnn_layers = [256, 128, 64, 1]

    model = torchmodel.TensorDictModel(archs.JetNN(phi_layers, f_layers, jet_dnn_layers, output_dnn_layers, do_batch_norm=True, dropout_prob=0.1), device=device)

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model().parameters(), lr=0.01) 
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10) 

    model.compile(criterion=criterion, optimizer=optimizer, compile_mode="reduce-overhead", scheduler=lr_scheduler)

    in_keys = [("jets"), ("constituents"), ("constituents_mask")]

    history = train(trainDataset, valDataset, model, in_keys)

    scores = test(testDataset, model, in_keys)

    return history, scores, testDataset.labels

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device) 
    torch.set_float32_matmul_precision('high')
    if("cuda" in str(device)):
        torch.backends.cuda.matmul.allow_tf32 = True    
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    history, scores, y = main(device)

    if device == "cuda":
        torch.cuda.empty_cache() 

    ypred = np.where(scores > 0.5, 1, 0)
    cMatrix = metrics.confusion_matrix(y, ypred, normalize="true")
    auc_score = metrics.roc_auc_score(y, scores)
    accuracy = metrics.accuracy_score(y, ypred)

    print(f"Confusion matrix: {cMatrix}")
    print(f"AUC: {auc_score}, Accuracy: {accuracy}")