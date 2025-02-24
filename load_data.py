from typing import Iterable, TypeVar, Optional, Union
from functools import singledispatch

import numpy as np 
import pyarrow as pa
from pyarrow import compute as pc
from torchmodel import datasets
from sklearn.model_selection import train_test_split

@singledispatch
def process_table_data(data, table : pa.Table) -> np.ndarray :
    raise NotImplementedError(f"No way to process associated table data of type {type(data)}")

@process_table_data.register
def _(data : Union[int, float, complex], table : pa.Table) -> np.ndarray :
    return np.full(len(table), data)

@process_table_data.register
def _(data : Union[list, tuple, np.ndarray], table : pa.Table) -> np.ndarray :
    if len(data) == len(table):
        return np.asarray(data)
    else:
        return np.full(len(table), np.asarray(data))

@process_table_data.register
def _(data : Union[pa.Array, pa.ChunkedArray], table : pa.Table) -> np.ndarray :
    return process_table_data(data.to_numpy(), table)

@process_table_data.register
def _(data : str, table : pa.Table) -> np.ndarray :
    if data in table.column_names:
        res_ = process_table_data(table[data], table)
        table = table.drop_columns(data)
        return res_
    else:
        raise KeyError(f"Column {data} not found in table")

def load_jet_data(*buffers, labels=None, sample_weights=None, tableKeys=None, columns=None, pool=None):
    #tracemalloc.start()
    nSamples = len(list(buffers))
    
    tables = []
    tableLengths = []
    sampleWeights = []
    targets = []

    if tableKeys is None:
        tableKeys = [f"table_{i}" for i in range(nSamples)]
    assert len(tableKeys) == nSamples
    
    if labels is not None and isinstance(labels, Union[int, float, complex, str]):
        labels = nSamples*[labels]
    elif labels is None:
        labels = list(range(0, nSamples))
    
    if sample_weights is not None and isinstance(sample_weights, Union[int, float, complex, str]):
        sample_weights = nSamples*[sample_weights]
    elif sample_weights is None:
        sample_weights = nSamples*[1.0]
    
    for ibuffer, buffer in enumerate(list(buffers)):
        tables.append(pa.ipc.open_file(buffer, memory_pool=pool).read_all())
        tableLengths.append(len(tables[-1]))
        sampleWeights.append(process_table_data(sample_weights[ibuffer], tables[-1]))
        targets.append(process_table_data(labels[ibuffer], tables[-1]))
        
        if isinstance(columns, Iterable):
            tables[-1] = tables[-1].select(columns)
        
        print("Number of ", tableKeys[ibuffer], " samples: ", tableLengths[-1])
    
    totalSamples = np.sum(tableLengths)
    sampleWeights = [sampleWeight/np.sum(sampleWeight)*(totalSamples/nSamples) for sampleWeight in sampleWeights]
    for ibuffer, table in enumerate(tables):
        print("Sum of ", tableKeys[ibuffer], " sample weights: ", np.sum(sampleWeights[ibuffer]))
        
    dataTable = pa.concat_tables(tables)
    dataTargets = np.concatenate(targets)
    dataSampleWeights = np.concatenate(sampleWeights)
    
    print("Total number of samples: ", len(dataTable))
    
    return dataTable, dataTargets, dataSampleWeights

def make_jet_datasets(dataTable, labels, sample_weights, test_size=0.2, val_size=0.2, seed=None, max_num_constits=None):
    randomState = None
    if seed is not None:
       if isinstance(seed, int):
           randomState = np.random.RandomState(seed)
       elif isinstance(seed, np.random.RandomState):
           randomState = seed
       else:
           raise ValueError("Invalid seed type")
    
    jetColumns = []
    constitColumns = []
    
    for col in dataTable.column_names:
        if pa.types.is_list(dataTable[col].type):
            constitColumns = constitColumns + [col] 
        else:
            jetColumns =  jetColumns + [col]
    
    print("jetColumns: ", jetColumns)
    print("constitColumns: ", constitColumns)
    
    if len(constitColumns) > 0:
        if max_num_constits is None:
            max_num_constits = 10000000
        
        nConstitArray = pc.list_value_length(dataTable[constitColumns[0]])
        maxNConstitActual = pc.max(nConstitArray).as_py()
        if maxNConstitActual < max_num_constits:
            print("Maximum number of jet constituents: ", maxNConstitActual)
            max_num_constits = maxNConstitActual
        elif maxNConstitActual > max_num_constits:
            print("Maximum number of jet constituents truncated to ", max_num_constits, " from ", maxNConstitActual)
        else:
            print("Maximum number of jet constituents: ", max_num_constits)
            
    print("Number of jet columns: ", len(jetColumns))
    print("Number of constituent columns: ", len(constitColumns))

    jetTable = dataTable.select(jetColumns)
    constitTable = dataTable.select(constitColumns) if len(constitColumns) > 0 else None   

    if constitTable is not None:   
        tables = [jetTable, constitTable]
        tableKeys = ["jets", "constituents"]
    else:
        tables = [jetTable]
        tableKeys = ["jets"]

    dsKwargs = {}
    if constitTable is not None:
        dsKwargs["ragged_dims"] = [None, -1]
        dsKwargs["pad_values"]  = [None, -1.]
        dsKwargs["pad_to_len"]  = [None, max_num_constits]
        dsKwargs["do_scale"]    = [True, True]
    else:
        dsKwargs["ragged_dims"] = [None]
        dsKwargs["pad_values"]  = [None]
        dsKwargs["pad_to_len"]  = [None]
        dsKwargs["do_scale"]    = [True]
        
    indices = np.arange(len(dataTable))
    trainIndices, testIndices  = train_test_split(indices, test_size=0.2, stratify=labels, random_state=randomState)
    if val_size is not None:
        trainIndices, valIndices = train_test_split(trainIndices, test_size=0.2, stratify=labels[trainIndices], random_state=randomState)

    trainDataset = datasets.ArrowTableDataset([table.take(trainIndices) for table in tables], labels[trainIndices], sample_weights[trainIndices],
                                              keys=tableKeys, **dsKwargs)
    testDataset = datasets.ArrowTableDataset([table.take(testIndices) for table in tables], labels[testIndices], sample_weights[testIndices],
                                         keys=tableKeys, scale_from=trainDataset, **dsKwargs)
    
    if val_size is None:
        return trainDataset, testDataset
    else:
        valDataset = datasets.ArrowTableDataset([table.take(valIndices) for table in tables], labels[valIndices], sample_weights[valIndices],
                                            keys=tableKeys, scale_from=trainDataset, **dsKwargs)
        return trainDataset, valDataset, testDataset

#if __name__ == "__main__":
#    pool = pa.default_memory_pool()
#    ppDataBuffer = pa.memory_map("/home/tanmaypani/wsl-stuff/workspace/macros/output/jetTable-pp200GeV_dijet.arrow", 'rb')
#    AuAuDataBuffer = pa.memory_map("/home/tanmaypani/wsl-stuff/workspace/macros/output/jetTable-0010_AuAu200GeV_dijet_withoutRec.arrow", 'rb')
#    
#    columns=["pt", "eta", "phi", "e", "nef", "ncon_charged", "ncon_neutral", "con_pt", "con_eta", "con_phi", "con_charge"]
#    table, targets, sample_weights  = load_jet_data(ppDataBuffer, AuAuDataBuffer, sample_weights="wt", tableKeys=["pp", "AuAu"], columns=columns, pool=pool)
#    trainDataset, valDataset, testDataset = make_jet_datasets(table, targets, sample_weights, test_size=0.2, val_size=0.2, seed=0)
#    print(len(trainDataset), len(valDataset), len(testDataset))
#    print(trainDataset.data[0].take([0]))
#    print(trainDataset.data[1].take([0]))
#    for mean, std in zip(trainDataset.mean_tensors, trainDataset.stddev_tensors):
#        print("Mean: ", mean)
#        print("Stddev: ", std)
#    print(trainDataset[0]["jets"])
#    print(trainDataset[0]["constituents"])
#    print(trainDataset[0]["sample_weights"])
#    print(trainDataset[0]["targets"])
    
    