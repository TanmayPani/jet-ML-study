from typing import Any, Callable, Iterable, List, Tuple, Union, Optional, TypeVar, Dict
import pyarrow as pa
import torch
import tensordict as td
from tensordict import TensorDict

_T = TypeVar("_T", torch.Tensor, Iterable[torch.Tensor])
_D = TypeVar("_D", Dict[str, _T], Iterable[Dict[str, _T]])

class Batch:
    def __init__(self, *batch):
        #print(len(batch))
        if len(batch) == 0:
            raise ValueError("Batch must not be empty!")
        elif len(batch) == 1:
            self.data = batch[0]
        else:
            self.data = torch.stack(batch)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
            
    def pin_memory(self):
        self.data = self.data.pin_memory()
        #for key, value in self.data.items():
        #    if isinstance(value, torch.Tensor):
        #        self.data[key] = value.pin_memory()
        #    elif isinstance(value, Iterable[torch.Tensor]):
        #        self.data[key] = [tensor.pin_memory() for tensor in value]
        return self

def batch_collate(batch : TensorDict  | Iterable) -> Batch:
    if isinstance(batch, TensorDict):
        return Batch(batch)
    else:
        return Batch(*batch)
    
class CustomCategoricalDataset(torch.utils.data.Dataset):
    def __init__(self, labels :Iterable, sample_weights : Iterable = None, **kwargs):
        super().__init__()
        self.labels = labels
        self.length = len(labels)
        self.do_one_hot = kwargs.pop("do_one_hot", False)
        if self.do_one_hot:
            self.num_classes = kwargs.pop("nclasses", 2)
        if sample_weights is not None:
            assert sample_weights.shape[0] == self.length
            self.sample_weights = sample_weights
        else:
            self.sample_weights = torch.ones(self.length, dtype=torch.float32)
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.__getitems__([idx])
    
    def __getitems__(self, idxs):
        _dict = {}
        if self.do_one_hot:
            _dict["targets"] = torch.as_tensor(torch.nn.functional.one_hot(torch.as_tensor(self.labels[idxs], dtype=torch.long), num_classes=self.num_classes), dtype=torch.float32)
        else:
            _dict["targets"] = torch.as_tensor(self.labels[idxs], dtype=torch.float32).unsqueeze_(1)
            _dict["sample_weights"] = torch.as_tensor(self.sample_weights[idxs], dtype=torch.float32).unsqueeze_(1)
            
        return TensorDict(_dict, batch_size = [len(idxs)])

_ArrowData = Union[pa.Table, List[pa.Table], Tuple[pa.Table, ...]]    
             
class ArrowTableDataset(CustomCategoricalDataset):
    def __init__(self,  data : _ArrowData, labels : Iterable, sample_weights : Iterable = None, keys : Iterable[str] = None, **kwargs):
        super().__init__(labels, sample_weights=sample_weights, **kwargs)
        
        if isinstance(data, pa.Table):
            self.data = [data]
        else:
            self.data = data
            for table in self.data:
                assert len(table) == self.length
                
        self.ntables = len(self.data) 
        
        self.ncols = [len(table.column_names) for table in self.data]
          
        self.keys = keys if keys is not None else [f"d{i}" for i in range(self.ntables)]    
        assert len(self.keys) == self.ntables
           
        self.ragged_dims = kwargs.pop("ragged_dims", [None]*self.ntables)
        assert len(self.ragged_dims) == self.ntables
        
        self.is_ragged = [False if dim is None else True for dim in self.ragged_dims]
        
        self.pad_value = kwargs.pop("pad_value", [-1]*self.ntables)
        assert len(self.pad_value) == self.ntables
        
        self.pad_to_len = kwargs.pop("pad_to_len", [50]*self.ntables)
        assert len(self.pad_to_len) == self.ntables
        
        self.do_scale = kwargs.pop("do_scale", [False]*self.ntables)
        assert len(self.do_scale) == self.ntables
        
        self.mean_tensors   = kwargs.pop("mean", [])
        
        self.stddev_tensors = kwargs.pop("stddev", [])
        
        _scale_from = kwargs.pop("scale_from", None)
        
        if self.do_scale:
            if len(self.mean_tensors) == 0 or len(self.stddev_tensors) == 0:
                if _scale_from is not None:
                    assert isinstance(_scale_from, ArrowTableDataset)
                    assert _scale_from.ntables == self.ntables
                    assert _scale_from.do_scale
                    
                    self.scale_from(_scale_from)
                for data in self.data:
                    _mean_tensor, _stddev_tensor = self._calculate_mean_stddev(data)
                    self.mean_tensors.append(_mean_tensor)
                    self.stddev_tensors.append(_stddev_tensor)
            else:
                assert len(self.mean_tensors) == self.ntables
                assert len(self.stddev_tensors) == self.ntables
                
                    
    
    def scale_from(self, other):
        assert self.ntables == other.ntables
        self.mean_tensors = other.mean_tensors.copy()
        self.stddev_tensors = other.stddev_tensors.copy()
               
    def _calculate_mean_stddev(self, data : pa.Table):
        mean_list = []
        stddev_list = []
        for col in data.column_names:
            if pa.types.is_list(data[col].type):
                mean_list.append(pa.compute.mean(pa.compute.list_flatten(data[col])).as_py())
                stddev_list.append(pa.compute.stddev(pa.compute.list_flatten(data[col])).as_py())
            else:
                mean_list.append(pa.compute.mean(data[col]).as_py())
                stddev_list.append(pa.compute.stddev(data[col]).as_py())
        return torch.as_tensor(mean_list, dtype=torch.float32), torch.as_tensor(stddev_list, dtype=torch.float32)
    
    def _pad_to_constant_length(self, tensor : torch.Tensor, max_length : int, dim : int = 0, value : Optional[float] = -1) -> torch.Tensor:
        _ndims = tensor.dim()
        if dim >= _ndims or dim < -_ndims:
            raise ValueError(f"dim must be < {_ndims} or >= -{_ndims}, but got {dim}!")
        elif dim < 0 and dim >= -_ndims:
            dim = _ndims + dim

        _pad = [0]*2*_ndims
        _pad[2*(_ndims - dim) - 1] = max_length - tensor.shape[dim] 

        _shape_good = list(tensor.shape)
        _mask_good = torch.full(_shape_good, True, dtype=torch.bool)
        if tensor.shape[dim] == max_length:
            return tensor, _mask_good
        elif tensor.shape[dim] < max_length:
            _shape_bad = _shape_good
            _shape_bad[dim] = max_length - _shape_bad[dim]
            _mask_bad = torch.full(_shape_bad, False, dtype=torch.bool)
            return torch.nn.functional.pad(tensor, tuple(_pad), value=value), torch.cat([_mask_good, _mask_bad], dim=dim)
        else:
            return torch.index_select(tensor, dim, torch.arange(max_length, dtype=torch.long)), torch.index_select(_mask_good, dim, torch.arange(max_length, dtype=torch.long))
        
            
    def _get_table_rows(self, tableIdx : int, idxs : list[int], inDict : TensorDict) -> torch.Tensor:
        
        _rowList = list(zip(*(self.data[tableIdx].take(idxs).to_pydict().values())))
        if not self.is_ragged[tableIdx]:
            _rows = torch.as_tensor(_rowList, dtype=torch.float32)
            if self.do_scale: 
                if self.mean_tensors[tableIdx].dim() < _rows[0].dim():
                    self.mean_tensors[tableIdx].unsqueeze_(1)
                    self.stddev_tensors[tableIdx].unsqueeze_(1)
                _rows.sub_(self.mean_tensors[tableIdx]).div_(self.stddev_tensors[tableIdx])
            inDict[self.keys[tableIdx]] = _rows
        
        else:
            _raggedRowList = []
            _raggedMaskList = []
            for irow, _row in enumerate(_rowList):
                _raggedRowTensor = torch.as_tensor(_row, dtype=torch.float32)
                if self.do_scale:
                    if irow == 0 and self.mean_tensors[tableIdx].dim() < _raggedRowTensor.dim():
                        self.mean_tensors[tableIdx].unsqueeze_(1)
                        self.stddev_tensors[tableIdx].unsqueeze_(1)
                    _raggedRowTensor.sub_(self.mean_tensors[tableIdx]).div_(self.stddev_tensors[tableIdx])
                
                _rowTensor, _maskTensor = self._pad_to_constant_length(_raggedRowTensor, self.pad_to_len[tableIdx], dim = self.ragged_dims[tableIdx], value = self.pad_value[tableIdx])
                _raggedRowList.append(_rowTensor)
                _raggedMaskList.append(_maskTensor)
            inDict[self.keys[tableIdx]] = torch.stack(_raggedRowList)
            inDict[f"{self.keys[tableIdx]}_mask"] = torch.stack(_raggedMaskList)
        
    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.__getitems__(idx)
        return self.__getitems__([idx]) 
    
    def __getitems__(self, idxs):
        _dict = super().__getitems__(idxs)
        for i, key in enumerate(self.keys):
           self._get_table_rows(i, idxs, _dict)
        return _dict