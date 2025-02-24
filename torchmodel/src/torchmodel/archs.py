import torch

class DNN(torch.nn.Module):
    def __init__(self, sizes, output_activation=None, dropout_prob=None, do_batch_norm=False, **kwargs):
        super().__init__()
        
        assert len(sizes) > 1
        self.layers = torch.nn.ModuleList()
        self.layer_sizes = sizes[:-1]
        self.output_size = sizes[-1]
        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if dropout_prob is not None:
                self.layers.append(torch.nn.Dropout(dropout_prob))
            self.layers.append(torch.nn.Linear(input_size, output_size))
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(output_size))
            self.layers.append(torch.nn.ReLU())
            
        self.layers.append(torch.nn.Linear(self.layer_sizes[-1], self.output_size))
        if output_activation is not None:
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(self.output_size))
            self.layers.append(output_activation)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Conv1dNN(torch.nn.Module):
    def __init__(self, sizes, output_activation=None, dropout_prob=None, do_batch_norm=False, **kwargs):
        super().__init__()
        
        assert len(sizes) > 1
        self.layers = torch.nn.ModuleList()
        self.layer_sizes = sizes[:-1]
        self.output_size = sizes[-1]
        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            if dropout_prob is not None:
                self.layers.append(torch.nn.Dropout(dropout_prob))
            self.layers.append(torch.nn.Conv1d(input_size, output_size, kernel_size=1))
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(output_size))
            self.layers.append(torch.nn.ReLU())
            
        self.layers.append(torch.nn.Conv1d(self.layer_sizes[-1], self.output_size, kernel_size=1))
        if output_activation is not None:
            if do_batch_norm:
                self.layers.append(torch.nn.BatchNorm1d(self.output_size))
            self.layers.append(output_activation)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class SumPooling(torch.nn.Module):
    def __init__(self, dim, mask_dim=1):
        super().__init__()
        self.dim = dim
        self.mask_dim = mask_dim
    
    def _resize_mask(self, mask, size, dim):
        return mask.select(dim, 0).unsqueeze_(dim).expand(size)
        
    def forward(self, x, mask=None):
        if mask is not None:
            _new_mask = self._resize_mask(mask, x.size(), self.mask_dim)
            x = torch.where(_new_mask, x, torch.zeros_like(x))
        return torch.sum(x, dim=self.dim)
    
class PFN(torch.nn.Module):
    def __init__(self, phi_sizes, f_sizes, output_activation=None,
                 phi_dropout_prob=None, phi_do_batch_norm=False, 
                f_dropout_prob=None, f_do_batch_norm=False):
        super().__init__()
        
        self.phi_sizes = phi_sizes
        self.f_sizes = [phi_sizes[-1]] + f_sizes
        
        assert len(self.phi_sizes) > 1 and len(self.f_sizes) > 1
        
        self.phi_layers = torch.nn.ModuleList()
        self.phi_layers.append(Conv1dNN(phi_sizes, output_activation=torch.nn.ReLU(), dropout_prob=phi_dropout_prob, do_batch_norm=phi_do_batch_norm))
        
        self.pooling_layer = SumPooling(-1)
        
        self.f_layers = torch.nn.ModuleList()
        self.f_layers.append(DNN(self.f_sizes, output_activation=output_activation, dropout_prob=f_dropout_prob, do_batch_norm=f_do_batch_norm))
                
    def forward(self, x, mask):
        for layer in self.phi_layers:
            x = layer(x)
        x = self.pooling_layer(x, mask)
        for layer in self.f_layers:
            x = layer(x)
        return x
        
class JetNN(torch.nn.Module):
    def __init__(self, phi_sizes, f_sizes, jet_dnn_sizes, output_dnn_sizes, dropout_prob=None, do_batch_norm=False):
        super().__init__()
        self.pfn = PFN(phi_sizes, f_sizes, 
                       phi_dropout_prob=dropout_prob, phi_do_batch_norm=do_batch_norm, 
                       f_dropout_prob=dropout_prob, f_do_batch_norm=do_batch_norm,
                       output_activation=torch.nn.ReLU()
                    )  
        self.jet_dnn = DNN(jet_dnn_sizes, 
                           dropout_prob=dropout_prob, do_batch_norm=do_batch_norm,
                           output_activation=torch.nn.ReLU()
                        )
        self.output_dnn_sizes = [jet_dnn_sizes[-1] + f_sizes[-1]] + output_dnn_sizes
        self.output_dnn = DNN(self.output_dnn_sizes, 
                              dropout_prob=dropout_prob, do_batch_norm=do_batch_norm
                            )        
        
    def forward(self, jet, constits, mask):
        jet_dnn_output = self.jet_dnn(jet)
        pfn_output = self.pfn(constits, mask)
        output = torch.cat([jet_dnn_output, pfn_output], dim=-1)
        output = self.output_dnn(output)
        return output
        
        