from networkx import dfs_edges
from rdkit import Chem

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import SimpleConv
from torch_geometric.data import Batch


class GCN(torch.nn.Module):
    def __init__(self, bias = False):
        super().__init__()
        neuron_num = [48,128,128,128,128,128,128]
        self.Linear_for_dim1 = pyg_nn.Linear(in_channels = 48, out_channels = 128, bias = bias)
        self.sc = SimpleConv(aggr='sum', combine_root='sum')
        self.Linear1 = pyg_nn.Linear(neuron_num[0], neuron_num[1], bias = bias)
        self.Linear2 = pyg_nn.Linear(neuron_num[1], neuron_num[2], bias = bias)
        self.Linear3 = pyg_nn.Linear(neuron_num[2], neuron_num[3], bias = bias)
        self.Linear4 = pyg_nn.Linear(neuron_num[3], neuron_num[4], bias = bias)
        self.Linear5 = pyg_nn.Linear(neuron_num[4], neuron_num[5], bias = bias)
        self.Linear6 = pyg_nn.Linear(neuron_num[5], neuron_num[6], bias = bias)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        
    def forward(self, data):
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr[:,0]
        
        x1 = self.sc(x, edge_index)#, edge_weight)
        x1 = self.Linear1(x1)
        x1 = F.relu(x1)
        x1 = x1 + self.Linear_for_dim1(x)
        
        x2 = self.sc(x1, edge_index)#, edge_weight)
        x2 = self.Linear2(x2) 
        x2 = F.relu(x2)
        x2 = x2 + x1 
        
        x3 = self.sc(x2, edge_index)#, edge_weight)
        x3 = self.Linear3(x3)
        x3 = F.relu(x3)
        x3 = x3 + x2
        
        x4 = self.sc(x3, edge_index)#, edge_weight)
        x4 = self.Linear4(x4)
        x4 = F.relu(x4)
        x4 = x4 + x3
        
        x5 = self.sc(x4, edge_index)#, edge_weight)
        x5 = self.Linear5(x5)
        x5 = F.relu(x5)
        x5 = x5 + x4
        
        x6 = self.sc(x5, edge_index)#, edge_weight)
        x6 = self.Linear6(x6)
        x6 = F.relu(x6)
        x6 = x6 + x5

        return x6



class GCN_eluent_interaction_5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN_compound = GCN()
        self.GCN_eluent = GCN()


        self.interaction_linear_list = nn.ModuleList([pyg_nn.Linear(in_channels = 256, out_channels = 1024),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(1024),
                                        pyg_nn.Linear(1024, 512),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(512)])

        
        self.pred_list = nn.ModuleList([pyg_nn.Linear(in_channels =512 , out_channels = 256),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(256),
                                        pyg_nn.Linear(in_channels =256 , out_channels = 128),
                                        nn.LeakyReLU(0.1),
                                        nn.BatchNorm1d(128),
                                        #nn.Dropout(0.3),
                                        pyg_nn.Linear(128, 1)])


        self.interaction_linear_list.apply(self._init_weights)
        self.pred_list.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m ,(nn.Linear, pyg_nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, compound_data, elu1_data, elu2_data):
        compound_GC_data = self.GCN_compound(compound_data)
        elu1_GC_data = self.GCN_eluent(elu1_data)
        elu2_GC_data = self.GCN_eluent(elu2_data)

        compund_pooled = pyg_nn.global_add_pool(compound_GC_data, compound_data.batch)
        
        elu1_pooled = pyg_nn.global_add_pool(elu1_GC_data, elu1_data.batch)
        elu2_pooled = pyg_nn.global_add_pool(elu2_GC_data, elu2_data.batch)
        mixed_eluent = elu1_pooled*(compound_data.ratio[:,0].view(-1,1)) + elu2_pooled*(compound_data.ratio[:,1].view(-1,1))

        mixed_eluent_compond_data = torch.cat([compund_pooled, mixed_eluent],dim = -1)

        interaction_GC_data = mixed_eluent_compond_data
        for layer in self.interaction_linear_list:
            interaction_GC_data = layer(interaction_GC_data)

        retention = interaction_GC_data
        for layer in self.pred_list:
            retention = layer(retention)
            
        return retention
