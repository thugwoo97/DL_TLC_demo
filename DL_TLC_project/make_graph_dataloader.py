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
from tqdm.auto import tqdm

from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors

import os


##########################################################################################################################################
############################################################    make graph    ############################################################
##########################################################################################################################################




################################################################### 6개 feature들에 대한 원-핫 인코딩 ############################################################### 

def atom_symbol_HNums(atom):
    
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O','S', 'H', 'F', 'Cl', 'Br', 'I','Se','Te','Si','P','B','Sn','Ge'])+
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))


def atom_degree(atom):
    return np.array(one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5 ,6])).astype(int) 


def atom_Aroma(atom):
    return np.array([atom.GetIsAromatic()]).astype(int)


def atom_Hybrid(atom):
    return np.array(one_of_k_encoding_unk(str(atom.GetHybridization()),['S','SP','SP2','SP3','SP3D','SP3D2'])).astype(int)


def atom_ring(atom):
    return np.array([atom.IsInRing()]).astype(int)


def atom_FC(atom):
    return np.array(one_of_k_encoding_unk(atom.GetFormalCharge(), [-4,-3,-2,-1, 0, 1, 2, 3, 4])).astype(int)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


################################################################### make feature matrix ############################################################### 

####### feature matrix with R/S configuration

def make_x(mol):

    #mol = Chem.MolFromSmiles(ChromSol)
    
    feature_matrix=np.vstack([np.hstack([atom_symbol_HNums(atom),
                    atom_degree(atom),
                    atom_Aroma(atom),
                    atom_Hybrid(atom),
                    atom_ring(atom),
                    atom_FC(atom)]) for atom in mol.GetAtoms()])

    chirality = Chem.FindMolChiralCenters(mol, includeUnassigned = True)

    all_atom_chirality = np.zeros((feature_matrix.shape[0], 3))
    for index, form in chirality:
        if form == 'R':
            all_atom_chirality[index,0] = 1
        elif form =='S':
            all_atom_chirality[index,1] = 1
        elif form =='?':
            all_atom_chirality[index,2] = 1
            

    feature_matrix = np.hstack([feature_matrix, all_atom_chirality])
    
    return feature_matrix


################################################################### make_edge_index ############################################################### 
def make_edge_index(mol):
    #mol = Chem.MolFromSmiles(ChromSol)
    starting_atoms = np.hstack([(bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()) for bond in mol.GetBonds()])
    terminal_atoms = np.hstack([(bond.GetEndAtom().GetIdx(), bond.GetBeginAtom().GetIdx()) for bond in mol.GetBonds()])
    edge_index = np.vstack([starting_atoms, terminal_atoms])
    return edge_index


################################################################### make_edge_attr ############################################################### 
# GCN 에서 사용은 안 하는 중

def make_edge_attr(mol):
    
    #mol = Chem.MolFromSmiles(ChromSol)
    
    bond_weight = {Chem.rdchem.BondType.SINGLE : 1,
                  Chem.rdchem.BondType.AROMATIC : 1.5,
                  Chem.rdchem.BondType.DOUBLE : 2,
                  Chem.rdchem.BondType.TRIPLE : 3}
    
    edge_attr = np.hstack([(bond_weight[bond.GetBondType()], bond_weight[bond.GetBondType()]) for bond in mol.GetBonds()]).reshape(-1,1)
    return edge_attr


################################################################### MolToGraph ############################################################### 


class MolToGraph(InMemoryDataset):
    def __init__(self, smiles, root, raw_name, preocessed_name, molecular_param, augment=False, transform=None, pre_transform=None, pre_filter=None):

        # smiles : 'ChromSol' or 'Chromophore_smiles' or 'Solvent_smiles' 
        
        self.smiles = smiles
        self.raw_name = raw_name
        self.preocessed_name = preocessed_name
        self.molecular_param = bool(molecular_param)
        self.augment = bool(augment)
        self.mean = np.load('./standardization/mean.npy')
        self.std = np.load('./standardization/std.npy')
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        

    @property
    def raw_file_names(self):
        return [self.raw_name]

    @property
    def processed_file_names(self):
        return [self.preocessed_name]

    def make_label(self, data, i):
        if self.augment:
            label = torch.tensor(data['Rf'].iloc[i]).view(1,-1)
        else:
            label = torch.tensor(((data['Rf'].iloc[i])-self.mean[0])/self.std[0], dtype = torch.float).view(1,-1)
        return label

    
    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv(self.root +'/'+ self.raw_name)

        data_list = []
        for i, ChromSol in tqdm(enumerate(df[self.smiles]), total=len(df)):

            # None을 써서 제작했더라도, csv로 불러오면 Nan으로 변경되므로 if ChromSol is None 사용하면 안됨
            
            if pd.isna(ChromSol):
                #x = torch.zeros((1,45))
                x = torch.zeros((1,48)) # R form S form 나타내는거 추가
                y = self.make_label(df,i)
                edge_index = torch.empty((2,0), dtype = torch.long)
                edge_attr = torch.empty((0,1), dtype = torch.float)
                ratio = torch.tensor(np.hstack([df['ratio1'].iloc[i],df['ratio2'].iloc[i]]), dtype = torch.float).view(1,-1)
                sample_id = torch.tensor(df['sample_id'].iloc[i], dtype = torch.long).view(1,-1)
                # compound_parameter / elu_parameter
                if self.molecular_param == True:
                    compound_param = torch.empty((1,4), dtype = torch.float)                
                    eluent1_param = torch.empty((1,4), dtype = torch.float)
                    eluent2_param = torch.empty((1,4), dtype = torch.float)
            
            else:
                mol = Chem.MolFromSmiles(ChromSol)
                x = torch.tensor(make_x(mol), dtype = torch.float)
                edge_index = torch.tensor(make_edge_index(mol), dtype = torch.long)
                edge_attr = torch.tensor(make_edge_attr(mol), dtype = torch.float)
                y = self.make_label(df,i)
                ratio = torch.tensor(np.hstack([df['ratio1'].iloc[i],df['ratio2'].iloc[i]]), dtype = torch.float).view(1,-1)
                sample_id = torch.tensor(df['sample_id'].iloc[i], dtype = torch.long).view(1,-1)
                # compound_parameter / elu_parameter
                if self.molecular_param == True:
                    compound_param = torch.tensor(np.hstack(df.loc[i,['HBA','HBD','TPSA', 'ASA']]), dtype = torch.float).view(1,-1)  
                    eluent1_param = torch.tensor(np.hstack(df.loc[i,['HBA_1','HBD_1','kamlet_1', 'rot_1']]), dtype = torch.float).view(1,-1)
                    eluent2_param = torch.tensor(np.hstack(df.loc[i,['HBA_2','HBD_2','kamlet_2', 'rot_2']]), dtype = torch.float).view(1,-1)
            

            if self.molecular_param == True:
                data = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y = y, ratio=ratio, sample_id = sample_id, compound_param = compound_param, eluent1_param = eluent1_param, eluent2_param = eluent2_param)
            else:
                data = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y = y, ratio=ratio, sample_id = sample_id)
 
 
            
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])



##########################################################################################################################################
############################################################    dataloader    ############################################################
##########################################################################################################################################


# 위에서 저장해둔 compound, elu1, elu2 그래프 데이터 불러와서 따로따로 dataloader 하지말고, 하나의 dataloader 에 다같이 불러오기
# __getitem__ 맨날 헷갈려서 아래 설명 적어둠

from torch.utils.data import Dataset

class TripleGraphDataset(Dataset):
    def __init__(self, compound_ds, elu1_ds, elu2_ds):
        assert len(compound_ds) == len(elu1_ds) == len(elu2_ds)
        self.compound_ds = compound_ds
        self.elu1_ds = elu1_ds
        self.elu2_ds = elu2_ds

    def __len__(self):
        return len(self.compound_ds)

    def __getitem__(self, idx):
        com  = self.compound_ds[idx]
        elu1 = self.elu1_ds[idx]
        elu2 = self.elu2_ds[idx]
        # 이렇게 하면 a = TripleGraphDataset(blah_blah_list) 이렇게 클래스 인스턴스 생성해서
        # a[3] 이렇게 썼을때, blah_blah_list[3] 가 호출되는 거임

        assert com.sample_id.item() == elu1.sample_id.item() == elu2.sample_id.item() , 'compound, elu1, elu2 짝지어지지 않음!!'

        return com, elu1, elu2


