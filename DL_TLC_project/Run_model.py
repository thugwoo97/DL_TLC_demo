from rdkit import Chem

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Subset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import SimpleConv
from torch_geometric.data import Batch
from tqdm.auto import tqdm
import os


def r2_score(preds, targets):
    nan_mask = ~torch.isnan(targets)
    preds = preds[nan_mask]
    targets = targets[nan_mask]
    mean = targets.mean()
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - mean) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-12))




class Run_deep_Rf():

    def __init__(self, device, relay, model, loss_fn, optimizer, scheduler, callback, last_ckpt_path, early_stopping=False, total_epoch = 1000):

        # relay : True or False --> True 일 경우 이어서 학습
        self.device = device
        self.relay = relay
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callback = callback
        self.start_epoch = 0
        self.total_epoch = total_epoch
        self.error_log_train= []
        self.r2_log_train=[]
        self.error_log_valid = []
        self.r2_log_valid=[]
        self.early_stopping = early_stopping
        
        if self.relay:
            self.last_ckpt = torch.load(last_ckpt_path)
            self.model.load_state_dict(self.last_ckpt['model_state_dict'])
            self.optimizer.load_state_dict(self.last_ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(self.last_ckpt['scheduler_state_dict'])
            self.start_epoch = self.last_ckpt['epoch']+1
            self.error_log_train = self.last_ckpt.get('train_loss', [])
            self.r2_log_train = self.last_ckpt.get('train_r2', [])
            self.error_log_valid = self.last_ckpt.get('valid_loss', [])
            self.r2_log_valid = self.last_ckpt.get('valid_r2', [])

            if self.start_epoch >=  self.total_epoch:
                raise ValueError(f'총 epoch: {self.total_epoch}가 이전에 저장된 epoch: {self.start_epoch} 보다 작음')
               
    def __call__(self, train_loader, valid_loader):
        
        for epoch in tqdm(range(self.start_epoch, self.total_epoch)):
            self.model.train()
            train_out_all=[]
            train_label_all=[]
            valid_out_all=[]
            valid_label_all=[]
            for com, elu1, elu2 in train_loader:
                com_train_batch = com.to(self.device)
                elu1_train_batch = elu1.to(self.device)
                elu2_train_batch = elu2.to(self.device)
                out = self.model(com_train_batch, elu1_train_batch, elu2_train_batch)   
                self.optimizer.zero_grad()             
                loss = self.loss_fn(out, com_train_batch.y)
                loss.backward()
                self.optimizer.step()
                
                train_out_all.append(out.detach().view(-1))
                train_label_all.append(com_train_batch.y.detach().view(-1))
                
            train_out_all = torch.cat(train_out_all, dim=0)
            train_label_all = torch.cat(train_label_all, dim=0)
            train_epoch_MAE = self.loss_fn(train_out_all, train_label_all, purpose = 'lookup')
            self.error_log_train.append(train_epoch_MAE.item())
            train_epoch_r2 = r2_score(train_out_all, train_label_all)
            self.r2_log_train.append(train_epoch_r2.item())
        
            self.model.eval()
            with torch.no_grad():      
                for com_val, elu1_val, elu2_val in valid_loader:
                    com_valid_batch = com_val.to(self.device)
                    elu1_valid_batch = elu1_val.to(self.device)
                    elu2_valid_batch = elu2_val.to(self.device)
                    pred = self.model(com_valid_batch, elu1_valid_batch, elu2_valid_batch)                    

                    valid_out_all.append(pred.detach().view(-1))
                    valid_label_all.append(com_valid_batch.y.detach().view(-1))
        
            valid_out_all = torch.cat(valid_out_all, dim=0)
            valid_label_all = torch.cat(valid_label_all, dim=0)
            valid_epoch_MAE = self.loss_fn(valid_out_all, valid_label_all, purpose = 'lookup')
            self.error_log_valid.append(valid_epoch_MAE.item())
            valid_epoch_r2 = r2_score(valid_out_all, valid_label_all)
            self.r2_log_valid.append(valid_epoch_r2.item())
        
            # schedular
            self.scheduler.step()
        
            # Early stop & callback
            early_stop = self.callback(epoch, self.model, self.optimizer, self.scheduler, 
                           train_loss = self.error_log_train,
                           train_r2 = self.r2_log_train,
                           valid_loss = self.error_log_valid,
                           valid_r2 = self.r2_log_valid)
                           
        
            print(f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f}')
        
            logging_path = 'train&valid_logging.txt'
            with open(logging_path, 'a') as f:
                f.write(f'Epoch : {epoch+1} | Train_loss : {self.error_log_train[-1]:.4f}, Train_r2 : {self.r2_log_train[-1]:.4f} | Valid_loss : {self.error_log_valid[-1]:.4f}, Valid_r2 : {self.r2_log_valid[-1]:.4f}')

            if early_stop:
                if self.early_stopping == True:
                    print(f'Early stopping at epoch {epoch+1}')
                    break