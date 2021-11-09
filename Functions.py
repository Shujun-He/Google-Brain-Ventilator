import numpy as np
import pandas as pd
import gc
import random
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import time




def MCMAE(output,targets,stds):
    maes=[]
    for i in range(output.shape[1]):
        mae=torch.abs(output[:,i]-targets[:,i]).mean()*stds[i]
        maes.append(mae)
    maes=torch.stack(maes).mean()
    return maes

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        lr=param_group['lr']
    return lr

class lr_AIAYN():
    '''
    Learning rate scheduler from the paper:
    Attention is All You Need
    '''
    def __init__(self,optimizer,d_model,warmup_steps=4000,factor=1):
        self.optimizer=optimizer
        self.d_model=d_model
        self.warmup_steps=warmup_steps
        self.step_num=0
        self.factor=factor

    def step(self):
        self.step_num+=1
        lr=self.d_model**-0.5*np.min([self.step_num**-0.5,
                                      self.step_num*self.warmup_steps**-1.5])*self.factor
        update_lr(self.optimizer,lr)
        return lr

def add_features(df):
    #df['area'] = df['time_step'] * df['u_in']
    #df['area'] = df.groupby('breath_id')['area'].cumsum()

    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()


    # fast area calculation
    df['time_delta'] = df['time_step'].diff()
    df['time_delta'].fillna(0, inplace=True)
    df['time_delta'].mask(df['time_delta'] < 0, 0, inplace=True)
    df['tmp'] = df['time_delta'] * df['u_in']
    df['area_true'] = df.groupby('breath_id')['tmp'].cumsum()
    df['tmp'] = df['u_out']*(-1)+1 # inversion of u_out

    df['u_in_lag1'] = df.groupby('breath_id')['u_in'].shift(1)
    #df['u_out_lag1'] = df.groupby('breath_id')['u_out'].shift(1)
    df['u_in_lag_back1'] = df.groupby('breath_id')['u_in'].shift(-1)
    #df['u_out_lag_back1'] = df.groupby('breath_id')['u_out'].shift(-1)
    df['u_in_lag2'] = df.groupby('breath_id')['u_in'].shift(2)
    #df['u_out_lag2'] = df.groupby('breath_id')['u_out'].shift(2)
    df['u_in_lag_back2'] = df.groupby('breath_id')['u_in'].shift(-2)
    #df['u_out_lag_back2'] = df.groupby('breath_id')['u_out'].shift(-2)
    df['u_in_lag3'] = df.groupby('breath_id')['u_in'].shift(3)
    #df['u_out_lag3'] = df.groupby('breath_id')['u_out'].shift(3)
    df['u_in_lag_back3'] = df.groupby('breath_id')['u_in'].shift(-3)
    #df['u_out_lag_back3'] = df.groupby('breath_id')['u_out'].shift(-3)
    df['u_in_lag4'] = df.groupby('breath_id')['u_in'].shift(4)
    #df['u_out_lag4'] = df.groupby('breath_id')['u_out'].shift(4)
    df['u_in_lag_back4'] = df.groupby('breath_id')['u_in'].shift(-4)
    #df['u_out_lag_back4'] = df.groupby('breath_id')['u_out'].shift(-4)
    df = df.fillna(0)

    df['breath_id__u_in__max'] = df.groupby(['breath_id'])['u_in'].transform('max')
    #df['breath_id__u_out__max'] = df.groupby(['breath_id'])['u_out'].transform('max')

    df['u_in_diff1'] = df['u_in'] - df['u_in_lag1']
    #df['u_out_diff1'] = df['u_out'] - df['u_out_lag1']
    df['u_in_diff2'] = df['u_in'] - df['u_in_lag2']
    #df['u_out_diff2'] = df['u_out'] - df['u_out_lag2']

    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']

    df['breath_id__u_in__diffmax'] = df.groupby(['breath_id'])['u_in'].transform('max') - df['u_in']
    df['breath_id__u_in__diffmean'] = df.groupby(['breath_id'])['u_in'].transform('mean') - df['u_in']

    df['u_in_diff3'] = df['u_in'] - df['u_in_lag3']
    #df['u_out_diff3'] = df['u_out'] - df['u_out_lag3']
    df['u_in_diff4'] = df['u_in'] - df['u_in_lag4']
    #df['u_out_diff4'] = df['u_out'] - df['u_out_lag4']
    #df['cross']= df['u_in']*df['u_out']
    #df['cross2']= df['time_step']*df['u_out']

    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['R__C'] = df["R"].astype(str) + '__' + df["C"].astype(str)
    df = pd.get_dummies(df)
    return df
