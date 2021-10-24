import numpy as np
import pandas as pd
import gc
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import time
from Dataset import *
from Network import *
from Functions import *

try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
from ranger import Ranger
import pickle
import argparse
from sklearn.preprocessing import RobustScaler, normalize

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../..', help='path of csv file with DNA sequences and labels')
    parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each batch during training')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight dacay used in optimizer')
    parser.add_argument('--save_freq', type=int, default=1, help='saving checkpoints per save_freq epochs')
    parser.add_argument('--dropout', type=float, default=.1, help='transformer dropout')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--nfolds', type=int, default=5, help='number of cross validation folds')
    parser.add_argument('--fold', type=int, default=0, help='which fold to train')
    parser.add_argument('--val_freq', type=int, default=1, help='which fold to train')
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient_accumulation_steps')
    parser.add_argument('--max_seq', type=int, default=64, help='max_seq')
    parser.add_argument('--embed_dim', type=int, default=128, help='embedding dimension size')
    #parser.add_argument('--batch_size', type=int, default=2048, help='batch_size')
    parser.add_argument('--nlayers', type=int, default=3, help='nlayers')
    parser.add_argument('--nfeatures', type=int, default=5, help='amount of features')
    parser.add_argument('--nheads', type=int, default=4, help='number of self-attention heads')
    parser.add_argument('--seed', type=int, default=2020, help='seed')
    parser.add_argument('--pos_encode', type=str, default='LSTM', help='method of positional encoding')
    opts = parser.parse_args()
    return opts

args=get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# device = torch.device("cpu")


#get features
t=time.time()
feature_cols = ['R', 'C','time_step','u_in','u_out']
target_cols = ['pressure']

train = pd.read_csv(os.path.join(args.path,'train.csv'))
test = pd.read_csv(os.path.join(args.path,'test.csv'))

all_pressure = np.sort( train.pressure.unique() )
PRESSURE_MIN = all_pressure[0]
PRESSURE_STEP = all_pressure[1] - all_pressure[0]


submission=pd.read_csv(os.path.join(args.path,'sample_submission.csv'))

print("Adding features")
train = add_features(train)
test = add_features(test)

print("Dropping some features")
targets = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
test = test.drop(['id', 'breath_id'], axis=1)

print("Normalizing")
RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)

print("Reshaping")
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])

args.nfeatures=train.shape[-1]

test_dataset = TestDataset(test)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


MODELS=[]
for fold in range(args.nfolds):
    model = SAKTModel(args.nfeatures, 10, 1, embed_dim=args.embed_dim, pos_encode=args.pos_encode,
                      max_seq=args.max_seq, nlayers=args.nlayers,
                      dropout=args.dropout,nheads=args.nheads).to(device)
    model=nn.DataParallel(model)
    model.load_state_dict(torch.load(f'{args.pos_encode}_models/model{fold}_pl.pth'))
    model.eval()
    MODELS.append(model)


preds=[]
for batch in tqdm(test_dataloader):
    features=batch.to(device)
    #features=features
    with torch.no_grad():
        temp=[]
        for model in MODELS:
            output=model(features,None)
            temp.append(output)
        #temp=torch.mean(torch.stack(temp,0),0)#[0]
        temp=torch.median(torch.stack(temp,0),0)[0]
        temp=torch.round( (temp - PRESSURE_MIN)/PRESSURE_STEP ) * PRESSURE_STEP + PRESSURE_MIN

        preds.append(temp.cpu())

preds=torch.cat(preds).reshape(-1).numpy()

submission['pressure']=preds

true_test=pd.read_csv(os.path.join(args.path,'1280_breaths_pl.csv'))

for i in range(len(true_test)):
    index=int(true_test['id'].iloc[i])-1
    #submission.id
    submission['pressure'][index]=true_test['pressure'][i]


submission.to_csv('submission_pl.csv',index=False)
