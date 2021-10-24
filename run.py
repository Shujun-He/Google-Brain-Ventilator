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
    parser.add_argument('--rnnlayers', type=int, default=3, help='number of reisdual rnn blocks')
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
print("Loading data")
train = pd.read_csv(os.path.join(args.path,'train.csv'))
test = pd.read_csv(os.path.join(args.path,'test.csv'))

masks=np.array(train['u_out']==0).reshape(-1, 80)
targets = train[['pressure']].to_numpy().reshape(-1, 80)
#exit()

# if os.path.isfile('train.npy') and os.path.isfile('test.npy'):
#     train=np.load('train.npy')
#     test=np.load('test.npy')
#     columns=np.load('columns.npy',allow_pickle=True)
# else:
print("Adding features")
train = add_features(train)
test = add_features(test)

print("Dropping some features")

train.drop(['pressure', 'id', 'breath_id'], axis=1, inplace=True)
test = test.drop(['id', 'breath_id'], axis=1)
columns=train.columns
np.save('columns',np.array(train.columns))

print("Normalizing")
RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)

print("Reshaping")
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])

    #np.save('train',train)
    #np.save('test',test)

args.nfeatures=train.shape[-1]
#exit()

from sklearn.model_selection import KFold

kf = KFold(n_splits=args.nfolds,random_state=args.seed,shuffle=True)

#Y= [int(group[i][1][0,playerintestingindex]) for i in range(len(group))]
#exit()
# train=group[list(kf.split(group))[args.fold][0]]
# val=group[list(kf.split(group))[args.fold][1]]

#exit()

train_features=[train[i] for i in list(kf.split(train))[args.fold][0]]
val_features=[train[i] for i in list(kf.split(train))[args.fold][1]]
train_targets=[targets[i] for i in list(kf.split(targets))[args.fold][0]]
val_targets=[targets[i] for i in list(kf.split(targets))[args.fold][1]]
train_masks=[masks[i] for i in list(kf.split(targets))[args.fold][0]]
val_masks=[masks[i] for i in list(kf.split(targets))[args.fold][1]]

#exit()

print(f"### in total there are {len(train_features)} in train###")
print(f"### in total there are {len(val_features)} in val###")

#exit()

train_dataset = SAKTDataset(train_features,train_targets,train_masks)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
del train_features

val_dataset = SAKTDataset(val_features,val_targets,val_masks)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=args.workers)
del val_features

#exit()

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

#initialize model
model = SAKTModel(args.nfeatures, 10, 1, embed_dim=args.embed_dim, pos_encode=args.pos_encode,
                  max_seq=args.max_seq, nlayers=args.nlayers, rnnlayers=args.rnnlayers,
                  dropout=args.dropout,nheads=args.nheads).to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#opt_level = 'O1'
#model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
optimizer = Ranger(model.parameters(), lr=8e-4)
criterion = nn.L1Loss(reduction='none')


# opt_level = 'O1'
# model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

model=nn.DataParallel(model)

#model.load_state_dict(torch.load('models/model1_epoch6.pth'))

from Logger import *
model_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_models'
log_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_logs'
results_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_val_results'

os.system(f'mkdir {log_dir}')
logger=CSVLogger(['epoch','train_loss','val_loss','val_mcmae'],f'{log_dir}/log_fold{args.fold}.csv')
os.system(f'mkdir {model_dir}')
os.system(f'mkdir {results_dir}')

#exit()

val_metric = 100
best_metric = 100
cos_epoch=int(args.epochs*0.75)
#scheduler=lr_AIAYN(optimizer,args.embed_dim,warmup_steps=3000)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,(args.epochs-cos_epoch)*len(train_dataloader))
steps_per_epoch=len(train_dataloader)
val_steps=len(val_dataloader)
for epoch in range(args.epochs):
    model.train()
    train_loss=0
    t=time.time()
    for step,batch in enumerate(train_dataloader):
        #series=batch.to(device)#.float()
        features,targets,mask=batch
        features=features.to(device)
        targets=targets.to(device)
        mask=mask.to(device)
        #exit()

        optimizer.zero_grad()
        output=model(features,None)
        #exit()
        #exit()

        loss=criterion(output,targets)#*loss_weight_vector
        loss=torch.masked_select(loss,mask)
        loss=loss.mean()
        loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        #scheduler.step()
        print ("Step [{}/{}] Loss: {:.3f} Time: {:.1f}"
                           .format(step+1, steps_per_epoch, train_loss/(step+1), time.time()-t),end='\r',flush=True)
        if epoch > cos_epoch:
            scheduler.step()
        #break
    print('')
    train_loss/=(step+1)

    #exit()
    model.eval()
    val_metric=[]
    val_loss=0
    t=time.time()
    preds=[]
    truths=[]
    masks=[]
    for step,batch in enumerate(val_dataloader):
        features,targets,mask=batch
        features=features.to(device)
        targets=targets.to(device)
        mask=mask.to(device)
        with torch.no_grad():
            output=model(features,None)

            loss=criterion(output,targets)
            loss=torch.masked_select(loss,mask)
            loss=loss.mean()
            val_loss+=loss.item()
            #val_metric.append(MCMAE(output.reshape(-1,4),labels.reshape(-1,4),stds[-4:]))
            preds.append(output.cpu())
            truths.append(targets.cpu())
            masks.append(mask.cpu())
        print ("Validation Step [{}/{}] Loss: {:.3f} Time: {:.1f}"
                           .format(step+1, val_steps, val_loss/(step+1), time.time()-t),end='\r',flush=True)

    preds=torch.cat(preds).numpy()
    truths=torch.cat(truths).numpy()
    masks=torch.cat(masks).numpy()
    val_metric=(np.abs(truths-preds)*masks).sum()/masks.sum()#*stds['pressure']
    #exit()
    print('')
    #val_metric=torch.stack(val_metric).mean().cpu().numpy()
    val_loss/=(step+1)

    logger.log([epoch+1,train_loss,val_loss,val_metric])


    if val_metric < best_metric:
        best_metric=val_metric
        torch.save(model.state_dict(),f'{model_dir}/model{args.fold}.pth')
        with open(f'{results_dir}/fold{args.fold}.p','wb+') as f:
            pickle.dump([preds,truths],f)
