import numpy as np
import pandas as pd
import gc
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import os

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

#new_rc_df=pd.read_csv(os.path.join(args.path,'r_c_analysis.csv'))

# train_breath_id=np.array(train['breath_id'])
# for i in tqdm(range(len(new_rc_df))):
#     breath_id=new_rc_df['breath_id'].iloc[i]
#     indices=(train_breath_id==breath_id)
    #train['R'][indices]=new_rc_df['R'][i]
    #train['C'][indices]=new_rc_df['C'][i]



masks=np.array(train['u_out']==0).reshape(-1, 80)
#exit()
targets = train[['pressure']].to_numpy().reshape(-1, 80)
#exit()
if os.path.isfile('train_new_rc.npy') and os.path.isfile('test_new_rc.npy'):
    train=np.load('train_new_rc.npy')
    test=np.load('test_new_rc.npy')
    columns=np.load('columns.npy',allow_pickle=True)
else:
    r_cs = pd.read_csv(os.path.join(args.path,'r_c_analysis.csv'))
    merged = train.merge(r_cs, on=['breath_id'], how='left')

    merged['R3'] = merged['R2'] * (merged['diff'] > 0.1) + merged['R_x'] * (merged['diff'] <= 0.1)
    merged['C3'] = merged['C2'] * (merged['diff'] > 0.1) + merged['C_x'] * (merged['diff'] <= 0.1)

    train['R'] = merged['R3'].values
    train['C'] = merged['C3'].values

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

    np.save('train_new_rc',train)
    np.save('test_new_rc',test)

args.nfeatures=train.shape[-1]
#exit()

from sklearn.model_selection import KFold

kf = KFold(n_splits=args.nfolds,random_state=args.seed,shuffle=True)





#exit()





#initialize model
model = SAKTModel(args.nfeatures, 10, 1, embed_dim=args.embed_dim, pos_encode=args.pos_encode,
                  max_seq=args.max_seq, nlayers=args.nlayers,rnnlayers=args.rnnlayers,
                  dropout=args.dropout,nheads=args.nheads).to(device)
criterion = nn.L1Loss(reduction='none')


# opt_level = 'O1'
# model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
model=nn.DataParallel(model)
model_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_models'
log_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_logs'
results_dir=f'{args.pos_encode}_rnn{args.rnnlayers}_transformer{args.nlayers}_val_results'

errors=[]
for i in tqdm(range(len(columns),len(columns)+1)):
    val_features=[train[i] for i in list(kf.split(train))[args.fold][1]]
    val_targets=[targets[i] for i in list(kf.split(targets))[args.fold][1]]
    val_masks=[masks[i] for i in list(kf.split(targets))[args.fold][1]]

    val_features=np.array(val_features)
    val_features_shape=val_features.shape[:2]

    val_features=val_features.reshape(-1,args.nfeatures)
    if i!=len(columns):
        np.random.shuffle(val_features[:,i])
    val_features=val_features.reshape(*val_features_shape,args.nfeatures)
    #exit()

    #print(f"### in total there are {len(val_features)} in val###")

    model.load_state_dict(torch.load(f'{model_dir}/model{args.fold}.pth'))


    #del train_features

    val_dataset = SAKTDataset(val_features,val_targets,val_masks)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=args.workers)
    #del val_features

    val_steps=len(val_dataloader)
    model.eval()
    val_metric=[]
    val_loss=0
    t=time.time()
    preds=[]
    truths=[]
    val_masks=[]
    all_features=[]
    for step,batch in enumerate(val_dataloader):
        features,gt,mask=batch
        features=features.to(device)
        gt=gt.to(device)
        mask=mask.to(device)
        with torch.no_grad():
            all_features.append(features.cpu())
            output=model(features,None)

            loss=criterion(output,gt)
            loss=torch.masked_select(loss,mask)
            loss=loss.mean()
            val_loss+=loss.item()
            #val_metric.append(MCMAE(output.reshape(-1,4),labels.reshape(-1,4),stds[-4:]))
            preds.append(output.cpu())
            truths.append(gt.cpu())
            val_masks.append(mask.cpu())
        # print ("Validation Step [{}/{}] Loss: {:.3f} Time: {:.1f}"
        #                    .format(step+1, val_steps, val_loss/(step+1), time.time()-t),end='\r',flush=True)

    preds=torch.cat(preds).numpy()
    all_features=torch.cat(all_features).numpy()
    truths=torch.cat(truths).numpy()
    val_masks=torch.cat(val_masks).numpy()
    val_metric=(np.abs(truths-preds)*val_masks).sum()/val_masks.sum()#*stds['pressure']
    #exit()
    #print('')
    #val_metric=torch.stack(val_metric).mean().cpu().numpy()
    val_loss/=(step+1)
    errors.append(val_metric)
    #break

val_features=[train[i] for i in list(kf.split(train))[args.fold][1]]
val_features=np.array(val_features)
u_in=val_features[:,:,list(columns).index('u_in')]

with open(f'{results_dir}/fold{args.fold}_new_rc.p','wb+') as f:
    pickle.dump([preds,truths,u_in,val_masks],f)
