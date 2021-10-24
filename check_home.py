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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1',  help='which gpu to use')
    parser.add_argument('--path', type=str, default='../../input', help='path of csv file with DNA sequences and labels')
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
    parser.add_argument('--nfeatures', type=int, default=100, help='amount of features')
    parser.add_argument('--nheads', type=int, default=4, help='number of self-attention heads')
    opts = parser.parse_args()
    return opts

args=get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# device = torch.device("cpu")


#get features
t=time.time()

players = pd.read_csv(os.path.join(args.path,'players.csv'))
rosters = pd.read_pickle(os.path.join(args.path,'rosters_train.pkl'))
targets = pd.read_pickle(os.path.join(args.path,'nextDayPlayerEngagement_train.pkl'))
scores = pd.read_pickle(os.path.join(args.path,'playerBoxScores_train.pkl'))
awards = pd.read_pickle(os.path.join(args.path,'awards_train.pkl'))
awards['wonAward']=1
scores = scores.groupby(['playerId', 'date']).sum().reset_index()
playerstwitter=pd.read_pickle(os.path.join(args.path,'playerTwitterFollowers_train.pkl'))
teamstwitter=pd.read_pickle(os.path.join(args.path,'teamTwitterFollowers_train.pkl')).rename(columns={'numberOfFollowers':'TeamnumberOfFollowers'})
season_train=pd.read_csv(os.path.join(args.path,'season_train.csv'))
transactions=pd.read_csv(os.path.join(args.path,'transactions_train.csv'))
transactions=transactions[transactions['playerId']==transactions['playerId']]
transactions['playerId']=transactions['playerId'].astype('int')
transactions['traded']=1
teamscores=pd.read_pickle(os.path.join(args.path,'teamBoxScores_train.pkl'))
#exit()


targets_cols = ['playerId', 'target1', 'target2', 'target3', 'target4', 'date']
players_cols = ['playerId', 'primaryPositionName','playerForTestSetAndFuturePreds']
rosters_cols = ['playerId', 'teamId', 'status', 'date']
teamscore_cols = ['teamId', 'date','home']
awards_cols= ['playerId', 'date', 'wonAward']
scores_cols = ['playerId', 'battingOrder', 'gamesPlayedBatting', 'flyOuts',
       'groundOuts', 'runsScored', 'doubles', 'triples', 'homeRuns',
       'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch',
       'atBats', 'caughtStealing', 'stolenBases', 'groundIntoDoublePlay',
       'groundIntoTriplePlay', 'plateAppearances', 'totalBases', 'rbi',
       'leftOnBase', 'sacBunts', 'sacFlies', 'catchersInterference',
       'pickoffs', 'gamesPlayedPitching', 'gamesStartedPitching',
       'completeGamesPitching', 'shutoutsPitching', 'winsPitching',
       'lossesPitching', 'flyOutsPitching', 'airOutsPitching',
       'groundOutsPitching', 'runsPitching', 'doublesPitching',
       'triplesPitching', 'homeRunsPitching', 'strikeOutsPitching',
       'baseOnBallsPitching', 'intentionalWalksPitching', 'hitsPitching',
       'hitByPitchPitching', 'atBatsPitching', 'caughtStealingPitching',
       'stolenBasesPitching', 'inningsPitched', 'saveOpportunities',
       'earnedRuns', 'battersFaced', 'outsPitching', 'pitchesThrown', 'balls',
       'strikes', 'hitBatsmen', 'balks', 'wildPitches', 'pickoffsPitching',
       'rbiPitching', 'gamesFinishedPitching', 'inheritedRunners',
       'inheritedRunnersScored', 'catchersInterferencePitching',
       'sacBuntsPitching', 'sacFliesPitching', 'saves', 'holds', 'blownSaves',
       'assists', 'putOuts', 'errors', 'chances', 'date']

feature_cols = ['playerId', 'playerForTestSetAndFuturePreds','label_transactions','traded',
        'home','wonAward','inSeason',
       'numberOfFollowers','label_primaryPositionName',
       'gamesPlayedBatting', 'flyOuts',
       'groundOuts', 'runsScored', 'doubles', 'triples', 'homeRuns',
       'strikeOuts', 'baseOnBalls', 'intentionalWalks', 'hits', 'hitByPitch',
       'atBats', 'caughtStealing', 'stolenBases', 'groundIntoDoublePlay',
       'groundIntoTriplePlay', 'plateAppearances', 'totalBases', 'rbi',
       'leftOnBase', 'sacBunts', 'sacFlies', 'catchersInterference',
       'pickoffs', 'gamesPlayedPitching', 'gamesStartedPitching',
       'completeGamesPitching', 'shutoutsPitching', 'winsPitching',
       'lossesPitching', 'flyOutsPitching', 'airOutsPitching',
       'groundOutsPitching', 'runsPitching', 'doublesPitching',
       'triplesPitching', 'homeRunsPitching', 'strikeOutsPitching',
       'baseOnBallsPitching', 'intentionalWalksPitching', 'hitsPitching',
       'hitByPitchPitching', 'atBatsPitching', 'caughtStealingPitching',
       'stolenBasesPitching', 'inningsPitched', 'saveOpportunities',
       'earnedRuns', 'battersFaced', 'outsPitching', 'pitchesThrown', 'balls',
       'strikes', 'hitBatsmen', 'balks', 'wildPitches', 'pickoffsPitching',
       'rbiPitching', 'gamesFinishedPitching', 'inheritedRunners',
       'inheritedRunnersScored', 'catchersInterferencePitching',
       'sacBuntsPitching', 'sacFliesPitching', 'saves', 'holds', 'blownSaves',
       'assists', 'putOuts', 'errors', 'chances','target1_mean',
 'target1_median',
 'target1_std',
 'target1_min',
 'target1_max',
 'target1_prob',
 'target2_mean',
 'target2_median',
 'target2_std',
 'target2_min',
 'target2_max',
 'target2_prob',
 'target3_mean',
 'target3_median',
 'target3_std',
 'target3_min',
 'target3_max',
 'target3_prob',
 'target4_mean',
 'target4_median',
 'target4_std',
 'target4_min',
 'target4_max',
 'target4_prob']

playertwittercol=['playerId', 'date','numberOfFollowers']
teamtwittercol=['teamId', 'date','TeamnumberOfFollowers']
transactions_cols=['traded','playerId','date','typeCode']

player_target_stats = pd.read_csv(os.path.join(args.path,"player_target_stats.csv"))
data_names=player_target_stats.columns.values.tolist()

# creat dataset
train = targets[targets_cols].merge(players[players_cols], on=['playerId'], how='left')
train = train.merge(rosters[rosters_cols], on=['playerId', 'date'], how='left')
train = train.merge(scores[scores_cols], on=['playerId', 'date'], how='left')
train = train.merge(player_target_stats, how='inner', left_on=["playerId"],right_on=["playerId"])
print(train.shape)
train = train.merge(playerstwitter[playertwittercol], on=['playerId', 'date'], how='left')
print(train.shape)
train = train.merge(teamstwitter[teamtwittercol], on=['teamId', 'date'], how='left')
print(train.shape)
awards=awards.drop_duplicates(subset=['playerId','date'])
train = train.merge(awards[awards_cols], on=['playerId', 'date'], how='left')
print(train.shape)
#exit()
train['wonAward']=train['wonAward'].fillna(0)
#print(train.shape)
train = train.merge(season_train, on=['date'], how='left')
print(train.shape)
#print(train.shape)
transactions=transactions.drop_duplicates(subset=['playerId','date'])
train = train.merge(transactions[transactions_cols], on=['playerId','date'], how='left')
print(train.shape)
teamscores=teamscores.drop_duplicates(subset=['teamId','date'])
train = train.merge(teamscores[teamscore_cols], on=['teamId','date'], how='left')
#teamscores
print(train.shape)
teamscores['gameHour']=[int(teamscores.gameTimeUTC.iloc[i].split('T')[1].split(':')[0]) for i in range(len(teamscores))]

#print(train.shape)
#exit()

#train.numberOfFollowers=np.log10(train.numberOfFollowers)
#train.numberOfFollowers=train.numberOfFollowers.fillna(0)
#train.TeamnumberOfFollowers=np.log10(train.TeamnumberOfFollowers)
#train.TeamnumberOfFollowers=train.TeamnumberOfFollowers.fillna(0)
#exit()

# label encoding
player2num = {c: i for i, c in enumerate(train['playerId'].unique())}
position2num = {c: i for i, c in enumerate(train['primaryPositionName'].unique())}
teamid2num = {c: i for i, c in enumerate(train['teamId'].unique())}
status2num = {c: i for i, c in enumerate(train['status'].unique())}
transactions2num = {c: i for i, c in enumerate(train['typeCode'].unique())}
train['label_playerId'] = train['playerId'].map(player2num)
train['label_primaryPositionName'] = train['primaryPositionName'].map(position2num)
train['label_teamId'] = train['teamId'].map(teamid2num)
train['label_status'] = train['status'].map(status2num)
train['label_transactions'] = train['typeCode'].map(transactions2num)

# plt.hist(train['label_transactions'])
# plt.show()
# exit()



feature_cols+=['target1', 'target2', 'target3', 'target4']

print(f'###number of features is {len(feature_cols)}###')

train_X = train[feature_cols]
exit()
