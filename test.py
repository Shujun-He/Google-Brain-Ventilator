import os
import warnings
from typing import Optional, Tuple

import pandas as pd


class Environment:
    def __init__(self,
                 data_dir: str,
                 eval_start_day: int,
                 eval_end_day: Optional[int],
                 use_updated: bool,
                 multiple_days_per_iter: bool):
        warnings.warn('this is mock module for mlb')

        postfix = '_updated' if use_updated else ''

        # recommend to replace this with pickle, feather etc to speedup preparing data
        df_train = pd.read_csv(os.path.join(data_dir, f'train{postfix}.csv'))

        players = pd.read_csv(os.path.join(data_dir, 'players.csv'))

        self.players = players[players['playerForTestSetAndFuturePreds'] == True]['playerId'].astype(str)
        if eval_end_day is not None:
            self.df_train = df_train.set_index('date').loc[eval_start_day:eval_end_day]
        else:
            self.df_train = df_train.set_index('date').loc[eval_start_day:]
        self.date = self.df_train.index.values
        self.n_rows = len(self.df_train)
        self.multiple_days_per_iter = multiple_days_per_iter

        assert self.n_rows > 0, 'no data to emulate'
        self.preds=[]

    def predict(self, df: pd.DataFrame) -> None:
        # if you want to emulate public LB, store your prediction here and calculate MAE
        self.preds.append(df)
        #self.preds.append(self.df_train['nextDayPlayerEngagement'].iloc)

    def iter_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.multiple_days_per_iter:
            for i in range(self.n_rows // 2):
                date1 = self.date[2 * i]
                date2 = self.date[2 * i + 1]
                sample_sub1 = self._make_sample_sub(date1)
                sample_sub2 = self._make_sample_sub(date2)
                sample_sub = pd.concat([sample_sub1, sample_sub2]).reset_index(drop=True)
                df = self.df_train.loc[date1:date2]

                yield df, sample_sub.set_index('date')
        else:
            for i in range(self.n_rows):
                date = self.date[i]
                sample_sub = self._make_sample_sub(date)
                df = self.df_train.loc[date:date]

                yield df, sample_sub.set_index('date')

    def _make_sample_sub(self, date: int) -> pd.DataFrame:
        next_day = (pd.to_datetime(date, format='%Y%m%d') + pd.to_timedelta(1, 'd')).strftime('%Y%m%d')
        sample_sub = pd.DataFrame()
        sample_sub['date_playerId'] = next_day + '_' + self.players
        sample_sub['target1'] = 0
        sample_sub['target2'] = 0
        sample_sub['target3'] = 0
        sample_sub['target4'] = 0
        sample_sub['date'] = date
        return sample_sub


class MLBEmulator:
    def __init__(self,
                 data_dir: str = '../input/mlb-player-digital-engagement-forecasting',
                 eval_start_day: int = 20210401,
                 eval_end_day: Optional[int] = 20210715,
                 use_updated: bool = True,
                 multiple_days_per_iter: bool = False):
        self.data_dir = data_dir
        self.eval_start_day = eval_start_day
        self.eval_end_day = eval_end_day
        self.use_updated = use_updated
        self.multiple_days_per_iter = multiple_days_per_iter

    def make_env(self) -> Environment:
        return Environment(self.data_dir,
                           self.eval_start_day,
                           self.eval_end_day,
                           self.use_updated,
                           self.multiple_days_per_iter)

embed_dim=384
max_seq=64
nlayers=6
#nfeatures=100
nheads=24
DEBUG=False
emulation_mode = True


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
from functools import reduce
from tqdm import tqdm
#import mlb
import torch.nn as nn
import numpy as np
import torch
import math

BASE_DIR = Path('../../input/')
TRAIN_DIR = Path('../../input/')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

null = np.nan
true = True
false = False

players = pd.read_csv(BASE_DIR / 'players.csv')

rosters = pd.read_pickle(TRAIN_DIR / 'rosters_train.pkl')
targets = pd.read_pickle(TRAIN_DIR / 'nextDayPlayerEngagement_train.pkl')
scores = pd.read_pickle(TRAIN_DIR / 'playerBoxScores_train.pkl')
scores = scores.groupby(['playerId', 'date']).sum().reset_index()
playerstwitter=pd.read_pickle(TRAIN_DIR/'playerTwitterFollowers_train.pkl')
teamstwitter=pd.read_pickle(TRAIN_DIR/'teamTwitterFollowers_train.pkl').rename(columns={'numberOfFollowers':'TeamnumberOfFollowers'})
awards = pd.read_pickle(TRAIN_DIR/'awards_train.pkl')
awards['wonAward']=1
train=pd.read_csv(BASE_DIR / 'train.csv')
season_train=pd.read_csv('../../input/mlbseasons/season_train.csv')
sample_player_twitter=pd.DataFrame(eval(train['playerTwitterFollowers'].iloc[0]))
sample_awards=pd.DataFrame(eval(train['awards'].iloc[1215]))


transactions=pd.read_csv(TRAIN_DIR/'transactions_train.csv')
transactions=transactions[transactions['playerId']==transactions['playerId']]
transactions['playerId']=transactions['playerId'].astype('int')
transactions['traded']=1
#del train
sample_transactions=pd.DataFrame(eval(train['transactions'].iloc[150]))

teamscores=pd.read_pickle(TRAIN_DIR/'teamBoxScores_train.pkl')
games=pd.read_csv(TRAIN_DIR/'games_train.csv')
standings=pd.read_csv(TRAIN_DIR/'standings_train.csv')

for col in standings.columns:
    if standings[col].dtype=='object':
        print(col)
#process some columns for
standings['streakCode']=standings['streakCode'].fillna('0')
standings['wildCardEliminationNumber']=standings['wildCardEliminationNumber'].fillna('0')
streakCode=[]
leagueGamesBack=[]
sportGamesBack=[]
divisionGamesBack=[]
eliminationNumber=[]
wildCardEliminationNumber=[]
for i in range(len(standings)):
    streakCode.append(int(standings['streakCode'].iloc[i].replace('L','-').replace('W','')))
    leagueGamesBack.append(float(standings['leagueGamesBack'].iloc[i].replace('-','0')))
    sportGamesBack.append(float(standings['sportGamesBack'].iloc[i].replace('-','0')))
    divisionGamesBack.append(float(standings['divisionGamesBack'].iloc[i].replace('-','0')))
    eliminationNumber.append(float(standings['eliminationNumber'].iloc[i].replace('-','0').replace('E','0')))
    wildCardEliminationNumber.append(float(standings['wildCardEliminationNumber'].iloc[i].replace('-','-1').replace('E','0')))
standings['streakCode']=np.array(streakCode)
standings['leagueGamesBack']=np.array(leagueGamesBack).astype('int')
standings['sportGamesBack']=np.array(sportGamesBack).astype('int')
standings['divisionGamesBack']=np.array(divisionGamesBack).astype('int')
standings['eliminationNumber']=np.array(eliminationNumber).astype('int')
standings['wildCardLeader']=standings['wildCardLeader'].fillna(0).astype('int')
standings['wildCardEliminationNumber']=np.array(wildCardEliminationNumber).astype('int')

targets_cols = ['playerId', 'target1', 'target2', 'target3', 'target4', 'date']
players_cols = ['playerId', 'primaryPositionName']
rosters_cols = ['playerId', 'teamId', 'status', 'date']
awards_cols= ['playerId', 'date', 'wonAward']
teamscore_cols = ['teamId', 'date','home','homeWinner','awayWinner','gameWon']
standings_cols=['date','teamId','leagueRank','pct']
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

feature_cols = ['playerId','label_transactions','traded',
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
       'assists', 'putOuts', 'errors', 'chances','leagueRank','pct',
       'gameWon',
       'target1_mean',
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


player_target_stats = pd.read_csv("../input/player_target_stats.csv")
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
awards=awards.drop_duplicates(subset=['playerId','date'])
train = train.merge(awards[awards_cols], on=['playerId', 'date'], how='left')
train['wonAward']=train['wonAward'].fillna(0)
train = train.merge(season_train, on=['date'], how='left')
transactions=transactions.drop_duplicates(subset=['playerId','date'])
train = train.merge(transactions[transactions_cols], on=['playerId','date'], how='left')
teamscores=teamscores.drop_duplicates(subset=['teamId','date'])
teamscores=teamscores.merge(games,on=['gamePk','date'], how='left')
gameWon=[]
for i in range(len(teamscores)):
    if teamscores['home'].iloc[i]==1 and teamscores['homeWinner'].iloc[i]==1:
        gameWon.append(1)
    elif teamscores['home'].iloc[i]==0 and teamscores['awayWinner'].iloc[i]==1:
        gameWon.append(1)
    elif teamscores['awayWinner'].iloc[i]==0 and teamscores['homeWinner'].iloc[i]==0:
        gameWon.append(2)
    else:
        gameWon.append(0)

teamscores['gameWon']=gameWon


train = train.merge(teamscores[teamscore_cols], on=['teamId','date'], how='left')
print(train.shape)
train = train.merge(standings[standings_cols], on=['date','teamId'], how='left')
print(train.shape)

# train.numberOfFollowers=np.log10(train.numberOfFollowers)
# train.numberOfFollowers=train.numberOfFollowers.fillna(0)
# train.TeamnumberOfFollowers=np.log10(train.TeamnumberOfFollowers)
# train.TeamnumberOfFollowers=train.TeamnumberOfFollowers.fillna(0)

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
print(train.shape)

feature_cols+=['target1', 'target2', 'target3', 'target4']
target_cols=['target1', 'target2', 'target3', 'target4']
train_X = train[feature_cols]
print(len(feature_cols))
del train

categorical_features=['label_primaryPositionName','gamesPlayedBatting','inSeason','label_transactions','home','gameWon',]
categorical_offsets=[1]
categorical_features_indices=[]
total_categories=0
n_uniques=[]
for feature in categorical_features:
    n_unique=len(train_X[feature].unique())
    n_uniques.append(n_unique)
    total_categories+=n_unique

    train_X[feature]=train_X[feature].fillna(-categorical_offsets[-1])
    categorical_offsets.append(categorical_offsets[-1]+n_unique)
    categorical_features_indices.append(feature_cols.index(feature))

total_categories+=1
categorical_offsets=torch.Tensor(categorical_offsets[:-1]).to(device).long()
categorical_features_indices=torch.Tensor(categorical_features_indices).to(device).long()


#train_y = train[['target1', 'target2', 'target3', 'target4']]
#exit()
train_X=train_X.fillna(0)
binary_features=['wonAward','traded','homeWinner','awayWinner']
means=[]
stds=[]
numerical_features_indices=[]
for i in range(1,train_X.shape[1]):
    if feature_cols[i] not in categorical_features:
        numerical_features_indices.append(i)
        values=train_X.iloc[:,i]
        if feature_cols[i]=='numberOfFollowers' or feature_cols[i]=='TeamnumberOfFollowers':
            values=values[values!=0]
        if feature_cols[i] in binary_features:
            print(feature_cols[i])
            means.append(0)
            stds.append(1)
        else:
            means.append(values.mean())
            stds.append(values.std())
numerical_features_indices=torch.Tensor(numerical_features_indices).to(device).long()

#exit()

means=torch.Tensor(means).to(device)
stds=torch.Tensor(stds).to(device)

group=list(train_X.groupby('playerId'))
del train_X
print("###Making groups###")
playerfollowerindex=feature_cols.index('numberOfFollowers')
#teamfollowerindex=feature_cols.index('TeamnumberOfFollowers')
for i in tqdm(range(len(group))):
    group[i]=(group[i][0],group[i][1].values[-max_seq*2:,:-4].astype('float32'))
    for j in range(1,len(group[i][1])):
        if group[i][1][j,playerfollowerindex]==0:
            group[i][1][j,playerfollowerindex]=group[i][1][j-1,playerfollowerindex]


group_dict={}
for item in group:
    group_dict[item[0]]=item[1][-max_seq:]
print(group_dict[item[0]].shape)


del group
