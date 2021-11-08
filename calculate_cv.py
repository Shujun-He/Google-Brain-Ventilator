import pandas as pd
import numpy as np
import pickle

nfolds=10

min_maes=[]
predictions=[]
ground_truths=[]
masks=[]
for i in range(nfolds):
    # df=pd.read_csv(f'LSTM_logs/log_fold{i}.csv')
    # min_maes.append(df.val_mcmae.min())
    with open(f'LSTM_rnn3_transformer3_val_results/fold{i}.p','rb') as f:
        preds, truths, mask=pickle.load(f)

    predictions.append(preds)
    ground_truths.append(truths)
    masks.append(mask)

predictions=np.concatenate(predictions)#.reshape(-1)
ground_truths=np.concatenate(ground_truths)
masks=np.concatenate(masks)

cv=np.mean(np.abs(predictions[masks]-ground_truths[masks]))
with open('cv.txt','w+') as f:
    f.write(f'cv mcmae: {cv}')

print(cv)
