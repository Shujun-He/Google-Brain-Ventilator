import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

nfolds=10
max_pressure=64.8209917386395

min_maes=[]
predictions=[]
ground_truths=[]
masks=[]
for i in range(nfolds):
    # df=pd.read_csv(f'LSTM_logs/log_fold{i}.csv')
    # min_maes.append(df.val_mcmae.min())
    with open(f'LSTM_val_results/fold{i}.p','rb') as f:
        preds, truths, u_in, mask=pickle.load(f)

    predictions.append(preds)
    ground_truths.append(truths)
    masks.append(mask)

predictions=np.concatenate(predictions)#.reshape(-1)
ground_truths=np.concatenate(ground_truths)
masks=np.concatenate(masks)

parabola_predictions=[]
ground_truths_predictions=[]

cnt=0
width_cutoff=3
parabola_errors=[]
os.system('mkdir parabola')
for p, gt, mask in zip(predictions,ground_truths,masks):
    if (gt==max_pressure).sum()>=width_cutoff:
        parabola_errors.append(np.abs(p[mask]-gt[mask]).mean())
        cnt+=1
        # plt.plot(p,'.',label='pred')
        # plt.plot(gt,'.',label='truth')
        # plt.legend()
        # plt.title(f"MAE:{parabola_errors[-1]}")
        # plt.savefig(f'parabola/{cnt}.png')
        #
        # #plt.show()
        # plt.close()
        # plt.clf()

print(cnt)
print(np.mean(parabola_errors))

#masks=np.concatenate(masks)

cv=np.mean(np.abs(predictions[masks]-ground_truths[masks]))
with open('cv.txt','w+') as f:
    f.write(f'cv mcmae: {cv}')

print(cv)
