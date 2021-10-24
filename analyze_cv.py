import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

nfolds=1

min_maes=[]
predictions=[]
ground_truths=[]
masks=[]
u_ins=[]
for i in range(nfolds):
    # df=pd.read_csv(f'LSTM_logs/log_fold{i}.csv')
    # min_maes.append(df.val_mcmae.min())
    with open(f'LSTM_val_results/fold{i}.p','rb') as f:
        preds,truths, u_in, mask=pickle.load(f)

    predictions.append(preds)
    ground_truths.append(truths)
    masks.append(mask)
    u_ins.append(u_in)

errors=[]
for pred, gt, mask in zip(predictions[0],ground_truths[0],masks[0]):
    errors.append(np.abs(pred[mask]-gt[mask]).mean())

top=np.argsort(errors)[-10:]
cnt=0

plt.subplots(2,5)
for i in range(5):
    for j in range(2):

        plt.subplot(2,5,cnt+1)
        for index in range(80):
            if masks[0][top[cnt]][index]!=masks[0][top[cnt]][index+1]:
                break
        plt.axvline(x=index,color='c')
        plt.plot(predictions[0][top[cnt]],label='pred')
        plt.plot(ground_truths[0][top[cnt]],label='truth')
        plt.plot(u_ins[0][top[cnt]],label='u_in')
        plt.title(f'MAE {errors[top[cnt]]}')
        plt.legend()
        #plt.show()
        cnt+=1

fig = plt.gcf()
fig.set_size_inches(18.5, 8.5)
#plt.show()
plt.savefig('most_errors_fold.png',dpi=200)
