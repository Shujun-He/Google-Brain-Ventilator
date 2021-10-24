import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


tests=[64,66]
start=100
fold=0
plt.subplots(2,5)
for i in range(5):
    for j in range(2):
        try:
            df1=pd.read_csv(f'../test{tests[0]}/LSTM_logs/log_fold{fold}.csv')
            df2=pd.read_csv(f'../test{tests[1]}/LSTM_rnn3_transformer3_logs/log_fold{fold}.csv')

            plt.subplot(2,5,fold+1)
            plt.plot(df1.val_mcmae[start:],label=f'test{tests[0]}-fold{fold}-min:{str(df1.val_mcmae.min())[:6]}')
            plt.plot(df2.val_mcmae[start:],label=f'test{tests[1]}-fold{fold}-min:{str(df2.val_mcmae.min())[:6]}')
            plt.legend()
            #plt.show()
        except:
            pass
        fold+=1
fig = plt.gcf()
fig.set_size_inches(18.5, 8.5)
#plt.show()
plt.savefig('compare.png',dpi=200)
