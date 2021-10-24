import pandas as pd
import matplotlib.pyplot as plt

for i in range(10):
    df=pd.read_csv(f'logs/log_fold{i}.csv')
    plt.plot(df.val_mcmae)

plt.show()
