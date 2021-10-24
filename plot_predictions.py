import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from sklearn.metrics import r2_score

with open('feature_metadata.p','rb') as f:
    metadata=pickle.load(f)
#{'means':means,"std":stds}
with open('val_results/fold1.p','rb') as f:
    preds,truths=pickle.load(f)

preds=preds.cpu().numpy()*metadata['std'].cpu().numpy()[-4:]+metadata['means'].cpu().numpy()[-4:]
truths=truths.cpu().numpy()*metadata['std'].cpu().numpy()[-4:]+metadata['means'].cpu().numpy()[-4:]

# indices=truths<20
# truths=truths[indices]
# preds=preds[indices]

plt.plot(preds,truths,'*')
plt.plot(truths,truths)
plt.title(f'r2: {r2_score(preds.reshape(-1),truths.reshape(-1))}\nmae:{np.abs(preds-truths).mean()}')
plt.show()
