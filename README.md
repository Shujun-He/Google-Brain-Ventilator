# Part of 1st place solution (LSTM CNN Transformer Encoder) of Google-Brain-Ventilator competition 

Competition website: https://www.kaggle.com/c/ventilator-pressure-prediction/overview <br />
Our solution write-up: https://www.kaggle.com/c/ventilator-pressure-prediction/discussion/285256

1. ```run.sh``` is used to run training. The only argument you need to change in run.sh is the ```--path``` argument. Change it to where you have ```train.csv``` and ```test.csv```
3. ```calculate_cv.py``` calculates cv and outputs in ```cv.txt```
3. ```predict.sh``` to make predictions, generate prediction file, and save 10-fold predictions. Similar to 1., change ```--path``` to where you have ```train.csv```, ```test.csv```, and ```sample_submission.csv```

## files
1. ```Network.py``` has the architecture
2. ```Dataset.py``` has the dataset object
3. ```Functions.py``` has some functions i use (mainly add_features)
4. ```Logger.py``` is the custom csv logger i use to log train/val loss and metrics
