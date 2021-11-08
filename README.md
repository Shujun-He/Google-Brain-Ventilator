# Google-Brain-Ventilator

1. ```run.sh``` is used to run training.
3. ```calculate_cv.py``` calculates cv and outputs in ```cv.txt```
3. ```predict.sh``` to make predictions, generate prediction file, and save 10-fold predictions

## files
1. ```Network.py``` has the architecture
2. ```Dataset.py``` has the dataset object
3. ```Functions.py``` has some functions i use (mainly add_features)
4. ```Logger.py``` is the custom csv logger i use to log train/val loss and metrics
