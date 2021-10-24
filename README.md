# Google-Brain-Ventilator

1. ```run1.sh``` and ```run2.sh``` are the scripts i use to run training. usually i nohup them w ```run.sh```
2. ```validate.sh``` does validation and feature importance
3. ```calculate_cv.py``` calculates cv and outputs in ```cv.txt```
4. ```predict.sh``` to make predictions, generate prediction file, and save 10-fold predictions

## files
```Network.py``` has the architecture
```Dataset.py``` has the dataset object
```Functions.py``` has some functions i use (mainly add_features)
```Logger.py``` is the custom csv logger i use to log train/val loss and metrics
