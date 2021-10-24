python -i predict.py --nfolds 10 --epochs 32 --embed_dim 256 --nheads 16 --nlayers 3 --gpu_id 0,1 \
--pos_encode LSTM --seed 2020 --nfeatures 35 --batch_size 1024
