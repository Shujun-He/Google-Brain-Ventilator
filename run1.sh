for i in {1..4};do
  python run.py --fold $i --nfolds 10 --epochs 150 --embed_dim 256 --nheads 16 --nlayers 3 --rnnlayers 3\
  --gpu_id 0 \
  --pos_encode LSTM --seed 2020 --nfeatures 50 --batch_size 128 --path ../.. --workers 8 --dropout 0
done
