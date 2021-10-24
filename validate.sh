for i in {8..9};do
  python validate.py --fold $i --nfolds 10 --epochs 150 --embed_dim 256 --nheads 16 --nlayers 3 --gpu_id 0 \
  --pos_encode LSTM --seed 2020 --nfeatures 50 --batch_size 512 --path ../.. --workers 4 --dropout 0
done
