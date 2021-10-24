for i in {0..9};do
  python run_pl.py --fold $i --nfolds 10 --epochs 12 --embed_dim 256 --nheads 16 --nlayers 3 --gpu_id 0,1 \
  --pos_encode LSTM --seed 2020 --nfeatures 50 --batch_size 256 --path ../.. --workers 8 --dropout 0 \
  --target_file ensemble_shujun_pp_zidmie.csv
done
