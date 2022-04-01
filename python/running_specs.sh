# LC
python -u ./train_cbmd.py  
# SAN
python -u ./train_cbmd.py --layers_size_enc 700 700 700 --layers_size_enc2 300 --layers_size_dec 900 900 900 --layers_size_prior 300 --layers_size_cls 700 --outfile san_test --num_accepts 18088 --num_acc_cls 12058 --dset SAN --latent_dim 400 --omega 0.1 --dim2 50 --epochs 301 --n_samples 5
