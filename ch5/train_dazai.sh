#!/bin/sh
python train.py --data_dir data/dazai --checkpoint_dir model/dazai --enable_checkpoint False  --gpu 0 --epochs 50
#python train.py --data_dir data/dazai \
#    --checkpoint_dir model/dazai \
#    --rnn_size 128 --gpu 0 --enable_checkpoint False
#    --checkpoint_dir /media/tajima/New\ Volume/cv/dazai \
#    --init_from "$1"
