#!/usr/bin/env bash
# AnnVI
~/anaconda3/bin/python3 train.py --name AnnVI --data AnnVI --data_root /media/datasets/zhangzjn/AnnVI/feature/ --img_size 256 --mode train --trainer l2face --gan_mode lsgan --gpus 0 --batch_size 16
