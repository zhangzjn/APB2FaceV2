## APB2FaceV2

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic) ![PyTorch 1.5.1](https://img.shields.io/badge/pytorch-1.5.1-green.svg?style=plastic)

Official pytorch implementation of the paper: "[APB2FACEV2: REAL-TIME AUDIO-GUIDED MULTI-FACE REENACTMENT](https://ieeexplore.ieee.org/abstract/document/9552566)".

## Using the Code

### Requirements

This code has been developed under `Python3.7`, `PyTorch 1.5.1` and `CUDA 10.1` on `Ubuntu 16.04`. 

## Datasets in the paper

Download **AnnVI** dataset from 
[Google Drive](https://drive.google.com/file/d/1xEnZwNLU4SmgFFh4WGV4KEOdegfFrOdp/view?usp=sharing) 
or 
[Baidu Cloud](https://pan.baidu.com/s/1oydpePBQieRoDmaENg3kfQ) (Key:str3) to `/media/datasets/AnnVI`.


### Train

```shell
python3 train.py --name AnnVI --data AnnVI --data_root DATASET_PATH --img_size 256 --mode train --trainer l2face --gan_mode lsgan --gpus 0 --batch_size 16
```

Results are stored in `checkpoints/xxx`

### Test

```shell
python3 test.py 
```

Results are stored in `checkpoints/AnnVI-Big/results`

### Citation

```angular2
@article{zhang2021real,
  title={Real-Time Audio-Guided Multi-Face Reenactment},
  author={Zhang, Jiangning and Zeng, Xianfang and Xu, Chao and Liu, Yong and Li, Hongliang},
  journal={IEEE Signal Processing Letters},
  year={2021},
  publisher={IEEE}
}
```
