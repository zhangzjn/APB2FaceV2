## APB2FaceV2

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic) ![PyTorch 1.5.1](https://img.shields.io/badge/pytorch-1.5.1-green.svg?style=plastic)

Official pytorch implementation of the paper that is under review for ICASSP'21: "[APB2FACEV2: REAL-TIME AUDIO-GUIDED MULTI-FACE REENACTMENT](https://arxiv.org/pdf/2004.14569.pdf)".

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
@article{zhang2020apb2facev2,
  title={APB2FaceV2: Real-Time Audio-Guided Multi-Face Reenactment},
  author={Zhang, Jiangning and Zeng, Xianfang and Xu, Chao and Chen, Jun and Liu, Yong and Jiang, Yunliang},
  journal={arXiv preprint arXiv:2010.13017},
  year={2020}
}
```
