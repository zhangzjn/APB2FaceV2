## APB2FaceV2

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic) ![PyTorch 1.5.1](https://img.shields.io/badge/pytorch-1.5.1-green.svg?style=plastic) ![License MIT](https://img.shields.io/github/license/zhangzjn/APB2FaceV2)

Official pytorch implementation of the paper that is under review for ICASSP'21: "[APB2FACEV2: REAL-TIME AUDIO-GUIDED MULTI-FACE REENACTMENT](https://arxiv.org/abs/2010.13017)".

## Using the Code

### Requirements

This code has been developed under `Python3.7`, `PyTorch 1.5.1` and `CUDA 10.1` on `Ubuntu 16.04`. 

## Datasets in the paper

Download **AnnVI** dataset from 
[Google Drive](https://drive.google.com/file/d/1xEnZwNLU4SmgFFh4WGV4KEOdegfFrOdp/view?usp=sharing) 
or 
[Baidu Cloud](https://pan.baidu.com/s/1oydpePBQieRoDmaENg3kfQ) (Key:str3) to `/media/datasets/zhangzjn/Audio2Face/AnnVI`.

### Test

```shell
python3 test.py 
```

You can view results in `checkpoints/AnnVI-Big/results`

### Train

The complete training code will be released as soon as the paper is accepted.

### Citation

TBA

### Acknowledgements

TBA
