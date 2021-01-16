import importlib
import torch


def get_dataset(opt):
    if opt.data in ['AnnVI', 'AnnXXX', 'Demo_wild']:
        if 'img' in opt.trainer:
            dataset_filename = 'data.ann_img_dataset'
        else:
            dataset_filename = 'data.ann_dataset'
    elif opt.data in ['VoxCeleb2']:
        if 'img' in opt.trainer:
            dataset_filename = 'data.voxceleb2_img_dataset'
        else:
            dataset_filename = 'data.voxceleb2_dataset'
    elif opt.data in ['Demo_comparison']:
        dataset_filename = 'data.Demo_comparison_dataset'
    datasetlib = importlib.import_module(dataset_filename)
    dataset = datasetlib.Dataset_
    dataset_res = dataset(opt)
    if opt.mode == 'train':
        dataloader_res = torch.utils.data.DataLoader(dataset_res, batch_size=opt.batch_size, num_workers=opt.worker_number)
    elif opt.mode == 'test':
        dataloader_res = torch.utils.data.DataLoader(dataset_res, batch_size=1, num_workers=1)
    return dataloader_res
