from util.options import get_opt
from trainer.Demo_l2face_trainer import Trainer_
import torch


mode = 0
# Same audio / Same idt
if mode == 0:
    from data.Demo_ann import Dataset_

    opt = get_opt()
    opt.data = 'AnnVI'
    opt.data_root = '/media/datasets/zhangzjn/Audio2Face/AnnVI/feature'
    opt.img_size = 256
    opt.resume = True
    opt.resume_name = 'AnnVI-Big'
    opt.logdir = '{}/{}'.format(opt.checkpoint, opt.resume_name)
    opt.resume_epoch = -1
    opt.gpus = [0]
    opt.results_dir = '{}/results/{}'.format(opt.logdir, mode)
    opt.video_repeat_times = 5
    opt.aud_counts = 300
    idts1 = ['man1', 'man2', 'man3', 'woman1', 'woman2', 'woman3']
    idts2 = ['man1', 'man2', 'man3', 'woman1', 'woman2', 'woman3']
    for idx1, idt1 in enumerate(idts1):
        for idx2, idt2 in enumerate(idts2):
            opt.ref_idt = idt1
            opt.aud_idt = idt2
            print('{} --> {} [{}/{}]'.format(idt2, idt1, idx1*len(idts1)+idx2+1, len(idts1) * len(idts1)))
            dataloader = Dataset_(opt)
            dataloader = torch.utils.data.DataLoader(dataloader, batch_size=1, num_workers=1)
            trainer = Trainer_(opt)
            trainer.run(dataloader)
            print()
