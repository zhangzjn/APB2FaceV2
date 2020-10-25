import os
import shutil
import cv2
import numpy as np
import time
import torch
from model.audio_net import AudioNet
from model.NAS_GAN import NAS_GAN
from util.net_util import init_net, print_networks
# from tensorboardX import SummaryWriter


class Trainer_():
    def __init__(self, opt):
        self.opt = opt
        self.video_repeat_times = opt.video_repeat_times
        self.aud_counts = opt.aud_counts
        self.device = torch.device('cuda:{}'.format(opt.gpus[0])) if opt.gpus[0] > -1 else torch.device('cpu')
        # self.writer = SummaryWriter(logdir=opt.logdir, comment='') if opt.mode == 'train' else None
        # audio
        self.netA = AudioNet()
        self.netA.load_state_dict(torch.load('model/pretrained/{}_best_{}.pth'.format(opt.data, opt.img_size),
                                             map_location={'cuda:0': 'cuda:{}'.format(opt.gpus[0])})['audio_net'])
        self.netA.to(self.device)
        # G
        layers = 9
        width_mult_list = [4. / 12, 6. / 12, 8. / 12, 10. / 12, 1.]
        width_mult_list_sh = [4 / 12, 6. / 12, 8. / 12, 10. / 12, 1.]
        state = torch.load('model/NAS_GAN_arch.pt', map_location='cpu')
        self.netG = NAS_GAN(state['alpha'], state['ratio'], state['ratio_sh'], layers=layers, width_mult_list=width_mult_list, width_mult_list_sh=width_mult_list_sh)
        self.netG = init_net(self.netG, opt.init_type, opt.gpus)
        print_networks(self.netG)
        if opt.resume:
            checkpoint = torch.load('{}/{}_{}_G.pth'.format(opt.logdir, opt.resume_epoch if opt.resume_epoch > -1 else 'latest', opt.img_size), map_location=self.device)
            self.netG.load_state_dict(checkpoint['netG'])
        self.netG.eval()

    def run(self, dataloader):
        results_dir = self.opt.results_dir
        aud_idt = self.opt.aud_idt
        ref_idt = self.opt.ref_idt
        img_size = self.opt.img_size
        save_dir = '{}/{}-{}'.format(results_dir, aud_idt, ref_idt)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        imgs_dir = '{}/imgs'.format(save_dir)
        imgs_dir_ref = '{}/imgs_ref'.format(save_dir)
        imgs_dir_aud = '{}/imgs_aud'.format(save_dir)
        os.mkdir(imgs_dir)
        os.mkdir(imgs_dir_ref)
        os.mkdir(imgs_dir_aud)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter('{}/{}-{}.avi'.format(save_dir, aud_idt, ref_idt), fourcc, 25.0, (img_size * 2, img_size))
        video_ref = cv2.VideoWriter('{}/{}-{}-ref.avi'.format(save_dir, aud_idt, ref_idt), fourcc, 25.0, (img_size, img_size))
        video_aud = cv2.VideoWriter('{}/{}-{}-aud.avi'.format(save_dir, aud_idt, ref_idt), fourcc, 25.0, (img_size, img_size))
        for batch_idx, test_data in enumerate(dataloader):
            self.set_input(test_data)
            self.forward()
            img1_fake = self.img1_fake.data[0].cpu().numpy()
            img1_real = self.img1.data[0].cpu().numpy()
            img1_fake_numpy = (np.transpose(img1_fake, (1, 2, 0)) + 1) / 2.0 * 255.0
            img1_real_numpy = (np.transpose(img1_real, (1, 2, 0)) + 1) / 2.0 * 255.0
            img1_fake_numpy = cv2.cvtColor(img1_fake_numpy, cv2.COLOR_BGR2RGB)
            img1_real_numpy = cv2.cvtColor(img1_real_numpy, cv2.COLOR_BGR2RGB)
            img1_fake_numpy = img1_fake_numpy.astype(np.uint8)
            img1_real_numpy = img1_real_numpy.astype(np.uint8)
            img_cat = np.concatenate([img1_fake_numpy, img1_real_numpy], axis=1)
            # img_out = np.concatenate([img1_fake_numpy, img1_real_numpy], axis=1)
            cv2.imwrite('{}/{}.jpg'.format(imgs_dir, batch_idx), img_cat)
            cv2.imwrite('{}/{}.jpg'.format(imgs_dir_ref, batch_idx), img1_fake_numpy)
            cv2.imwrite('{}/{}.jpg'.format(imgs_dir_aud, batch_idx), img1_real_numpy)
            for _ in range(self.video_repeat_times):  # five times slower
                video.write(img_cat)
                video_ref.write(img1_fake_numpy)
                video_aud.write(img1_real_numpy)
            print('\r{}/{}'.format(batch_idx+1, len(dataloader)), end='')
            if batch_idx == self.aud_counts - 1:
                break
        video.release()

    def run_decoupling_pose(self, dataloader):
        results_dir = self.opt.results_dir
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)
        img_size = self.opt.img_size

        for batch_idx, test_data in enumerate(dataloader):
            pose_names = ['pitch', 'roll', 'yaw']
            for pose_cnt, pose_name in enumerate(pose_names):
                save_dir = '{}/{}/{}'.format(results_dir, pose_name, batch_idx)
                os.makedirs(save_dir, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video = cv2.VideoWriter('{}/{}.avi'.format(save_dir, batch_idx), fourcc, 25.0, (img_size, img_size))
                self.set_input(test_data)
                thr = 60
                angles = list(range(-thr, thr, 2))
                for idx, angle in enumerate(angles):
                    self.pose1[0][pose_cnt] = angle / 100
                    self.forward()
                    img1_fake = self.img1_fake.data[0].cpu().numpy()
                    img1_fake_numpy = (np.transpose(img1_fake, (1, 2, 0)) + 1) / 2.0 * 255.0
                    img1_fake_numpy = cv2.cvtColor(img1_fake_numpy, cv2.COLOR_BGR2RGB)
                    img1_fake_numpy = img1_fake_numpy.astype(np.uint8)
                    cv2.imwrite('{}/{}.jpg'.format(save_dir, idx), img1_fake_numpy)
                    for _ in range(self.video_repeat_times):  # five times slower
                        video.write(img1_fake_numpy)
                    print('\rBatch: {}/{} | Pose: {}/{} | Angle: {}/{}'.format(
                        batch_idx+1, len(dataloader), pose_cnt + 1, len(pose_names), idx + 1, len(angles)), end='')
                video.release()
            if batch_idx == self.aud_counts - 1:
                break

    def run_decoupling_eye(self, dataloader):
        results_dir = self.opt.results_dir
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir, exist_ok=True)
        img_size = self.opt.img_size

        for batch_idx, test_data in enumerate(dataloader):
            eye_names = ['l', 'r', 'lr']
            for eye_cnt, eye_name in enumerate(eye_names):
                save_dir = '{}/{}/{}'.format(results_dir, eye_name, batch_idx)
                os.makedirs(save_dir, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video = cv2.VideoWriter('{}/{}.avi'.format(save_dir, batch_idx), fourcc, 25.0, (img_size, img_size))
                self.set_input(test_data)
                thr = 100
                angles = list(range(-thr, thr, 2))
                for idx, angle in enumerate(angles):
                    if 'l' in eye_name:
                        self.eye1[0][0] = angle / 100
                    if 'r' in eye_name:
                        self.eye1[0][1] = angle / 100
                    self.forward()
                    img1_fake = self.img1_fake.data[0].cpu().numpy()
                    img1_fake_numpy = (np.transpose(img1_fake, (1, 2, 0)) + 1) / 2.0 * 255.0
                    img1_fake_numpy = cv2.cvtColor(img1_fake_numpy, cv2.COLOR_BGR2RGB)
                    img1_fake_numpy = img1_fake_numpy.astype(np.uint8)
                    cv2.imwrite('{}/{}.jpg'.format(save_dir, idx), img1_fake_numpy)
                    for _ in range(self.video_repeat_times):  # five times slower
                        video.write(img1_fake_numpy)
                    print('\rBatch: {}/{} | Pose: {}/{} | Angle: {}/{}'.format(
                        batch_idx+1, len(dataloader), eye_cnt + 1, len(eye_names), idx + 1, len(angles)), end='')
                video.release()
            if batch_idx == self.aud_counts - 1:
                break

    def set_input(self, input):
        aud_feat1, pose1, eye1, img1, img2 = input
        self.aud_feat1 = aud_feat1.to(self.device)
        self.pose1 = pose1.to(self.device)
        self.eye1 = eye1.to(self.device)
        self.img1 = img1.to(self.device)
        self.img2 = img2.to(self.device)

    def forward(self):
        latent, landmark = self.netA(self.aud_feat1, self.pose1, self.eye1)
        self.img1_fake = self.netG(self.img2, latent)
