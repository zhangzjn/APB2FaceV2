import cv2
import numpy as np
import time

import torch
from model.audio_net import AudioNet
from model.NAS_GAN import NAS_GAN
from model.patchgan_dis import NLayerDiscriminator
from loss.ganloss import GANLoss
from util.net_util import init_net, get_scheduler, print_networks
from tensorboardX import SummaryWriter


class Trainer_():
    def __init__(self, opt, logger):
        self.opt = opt
        self.logger = logger
        self.isTrain = True if opt.mode == 'train' else False
        self.device = torch.device('cuda:{}'.format(opt.gpus[0])) if opt.gpus[0] > -1 else torch.device('cpu')
        self.epoch = 0
        self.iters = 0
        self.GpD = 1
        self.GpD_iters = 0
        self.writer = SummaryWriter(logdir=opt.logdir, comment='') if opt.mode == 'train' else None
        # audio
        self.netA = AudioNet()
        self.netA.load_state_dict(torch.load('model/pretrained/{}_best_{}.pth'.format(opt.data, opt.img_size),
                                             map_location={'cuda:0': 'cuda:{}'.format(opt.gpus[0])})['audio_net'])
        self.netA.to(self.device)
        # G
        # self.netG = UNet_D4XS()
        # self.netG = Resnet_kernel(n_blocks=6)
        layers = 9
        width_mult_list = [4. / 12, 6. / 12, 8. / 12, 10. / 12, 1.]
        width_mult_list_sh = [4 / 12, 6. / 12, 8. / 12, 10. / 12, 1.]
        state = torch.load('model/NAS_GAN_arch.pt', map_location='cpu')
        self.netG = NAS_GAN(state['alpha'], state['ratio'], state['ratio_sh'], layers=layers, width_mult_list=width_mult_list, width_mult_list_sh=width_mult_list_sh)
        self.netG = init_net(self.netG, opt.init_type, opt.gpus)
        print_networks(self.netG)
        if opt.resume:
            # checkpoint = torch.load('{}/{}_{}_G.pth'.format(opt.logdir, opt.resume_epoch if opt.resume_epoch > -1 else 'latest', opt.img_size), map_location={'cuda:0': 'cuda:{}'.format(opt.gpus[0])})
            checkpoint = torch.load('{}/{}_{}_G.pth'.format(opt.logdir, opt.resume_epoch if opt.resume_epoch > -1 else 'latest', opt.img_size), map_location=self.device)
            self.netG.load_state_dict(checkpoint['netG'])
            self.epoch = checkpoint['epoch']
        # D
        if self.isTrain:
            self.netD = NLayerDiscriminator(ndf=64)
            self.netD = init_net(self.netD, opt.init_type, opt.gpus)
            if opt.resume:
                checkpoint = torch.load('{}/{}_{}_D.pth'.format(opt.logdir, opt.resume_epoch if opt.resume_epoch > -1 else 'latest', opt.img_size))
                self.netD.load_state_dict(checkpoint['netD'])
        if self.isTrain:
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.init_optim()

    def init_optim(self):
        if self.opt.optim == 'Adam':
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        elif self.opt.optim == 'SGD':
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=self.opt.lr * 100, momentum=0.9)
            self.optimizer_D = torch.optim.SGD(self.netD.parameters(), lr=self.opt.lr * 100, momentum=0.9)
        elif self.opt.optim == 'RMSprop':
            self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=self.opt.lr * 100)
            self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(), lr=self.opt.lr * 100)
        self.scheduler_G = get_scheduler(self.optimizer_G, self.opt)
        self.scheduler_D = get_scheduler(self.optimizer_D, self.opt)

    def reset(self):
        if self.isTrain:
            self.netA.eval()
            self.netG.train()
            self.netD.train()
        else:
            self.netA.eval()
            self.netG.eval()
        self.loss_log_L1 = 0
        self.loss_log_G = 0
        self.loss_log_D_R = 0
        self.loss_log_D_F = 0

    def run(self, dataloader, epoch=None):
        if self.isTrain:  # train
            self.epoch += 1
            self.reset()
            self.scheduler_G.step()
            self.scheduler_D.step()
            for batch_idx, train_data in enumerate(dataloader):
                self.iters += 1
                self.set_input(train_data)
                self.optimize_parameters()
                self.writer.add_scalar('L1', self.loss_log_L1 / (batch_idx + 1), self.iters)
                if self.iters % self.opt.show_iters == 0:
                    self.writer.add_images('Real_Image', (self.img1 + 1) / 2, self.iters)
                    self.writer.add_images('Fake_Image', (self.img1_fake + 1) / 2, self.iters)
                    self.writer.flush()
                log_string = 'train --> '
                log_string += '[epoch {} | '.format(self.epoch)
                log_string += 'iters {}] '.format(self.iters)
                log_string += 'batch {}/{} '.format(batch_idx + 1, len(dataloader))
                log_string += '|loss_L1 {:.5f}'.format(self.loss_log_L1 / (batch_idx + 1))
                log_string += '|loss_G {:.5f}'.format(self.loss_log_G / (batch_idx + 1))
                log_string += '|loss_D_R {:.5f}'.format(self.loss_log_D_R / (batch_idx + 1))
                log_string += '|loss_D_F {:.5f}'.format(self.loss_log_D_F / (batch_idx + 1))
                print('\r'+log_string, end='')
            print('\r', end='')
            self.logger.info(log_string)
            # checkpoint
            self.save('latest')
            if self.epoch % self.opt.record_every == 0:
                self.save()
        else:  # test
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video = cv2.VideoWriter('{}/{}_{}_{}.avi'.format(self.opt.logdir, self.opt.idt_test, self.opt.img_size, time.strftime("%Y%m%d-%H%M%S")), fourcc, 25.0,
                                  (self.opt.img_size * 2, self.opt.img_size))

            self.reset()
            for batch_idx, test_data in enumerate(dataloader):
                self.set_input(test_data)
                self.forward()
                img1_fake = self.img1_fake.data[0].cpu().numpy()
                img1_real = self.img1.data[0].cpu().numpy()
                img1_fake_numpy = (np.transpose(img1_fake, (1, 2, 0)) + 1) / 2.0 * 255.0
                img1_real_numpy = (np.transpose(img1_real, (1, 2, 0)) + 1) / 2.0 * 255.0
                # img1_fake_numpy = cv2.cvtColor(img1_fake_numpy, cv2.COLOR_RGB2BGR)
                # img1_real_numpy = cv2.cvtColor(img1_real_numpy, cv2.COLOR_RGB2BGR)
                img1_fake_numpy = img1_fake_numpy.astype(np.uint8)
                img1_real_numpy = img1_real_numpy.astype(np.uint8)
                # cv2.imwrite('test.jpg', img1_fake_numpy)
                img_out = np.concatenate([img1_fake_numpy, img1_real_numpy], axis=1)
                for _ in range(5):  # five times slower
                    video.write(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
                print('\r{}/{}'.format(batch_idx+1, len(dataloader)), end='')
                if batch_idx == 300 - 1:
                    break
            video.release()

    def set_input(self, input):
        aud_feat1, pose1, eye1, img1, img2 = input
        self.aud_feat1 = aud_feat1.to(self.device)
        self.pose1 = pose1.to(self.device)
        self.eye1 = eye1.to(self.device)
        self.img1 = img1.to(self.device)
        self.img2 = img2.to(self.device)

    def forward(self):
        latent, landmark = self.netA(self.aud_feat1, self.pose1, self.eye1)
        # self.img1_fake, self.img1_inter = self.netG(self.img2, latent)
        self.img1_fake = self.netG(self.img2, latent)

    def backward_D(self):
        lambda_D = 1
        fake_12 = torch.cat((self.img2, self.img1_fake), 1)
        pred_fake = self.netD(fake_12.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_12 = torch.cat((self.img2, self.img1), 1)
        pred_real = self.netD(real_12)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * lambda_D
        self.loss_D.backward()
        self.loss_log_D_F += self.loss_D_fake.item()
        self.loss_log_D_R += self.loss_D_real.item()

    def backward_G(self):
        lambda_GAN = 1
        lambda_L1 = 100
        fake_AB = torch.cat((self.img2, self.img1_fake), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * lambda_GAN
        self.loss_G_L1 = self.criterionL1(self.img1_fake, self.img1) * lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        self.loss_log_G += self.loss_G_GAN.item()
        self.loss_log_L1 += self.loss_G_L1.item()

    def optimize_parameters(self):
        self.forward()
        if self.GpD_iters == 0:
            # update D
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.GpD_iters += 1
        self.GpD_iters = self.GpD_iters % self.GpD

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save(self, mode=None):
        state_G = {
            'netA': self.netA.state_dict(),
            'netG': self.netG.state_dict(),
            'epoch': self.epoch,
        }
        state_D = {
            'netD': self.netD.state_dict()
        }
        torch.save(state_G, '{}/{}_{}_G.pth'.format(self.opt.logdir, mode if mode else self.epoch, self.opt.img_size))
        torch.save(state_D, '{}/{}_{}_D.pth'.format(self.opt.logdir, mode if mode else self.epoch, self.opt.img_size))