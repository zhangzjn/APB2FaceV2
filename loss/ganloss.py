import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        loss = 0
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

    def cal_gradient_penalty(self, netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
        if lambda_gp > 0.0:
            if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
                interpolatesv = real_data
            elif type == 'fake':
                interpolatesv = fake_data
            elif type == 'mixed':
                alpha = torch.rand(real_data.shape[0], 1)
                alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                    *real_data.shape)
                alpha = alpha.to(device)
                interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
            else:
                raise NotImplementedError('{} not implemented'.format(type))
            interpolatesv.requires_grad_(True)
            disc_interpolates = netD(interpolatesv)
            gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)
            gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
            gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
            return gradient_penalty, gradients
        else:
            return 0.0, None