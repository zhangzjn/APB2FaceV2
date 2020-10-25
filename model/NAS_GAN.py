import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ======> Basic Ops
PRIMITIVES = [
    'skip',
    'conv3x3',
    'residual',
    'dwsblock',
]


class SkipConnect(nn.Module):

    def __init__(self, C_in, C_out, stride=1):
        super(SkipConnect, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.InstanceNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if hasattr(self, 'conv'):
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
        else:
            out = x
        return out


class Conv3x3(nn.Module):

    def __init__(self, C_in, C_out, stride=1, dilation=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.bn1 = nn.InstanceNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class BasicResidual(nn.Module):

    def __init__(self, C_in, C_out, stride=1, dilation=1, groups=1):
        super(BasicResidual, self).__init__()

        self.conv1 = nn.Conv2d(C_in, C_out, 3, stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.bn1 = nn.InstanceNorm2d(C_out)
        self.conv2 = nn.Conv2d(C_out, C_out, 3, 1, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.bn2 = nn.InstanceNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)

        if C_in != C_out or stride != 1:
            self.skip = nn.Conv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False)
            self.bn3 = nn.InstanceNorm2d(C_out)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, 'skip'):
            identity = self.bn3(self.skip(identity))

        out += identity
        out = self.relu(out)

        return out


class DwsBlock(nn.Module):

    def __init__(self, C_in, C_out, stride=1, dilation=1, groups=1):
        super(DwsBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_in * 4, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False)
        self.bn1 = nn.InstanceNorm2d(C_in * 4)
        self.conv2 = nn.Conv2d(C_in * 4, C_in * 4, 3, stride, padding=dilation, dilation=dilation, groups=C_in * 4,
                               bias=False)
        self.bn2 = nn.InstanceNorm2d(C_in * 4)
        self.conv3 = nn.Conv2d(C_in * 4, C_out, 1, 1, padding=0, dilation=dilation, groups=groups, bias=False)
        self.bn3 = nn.InstanceNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)
        if C_in != C_out or stride != 1:
            self.skip = nn.Conv2d(C_in, C_out, 1, stride, padding=0, dilation=dilation, groups=1, bias=False)
            self.bn4 = nn.InstanceNorm2d(C_out)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if hasattr(self, 'skip'):
            identity = self.bn4(self.skip(identity))

        out += identity
        out = self.relu(out)

        return out


OPS = {
    'skip': lambda C_in, C_out, stride, width_mult_list: SkipConnect(C_in, C_out, stride),
    'conv3x3': lambda C_in, C_out, stride, width_mult_list: Conv3x3(C_in, C_out, stride=stride, dilation=1),
    'conv3x3_d2': lambda C_in, C_out, stride, width_mult_list: Conv3x3(C_in, C_out, stride=stride, dilation=2),
    'conv3x3_d4': lambda C_in, C_out, stride, width_mult_list: Conv3x3(C_in, C_out, stride=stride, dilation=4),
    'residual': lambda C_in, C_out, stride, width_mult_list: BasicResidual(C_in, C_out, stride=stride, dilation=1),
    'dwsblock': lambda C_in, C_out, stride, width_mult_list: DwsBlock(C_in, C_out, stride=stride, dilation=1),
}


class ConvNorm(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False,
                 width_mult_list=[1.]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias

        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation,
                              groups=self.groups, bias=bias)
        self.bn = nn.InstanceNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        super(Conv, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation,
                              groups=self.groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvTranspose2dNorm(nn.Module):

    def __init__(self, C_in, C_out, kernel_size=3, stride=2, dilation=1, groups=1, bias=False):
        super(ConvTranspose2dNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        self.padding = 1
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.conv = nn.ConvTranspose2d(C_in, C_out, kernel_size, stride, padding=self.padding, output_padding=1,
                                       dilation=dilation, groups=self.groups, bias=bias)
        self.bn = nn.InstanceNorm2d(C_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def make_divisible(v, divisor=8, min_value=3):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, op_idx, stride=1):
        super(MixedOp, self).__init__()
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, stride, width_mult_list=[1.])

    def forward(self, x):
        return self._op(x)


class SingleOp(nn.Module):

    def __init__(self, op, C_in, C_out, kernel_size=3, stride=1):
        super(SingleOp, self).__init__()
        self._op = op(C_in, C_out, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        result = self._op(x)

        return result

# ======> Adaptive Convolution
class AdaConv(nn.Module):

    def __init__(self, inp=256, outp=256, groups=True, bias=True):
        super(AdaConv, self).__init__()
        self.inp = inp
        self.outp = outp
        self.groups = groups
        self.bias = bias
        outp_weight = outp if groups else outp ** 2
        # conv_params_predictor
        # self.shared_conv = nn.Sequential(nn.Conv2d(inp, inp, kernel_size=1), nn.ReLU(True))
        self.shared_conv = nn.Sequential()
        self.weight_conv = nn.Sequential(nn.Conv2d(inp, inp, kernel_size=1), nn.ReLU(True),
                                         nn.Conv2d(inp, outp_weight, kernel_size=1))
        if bias:
            self.bias_conv = nn.Sequential(nn.Conv2d(inp, inp, kernel_size=1), nn.ReLU(True),
                                           nn.Conv2d(inp, outp, kernel_size=1))

    def forward(self, x, latent):
        latent_ = self.shared_conv(latent)
        weights = self.weight_conv(latent_)
        bias = self.bias_conv(latent_) if self.bias else None
        weights = weights.view(weights.shape[0], self.outp, 1, 1, 1) if self.groups else weights.view(weights.shape[0], self.outp, self.outp, 1, 1)
        bias = bias.view(bias.shape[0], self.outp) if self.bias else None
        out = [F.conv2d(x[i].unsqueeze(0), weights[i], bias[i], groups=self.outp if self.groups else 1) for i in range(x.shape[0])]
        return torch.cat(out, dim=0)


class NAS_GAN(nn.Module):
    def __init__(self, alpha, ratio, ratio_sh, layers, width_mult_list, width_mult_list_sh):
        super(NAS_GAN, self).__init__()
        assert layers >= 3
        self.layers = layers
        self.len_stem = 3
        self.len_header = 3
        op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)
        ratio_list = F.softmax(ratio, dim=-1).argmax(-1)
        ratio_list_sh = F.softmax(ratio_sh, dim=-1).argmax(-1)

        # Construct Stem
        self.stem = nn.ModuleList()
        self.stem.append(SingleOp(ConvNorm, 3, make_divisible(64 * width_mult_list_sh[ratio_list_sh[0]]), 7))
        in_features = 64
        out_features = in_features * 2
        for i in range(2):
            self.stem.append(SingleOp(ConvNorm, make_divisible(in_features * width_mult_list_sh[ratio_list_sh[i]]),
                                      make_divisible(out_features * width_mult_list_sh[ratio_list_sh[i + 1]]),
                                      3, stride=2))
            in_features = out_features
            out_features = in_features * 2
        # Construct Blocks
        self.cells = nn.ModuleList()
        self.ada_convs = nn.ModuleList()
        group, bias = True, True
        for i in range(layers):
            cha = 152
            if i == 0:
                self.cells.append(MixedOp(88, cha, op_idx_list[i]))
                self.ada_convs.append(AdaConv(256, cha, group, bias))
            elif i == layers - 1:
                self.cells.append(MixedOp(cha, 128, op_idx_list[i]))
                self.ada_convs.append(AdaConv(256, 128, group, bias))
            else:
                self.cells.append(MixedOp(cha, cha, op_idx_list[i]))
                self.ada_convs.append(AdaConv(256, cha, group, bias))

        # Construct Header
        self.header = nn.ModuleList()
        out_features = in_features // 2
        self.header.append(SingleOp(ConvTranspose2dNorm,
                                    make_divisible(in_features * width_mult_list[ratio_list[self.layers - 1]]),
                                    make_divisible(out_features * width_mult_list_sh[ratio_list_sh[self.len_stem]]),
                                    3, stride=2))
        in_features = out_features
        out_features = in_features // 2
        self.header.append(SingleOp(ConvTranspose2dNorm,
                                    make_divisible(in_features * width_mult_list_sh[ratio_list_sh[self.len_stem]]),
                                    make_divisible(out_features * width_mult_list_sh[ratio_list_sh[self.len_stem + 1]]),
                                    3, stride=2))
        self.header.append(SingleOp(Conv, make_divisible(64 * width_mult_list_sh[ratio_list_sh[self.len_stem + 1]]),
                                    3, 7))
        self.tanh = nn.Tanh()

    def forward(self, input, latent):
        out = input
        for i, module in enumerate(self.stem):
            out = module(out)

        latent = latent.unsqueeze(2).unsqueeze(3)
        for i, cell in enumerate(self.cells):
            out = cell(out)
            out = self.ada_convs[i](out, latent)

        for i, module in enumerate(self.header):
            out = module(out)

        out = self.tanh(out)

        return out


if __name__ == '__main__':
    import time

    def print_networks(net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('[Network G] Total number of parameters : {:.3f} M'.format(num_params / 1e6))

    layers = 9
    width_mult_list = [4. / 12, 6. / 12, 8. / 12, 10. / 12, 1.]
    width_mult_list_sh = [4 / 12, 6. / 12, 8. / 12, 10. / 12, 1.]
    state = torch.load('NAS_GAN_arch.pt', map_location='cpu')
    net = NAS_GAN(state['alpha'], state['ratio'], state['ratio_sh'], layers=layers,
                    width_mult_list=width_mult_list, width_mult_list_sh=width_mult_list_sh)
    inputs = torch.randn(8, 3, 256, 256)
    latent = torch.randn(8, 256)
    print(net(inputs, latent).shape)
    print_networks(net)

    inputs = torch.randn(1, 3, 128, 128)
    latent = torch.randn(1, 256)
    out = net(inputs, latent)
    repeats = 100
    start = time.time()
    for i in range(repeats):
        out = net(inputs, latent)
        print('\r{}/{}'.format(i + 1, repeats), end='')
    end = time.time()
    print('[CPU] time: {:.3f}'.format(end - start))

    torch.cuda.set_device(1)
    inputs = torch.randn(1, 3, 128, 128).cuda()
    latent = torch.randn(1, 256).cuda()
    net = net.cuda()
    out = net(inputs, latent)
    repeats = 1000
    start = time.time()
    for i in range(repeats):
        out = net(inputs, latent)
        print('\r{}/{}'.format(i + 1, repeats), end='')
    end = time.time()
    print('[GPU] time: {:.3f}'.format(end - start))
