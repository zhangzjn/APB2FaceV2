import math
import torch
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc=3+3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_bias=False):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
                                    nn.LeakyReLU(negative_slope=0.2))
        self.skip = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, padding=0),
                                    nn.LeakyReLU(negative_slope=0.2))


    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class StyleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super(StyleDiscriminator, self).__init__()
        convs = []
        convs.append(ResBlock(input_nc, ndf // 2))
        convs.append(ResBlock(ndf // 2, ndf * 1))
        convs.append(ResBlock(ndf * 1, ndf * 2))
        convs.append(ResBlock(ndf * 2, ndf * 4))
        convs.append(ResBlock(ndf * 4, ndf * 8))
        convs.append(ResBlock(ndf * 8, ndf * 8))

        self.convs = nn.Sequential(*convs)

        self.linears = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Linear(ndf * 8, 1))

    def forward(self, input):
        x = self.convs(input)
        batch, channel, height, width = x.shape
        x = x.view(batch, -1)
        out = self.linears(x)
        return out
