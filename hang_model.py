import torch
import torch.nn as nn
import torch
from torch import nn
from torch import Tensor
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import math


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=True):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class VAE_reset(nn.Module):
    def __init__(self, args):
        super(VAE_reset, self).__init__()

        self.conv_mindim = args['conv_mindim']
        conv_mindim = args['conv_mindim']
        self.args = args
        self.training = True
        self.kl_loss = 0
        self.mse = 0
        self.encoder = nn.Sequential(
            nn.Conv2d(3, conv_mindim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_mindim, conv_mindim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim * 2),
            nn.ReLU(inplace=True),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
        )

        self.fc11 = nn.Linear(conv_mindim * 2 * 8 * 8, conv_mindim * 2 * 8 * 8)
        self.fc12 = nn.Linear(conv_mindim * 2 * 8 * 8, conv_mindim * 2 * 8 * 8)

        self.decoder = nn.Sequential(
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.ConvTranspose2d(conv_mindim * 2, conv_mindim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                conv_mindim, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.conv_mindim * 2 * 8 * 8)
        # if self.args['constant']['encoder_softmax']:
        #     h1 = F.softmax(h1, dim=1)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.conv_mindim * 2, 8, 8)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


    def loss_vae_mse(self, x, recon_x, mu, logvar):
        self.mse = F.mse_loss(recon_x, x, reduction='sum')
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        # self.kl_loss /= batch_size * 3 * 1024
        self.mse /= batch_size
        self.kl_loss /= batch_size
        # return mse
        return self.mse, self.kl_loss

    def latest_losses(self):
        return self.mse, self.kl_loss


class DeepUNet(nn.Module):
    def __init__(self, args, in_channels=3, out_channels=3):
        super(DeepUNet, self).__init__()
        self.code1 = 0
        self.code2 = 0
        self.std = 20

        conv_mindim = args['conv_mindim']
        args['model']['denoise'] = 'DeepUNet'

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, conv_mindim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(conv_mindim, conv_mindim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim * 2),
            nn.ReLU(inplace=True),
        )
        self.encoder3 = nn.Sequential(
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
        )

        self.decoder3 = nn.Sequential(
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(conv_mindim * 4, conv_mindim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim*2),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(
                conv_mindim*2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # self.decoder2 = nn.Sequential(
        #     nn.ConvTranspose2d(conv_mindim * 2, conv_mindim, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(conv_mindim),
        #     nn.ReLU(inplace=True),
        # )
        # self.decoder1 = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         conv_mindim, 3, kernel_size=4, stride=2, padding=1, bias=False),
        # )

    def forward(self, x):
        # 编码器
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        d3 = self.decoder3(x3)
        d2 = self.decoder2(torch.cat((d3, x2), 1))
        d1 = self.decoder1(d2)

        # d2 = self.decoder2(d3+x2)
        # d1 = self.decoder1(d2)

        # compression
        x3 = x3 - torch.randn(x3.size()).cuda() * self.std
        self.code2 = x2 - torch.randn(x2.size()).cuda() * self.std
        self.code1 = torch.sigmoid(x3)

        # 解码器

        return d1

    def loss_mse(self, x, recon_x):
        criterion = nn.MSELoss()
        mse = criterion(x, recon_x)
        return mse

    def loss_binary(self):
        entro1 = (self.code1 ** 2).mean()
        entro2 = (self.code2 ** 2).mean()
        return entro1+entro2


class FreezUNet(nn.Module):
    def __init__(self, args):
        super(FreezUNet, self).__init__()
        self.code1 = 0
        self.code2 = 0
        self.std = 20
        self.freeze_encoder = False

        conv_mindim = args['conv_mindim']
        args['model']['denoise'] = 'FreezUNet'

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, conv_mindim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(conv_mindim, conv_mindim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim * 2),
            nn.ReLU(inplace=True),
        )
        self.encoder3 = nn.Sequential(
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
        )

        self.decoder3 = nn.Sequential(
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(conv_mindim * 4, conv_mindim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim*2),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(
                conv_mindim*2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # self.decoder2 = nn.Sequential(
        #     nn.ConvTranspose2d(conv_mindim * 2, conv_mindim, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(conv_mindim),
        #     nn.ReLU(inplace=True),
        # )
        # self.decoder1 = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         conv_mindim, 3, kernel_size=4, stride=2, padding=1, bias=False),
        # )

    def forward(self, x):
        # 编码器
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        if self.freeze_encoder:
            for param in self.encoder3.parameters():
                param.requires_grad = False
            for param in self.encoder2.parameters():
                param.requires_grad = False
            for param in self.encoder1.parameters():
                param.requires_grad = False

        d3 = self.decoder3(x3)
        d2 = self.decoder2(torch.cat((d3, x2), 1))
        d1 = self.decoder1(d2)

        # d2 = self.decoder2(d3+x2)
        # d1 = self.decoder1(d2)

        # compression
        x3 = x3 - torch.randn(x3.size()).cuda() * self.std
        self.code2 = x2 - torch.randn(x2.size()).cuda() * self.std
        self.code1 = torch.sigmoid(x3)

        # 解码器

        return d1

    def loss_mse(self, x, recon_x):
        criterion = nn.MSELoss()
        mse = criterion(x, recon_x)
        return mse

    def loss_binary(self):
        entro = (self.code1 ** 2).mean()
        entro += (self.code2 ** 2).mean()
        return entro




class AENet(nn.Module):
    def __init__(self, args):
        super(AENet, self).__init__()
        self.code = 0
        self.std = 20
        args['model']['denoise'] = 'AENet'
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            # 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            #             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            #             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            #             nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),  # [batch, 3, 32, 32]
            # nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # x3 = x3 - torch.randn(x3.size()).cuda() * self.std

        # self.code = torch.sigmoid(x3)

        # 解码器

        return decoded

    def loss_mse(self, x, recon_x):
        criterion = nn.MSELoss()
        mse = criterion(x, recon_x)
        return mse

    def loss_binary(self):
        entro = (self.code ** 2).mean()
        return entro



class Image_UNet(nn.Module):
    def __init__(self, args):
        super(Image_UNet, self).__init__()
        self.code1 = 0
        self.code2 = 0
        self.std = 20
        self.freeze_encoder = False

        conv_mindim = args['conv_mindim']
        args['model']['denoise'] = 'FreezUNet'

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, conv_mindim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(conv_mindim, conv_mindim * 2, kernel_size=6, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim * 2),
            nn.ReLU(inplace=True),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(conv_mindim*2, conv_mindim * 2, kernel_size=6, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim * 2),
            nn.ReLU(inplace=True),
        )
        self.encoder4 = nn.Sequential(
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
        )


        self.decoder4 = nn.Sequential(
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(conv_mindim * 4, conv_mindim*2, kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim*2),
            nn.ReLU(inplace=True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(conv_mindim * 2, conv_mindim*2, kernel_size=6, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim*2),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(
                conv_mindim*2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # self.decoder2 = nn.Sequential(
        #     nn.ConvTranspose2d(conv_mindim * 2, conv_mindim, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(conv_mindim),
        #     nn.ReLU(inplace=True),
        # )
        # self.decoder1 = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         conv_mindim, 3, kernel_size=4, stride=2, padding=1, bias=False),
        # )



    def forward(self, x):
        # 编码器
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        if self.freeze_encoder:
            for param in self.encoder4.parameters():
                param.requires_grad = False
            for param in self.encoder3.parameters():
                param.requires_grad = False
            for param in self.encoder2.parameters():
                param.requires_grad = False
            for param in self.encoder1.parameters():
                param.requires_grad = False


        d4 = self.decoder4(x4)
        d3 = self.decoder3(torch.cat((d4, x3), 1))
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)

        # d2 = self.decoder2(d3+x2)
        # d1 = self.decoder1(d2)

        # compression
        x4 = x4 - torch.randn(x3.size()).cuda() * self.std
        self.code2 = x4 - torch.randn(x4.size()).cuda() * self.std
        self.code1 = torch.sigmoid(x4)

        # 解码器

        return d1

    def loss_mse(self, x, recon_x):
        criterion = nn.MSELoss()
        mse = criterion(x, recon_x)
        return mse

    def loss_binary(self):
        entro = (self.code1 ** 2).mean()
        entro += (self.code2 ** 2).mean()
        return entro




class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        delta = torch.max(x) - torch.min(x)
        x = (x/delta + 0.5)
        return x.round() * 2 - 1
    @staticmethod
    def backward(ctx, g):
        return g


class QuantizeUNet(nn.Module):
    def __init__(self, args):
        super(QuantizeUNet, self).__init__()
        self.code1 = 0
        self.code2 = 0
        self.std = 20
        self.freeze_encoder = False

        self.bit = args['quan_bit']
        self.alpha = torch.tensor(args['quan_alpha'])
        conv_mindim = args['conv_mindim']
        args['model']['denoise'] = 'QuantizeUNet'

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, conv_mindim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim),
            nn.ReLU(inplace=True),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(conv_mindim, conv_mindim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim * 2),
            nn.ReLU(inplace=True),
        )
        self.encoder3 = nn.Sequential(
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
        )

        self.decoder3 = nn.Sequential(
            ResBlock(conv_mindim * 2, conv_mindim * 2),
            nn.BatchNorm2d(conv_mindim * 2),
            ResBlock(conv_mindim * 2, conv_mindim * 2),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(conv_mindim * 4, conv_mindim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_mindim*2),
            nn.ReLU(inplace=True),
        )
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(
                conv_mindim*2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

        # self.decoder2 = nn.Sequential(
        #     nn.ConvTranspose2d(conv_mindim * 2, conv_mindim, kernel_size=4, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(conv_mindim),
        #     nn.ReLU(inplace=True),
        # )
        # self.decoder1 = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         conv_mindim, 3, kernel_size=4, stride=2, padding=1, bias=False),
        # )

        # quantize

    def sgn(self, x):
        x = RoundWithGradient.apply(x)

        return x


    def phi_function(self, x, mi, alpha, delta):

        # alpha should less than 2 or log will be None
        # alpha = alpha.clamp(None, 2)
        # alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)
        s = 1/(1-alpha)
        k = torch.log(2/alpha - 1) * (1/delta)
        x = (((x - mi) *k ).tanh()) * s
        return x


    def dequantize(self, x, lower_bound, delta, interval):

        # save mem
        x =  ((x+1)/2 + interval) * delta + lower_bound

        return x


    def forward(self, x):
        # 编码器
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        if self.freeze_encoder:
            for param in self.encoder3.parameters():
                param.requires_grad = False
            for param in self.encoder2.parameters():
                param.requires_grad = False
            for param in self.encoder1.parameters():
                param.requires_grad = False

        cur_max = torch.max(x3)
        cur_min = torch.min(x3)
        delta = (cur_max - cur_min) / (2 ** self.bit - 1)
        interval = (x3 - cur_min) // delta
        mi = (interval + 0.5) * delta + cur_min
        x3 = self.phi_function(x3, mi, self.alpha, delta)
        x3 = self.sgn(x3)
        x3 = self.dequantize(x3, cur_min, delta, interval)


        d3 = self.decoder3(x3)
        d2 = self.decoder2(torch.cat((d3, x2), 1))
        d1 = self.decoder1(d2)

        # d2 = self.decoder2(d3+x2)
        # d1 = self.decoder1(d2)

        # compression
        x3 = x3 - torch.randn(x3.size()).cuda() * self.std
        self.code2 = x2 - torch.randn(x2.size()).cuda() * self.std
        self.code1 = torch.sigmoid(x3)

        # 解码器

        return d1

    def loss_mse(self, x, recon_x):
        criterion = nn.MSELoss()
        mse = criterion(x, recon_x)
        return mse

    def loss_binary(self):
        entro = (self.code1 ** 2).mean()
        entro += (self.code2 ** 2).mean()
        return entro

