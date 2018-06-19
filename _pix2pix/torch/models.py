import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorUNet(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):
        """ Generator model

        Args:
            input_nc: input image dimension
            output_nc: output image dimension
            ngf: the number of filters
        """
        super().__init__()
        self.enc1 = encoder_layer(input_nc, ngf, activation=False, batchnorm=False)
        self.enc2 = encoder_layer(ngf, ngf * 2)
        self.enc3 = encoder_layer(ngf * 2, ngf * 4)
        self.enc4 = encoder_layer(ngf * 4, ngf * 8)
        self.enc5 = encoder_layer(ngf * 8, ngf * 8)
        self.enc6 = encoder_layer(ngf * 8, ngf * 8)
        self.enc7 = encoder_layer(ngf * 8, ngf * 8)
        self.enc8 = encoder_layer(ngf * 8, ngf * 8, batchnorm=False)
        self.dec1 = decoder_layer(ngf * 8, ngf * 8, dropout=True)
        self.dec2 = decoder_layer(ngf * 8 * 2, ngf * 8, dropout=True)
        self.dec3 = decoder_layer(ngf * 8 * 2, ngf * 8, dropout=True)
        self.dec4 = decoder_layer(ngf * 8 * 2, ngf * 8)
        self.dec5 = decoder_layer(ngf * 8 * 2, ngf * 4)
        self.dec6 = decoder_layer(ngf * 4 * 2, ngf * 2)
        self.dec7 = decoder_layer(ngf * 2 * 2, ngf)
        self.dec8 = decoder_layer(ngf * 2, output_nc, batchnorm=False)
        self.apply(weights_init)

    def forward(self, x):
        ''' Encoder '''
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        ''' Decoder '''
        d1 = torch.cat((self.dec1(e8), e7), 1)
        d2 = torch.cat((self.dec2(d1), e6), 1)
        d3 = torch.cat((self.dec3(d2), e5), 1)
        d4 = torch.cat((self.dec4(d3), e4), 1)
        d5 = torch.cat((self.dec5(d4), e3), 1)
        d6 = torch.cat((self.dec6(d5), e2), 1)
        d7 = torch.cat((self.dec7(d6), e1), 1)
        d8 = self.dec8(d7)

        output = F.tanh(d8)
        return output


class Discriminator(nn.Module):

    def __init__(self, input_nc, output_nc, ndf):
        """ Discriminator model

        Args:
            input_nc: input image dimension
            output_nc: output image dimension
            ngf: the number of filters
        """
        super().__init__()
        std_layer = encoder_layer
        self.model = nn.Sequential(
            std_layer(input_nc + output_nc, ndf, activation=False, batchnorm=False),
            std_layer(ndf, ndf * 2),
            std_layer(ndf * 2, ndf * 4),
            std_layer(ndf * 4, ndf * 8, stride=1),
            std_layer(ndf * 8, 1, stride=1),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


def seqence_layer(*compnents):
    return nn.Sequential(*[c for c in compnents if c is not None])


def encoder_layer(in_dim, out_dim, stride=2,
                  activation=True, batchnorm=True):
    return seqence_layer(
            nn.LeakyReLU(0.2, True) if activation else None,
            nn.Conv2d(in_dim, out_dim, 4, stride, 1),
            nn.BatchNorm2d(out_dim) if batchnorm else None,
        )


def decoder_layer(in_dim, out_dim, stride=2,
                  activation=True, batchnorm=True, dropout=False):
    return seqence_layer(
            nn.ReLU(True) if activation else None,
            nn.ConvTranspose2d(in_dim, out_dim, 4, stride, 1),
            nn.BatchNorm2d(out_dim) if batchnorm else None,
            nn.Dropout(0.5) if dropout else None
        )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
