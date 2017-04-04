import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):
        """ Generator model

        Args:
            input_nc: input image dimension
            output_nc: output image dimension
            ngf: the number of filters
        """
        super().__init__()

    def _build_encoder_layers(self, input_nc, ngf):
        self.encoder = [
            encoder_layer(input_nc, ngf, activation=False, batchnorm=False),
            encoder_layer(ngf, ngf * 2),
            encoder_layer(ngf * 2, ngf * 4),
            encoder_layer(ngf * 4, ngf * 8),
            encoder_layer(ngf * 8, ngf * 8),
            encoder_layer(ngf * 8, ngf * 8),
            encoder_layer(ngf * 8, ngf * 8),
            encoder_layer(ngf * 8, ngf * 8, batchnorm=False)]
        for i, layer in enumerate(self.encoder, 1):
            setattr(self, 'enc%d' % i, layer)

    def _build_decoder_layers(self, output_nc, ngf):
        self.decoder = [
            decoder_layer(ngf * 8, ngf * 8, dropout=True),
            decoder_layer(ngf * 8, ngf * 8, dropout=True),
            decoder_layer(ngf * 8, ngf * 8, dropout=True),
            decoder_layer(ngf * 8, ngf * 8),
            decoder_layer(ngf * 8, ngf * 4),
            decoder_layer(ngf * 4, ngf * 2),
            decoder_layer(ngf * 2, ngf),
            decoder_layer(ngf, output_nc, batchnorm=False)]
        for i, layer in enumerate(self.decoder, 1):
            setattr(self, 'dec%d' % i, layer)

    def _build_unet_decoder_layers(self, output_nc, ngf):
        self.decoder = [
            decoder_layer(ngf * 8, ngf * 8, dropout=True),
            decoder_layer(ngf * 8 * 2, ngf * 8, dropout=True),
            decoder_layer(ngf * 8 * 2, ngf * 8, dropout=True),
            decoder_layer(ngf * 8 * 2, ngf * 8),
            decoder_layer(ngf * 8 * 2, ngf * 4),
            decoder_layer(ngf * 4 * 2, ngf * 2),
            decoder_layer(ngf * 2 * 2, ngf),
            decoder_layer(ngf * 2, output_nc, batchnorm=False)]
        for i, layer in enumerate(self.decoder, 1):
            setattr(self, 'dec%d' % i, layer)


class GeneratorUNet(Generator):

    def __init__(self, input_nc, output_nc, ngf):
        super().__init__(input_nc, output_nc, ngf)
        self._build_encoder_layers(input_nc, ngf)
        self._build_unet_decoder_layers(output_nc, ngf)
        self.apply(weights_init)

    def forward(self, x):
        ''' Encode '''
        enc_x = []
        for i, enc in enumerate(self.encoder):
            enc_x.append(x)
            x = enc(x)
        ''' Decode '''
        for i, dec in enumerate(self.decoder[:-1]):
            x = torch.cat((dec(x), enc_x[-i - 1]), 1)
        x = self.decoder[-1](x)
        return F.tanh(x)


class GeneratorEncoderDecode(Generator):

    def __init__(self, input_nc, output_nc, ngf):
        super().__init__(input_nc, output_nc, ngf)
        self._build_encoder_layers(input_nc, ngf)
        self._build_decoder_layers(output_nc, ngf)
        self.apply(weights_init)

    def forward(self, x):
        ''' Encode '''
        for enc in self.encoder:
            x = enc(x)
        ''' Decode '''
        for dec in self.decoders:
            x = dec(x)
        return F.tanh(x)


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
