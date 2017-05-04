import os
import time

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

from pix2pix.torch.models import Discriminator, GeneratorUNet
from pix2pix import utils

real_label, fake_label = 1, 0


class Pix2Pix():

    def __init__(self, opt, dataset):
        self.opt = opt
        self.dataset = dataset
        self.AtoB = self.opt.direction == 'AtoB'
        self.generator = GeneratorUNet(opt.input_nc, opt.output_nc, opt.ngf)
        self.discriminator = Discriminator(opt.input_nc, opt.output_nc, opt.ndf)

        self.Tensor = torch.cuda.FloatTensor if self.opt.cuda else torch.Tensor
        self.input_a = self.Tensor(opt.batchSize,
                                   opt.input_nc, opt.output_nc, opt.imageSize)
        self.input_b = self.Tensor(opt.batchSize,
                                   opt.input_nc, opt.output_nc, opt.imageSize)
        self.label = Variable(self.Tensor(opt.batchSize), requires_grad=False)

        utils.mkdir(opt.log_dir)
        utils.mkdir(opt.result_dir)

    def setup(self):
        η, β1 = self.opt.lr, self.opt.beta1
        self.λ = self.opt.lamb

        self.criterion = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=η, betas=(β1, 0.999))
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=η, betas=(β1, 0.999))

        if self.opt.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

    def train(self, data_loader):

        def batch_logging():
            print('Epoch#{}/{} ({:02d}/{:02d}) =>'
                  ' D_real: {:.4f} D_fake: {:.4f}'
                  ' G_GAN: {:.4f} G_L1: {:.4f}'.format(
                        epoch, opt.epochs, batch, len(data_loader),
                        d_loss[0].data[0], d_loss[1].data[0],
                        g_loss[0].data[0], g_loss[1].data[0]))

        def epoch_logging():
            torchvision.utils.save_image(
                self.fake_b.data / 2 + 0.5, os.path.join(
                    opt.log_dir, 'fake_samples_epoch%d.png' % (epoch)))
            self.save_model('epoch%d' % (epoch))

        opt = self.opt
        self.setup()

        for epoch in range(1, opt.epochs + 1):
            s = time.time()
            for batch, (img_a, img_b) in enumerate(data_loader, start=1):
                self.create_example(img_a, img_b)
                d_loss = self.update_discriminator()
                g_loss = self.update_generator()

                if batch % opt.log_freq == 0:
                    batch_logging()
            print('Epoch#{}: {:.4f} sec'.format(epoch, time.time() - s))

            if epoch % opt.save_freq == 0 or epoch == opt.epochs:
                epoch_logging()

    def test(self, data_loader):
        opt = self.opt

        assert opt.netG
        generator = torch.load(opt.netG)
        generator = generator.cuda() if opt.cuda else generator

        folder = utils.mkdir(opt.dataset, parent=opt.result_dir)
        for (img_a, img_b), filepath in zip(data_loader,
                                            data_loader.dataset.imgs):
            img = img_a if self.AtoB else img_b
            input_img = Variable(img).cuda()
            out = generator(input_img)
            print('Processing on {}'.format(filepath))

            filename = os.path.basename(filepath)
            torchvision.utils.save_image(
                out.data[0], os.path.join(folder, filename), nrow=1)

    def update_discriminator(self):
        ''' discriminator
            Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        '''
        label = self.label
        self.d_optimizer.zero_grad()

        # Train with real
        real_ab = torch.cat((self.real_a, self.real_b), 1)
        output = self.discriminator(real_ab)
        label.data.resize_(output.size()).fill_(real_label)
        d_real_loss = self.criterion(output, label)

        # Train with fake
        fake_ab = torch.cat((self.real_a, self.fake_b), 1)
        output = self.discriminator(fake_ab.detach())
        label.data.resize_(output.size()).fill_(fake_label)
        d_fake_loss = self.criterion(output, label)

        d_loss = (d_real_loss + d_fake_loss) * 0.5
        d_loss.backward()

        self.d_optimizer.step()
        return d_real_loss, d_fake_loss

    def update_generator(self):
        ''' generator
            Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        '''
        label = self.label
        self.g_optimizer.zero_grad()

        fake_ab = torch.cat((self.real_a, self.fake_b), 1)
        output = self.discriminator(fake_ab)
        label.data.resize_(output.size()).fill_(real_label)

        g_loss_gan = self.criterion(output, label)
        g_loss_l1 = self.λ * self.criterion_l1(self.fake_b, self.real_b)
        g_loss = g_loss_gan + g_loss_l1
        g_loss.backward()

        self.g_optimizer.step()
        return g_loss_gan, g_loss_l1

    def create_example(self, img_a, img_b):
        if self.AtoB:
            a, b = img_a, img_b
        else:
            b, a = img_a, img_b

        self.input_a.resize_(a.size()).copy_(a)
        self.input_b.resize_(b.size()).copy_(b)

        self.real_a = Variable(self.input_a)
        self.fake_b = self.generator(self.real_a)
        self.real_b = Variable(self.input_b)

    def save_model(self, suffix=None):
        # torch.save(self.generator.state_dict(), './generator.pkl')
        # torch.save(self.discriminator.state_dict(), './discriminator.pkl')
        folder = self.opt.log_dir
        torch.save(self.generator,
                   os.path.join(folder, 'generator_%s.pth' % suffix))
        torch.save(self.discriminator,
                   os.path.join(folder, 'discriminator_%s.pth' % suffix))

    def detail(self):
        print(self.generator)
        print(self.discriminator)
