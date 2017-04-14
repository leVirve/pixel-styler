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
        utils.mkdir(opt.log_dir)
        utils.mkdir(opt.result_dir)

    def setup(self):
        self._build_tensors()
        self._define_criterions()
        self._build_placeholders()
        self._define_optimizers()
        if self.opt.cuda:
            self.cuda()

    def train(self, data_loader):

        def batch_logging():
            print('Epoch#{}/{} ({:02d}/{:02d}) =>'
                  ' Loss_D: {:.4f} Loss_G: {:.4f}'
                  ' D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
                        epoch + 1, opt.epochs, batch, len(data_loader),
                        d_loss.data[0], g_loss.data[0], *d_f1, *d_f2))

        def epoch_logging():
            torchvision.utils.save_image(
                fake_b.data, os.path.join(
                    opt.log_dir, 'fake_samples_epoch%d.png' % (epoch + 1)))
            self.save_model('epoch%d' % (epoch + 1))

        opt = self.opt
        self.setup()

        for epoch in range(opt.epochs):
            s = time.time()
            for batch, (img_a, img_b) in enumerate(data_loader, start=1):
                fake_b, real_ab, fake_ab = self.create_example(img_a, img_b)
                d_f1, d_loss = self.update_discriminator(real_ab, fake_ab)
                d_f2, g_loss = self.update_generator(fake_b, fake_ab)

                if batch % opt.log_freq == 0:
                    batch_logging()
            print('Epoch#{}: {:.4f} sec'.format(epoch + 1, time.time() - s))

            if epoch % opt.save_freq == 0:
                epoch_logging()

    def test(self, data_loader):
        opt = self.opt

        assert opt.netG
        generator = torch.load(opt.netG)
        generator = generator.cuda() if opt.cuda else generator

        folder = utils.mkdir(opt.dataset, parent=opt.result_dir)
        for (img_a, img_b), filepath in zip(data_loader,
                                            data_loader.dataset.imgs):
            input_img = Variable(img_b).cuda()
            out = generator(input_img)
            print('Processing on {}'.format(filepath))

            filename = os.path.basename(filepath)
            torchvision.utils.save_image(
                out.data[0], os.path.join(folder, filename), nrow=1)

    def update_discriminator(self, real_ab, fake_ab):
        ''' discriminator
            Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        '''
        label = self.label
        self.d_optimizer.zero_grad()

        # Train with real
        output = self.discriminator(real_ab)
        label.data.resize_(output.size()).fill_(real_label)
        d_real_loss = self.criterion(output, label)
        d_real_loss.backward()
        d_x = output.data.mean()

        # Train with fake
        output = self.discriminator(fake_ab)
        label.data.resize_(output.size()).fill_(fake_label)
        d_fake_loss = self.criterion(output, label)
        d_fake_loss.backward()
        d_g_z1 = output.data.mean()

        d_loss = (d_real_loss + d_fake_loss) / 2.0

        self.d_optimizer.step()
        return (d_x, d_g_z1), d_loss

    def update_generator(self, fake_b, fake_ab):
        ''' generator
            Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        '''
        label = self.label
        self.generator.zero_grad()

        output = self.discriminator(fake_ab)
        label.data.resize_(output.size()).fill_(real_label)
        g_loss = self.criterion(output, label) + \
            self.λ * self.criterion_l1(fake_b, self.real_b)
        g_loss.backward()
        d_g_z2 = output.data.mean()

        self.g_optimizer.step()
        return (d_g_z2,), g_loss

    def create_example(self, img_a, img_b):
        if self.AtoB:
            a, b = img_a, img_b
        else:
            b, a = img_a, img_b

        self.real_a.data.resize_(img_a.size()).copy_(a)
        self.real_b.data.resize_(img_b.size()).copy_(b)

        fake_b = self.generator(self.real_a)
        real_ab = torch.cat((self.real_a, self.real_b), 1)
        fake_ab = torch.cat((self.real_a, fake_b.detach()), 1)
        return fake_b, real_ab, fake_ab

    def save_model(self, suffix=None):
        # torch.save(self.generator.state_dict(), './generator.pkl')
        # torch.save(self.discriminator.state_dict(), './discriminator.pkl')
        folder = self.opt.log_dir
        torch.save(self.generator,
                   os.path.join(folder, 'generator_%s.pth' % suffix))
        torch.save(self.discriminator,
                   os.path.join(folder, 'discriminator_%s.pth' % suffix))

    def _build_tensors(self):
        opt = self.opt
        batch_size, input_nc, output_nc, img_sz = (
            opt.batchSize, opt.input_nc, opt.output_nc, opt.imageSize)
        self.label = torch.FloatTensor(batch_size)
        self.real_a = torch.FloatTensor(batch_size, input_nc, img_sz, img_sz)
        self.real_b = torch.FloatTensor(batch_size, output_nc, img_sz, img_sz)
        self.fake_b = None
        self.real_ab = None
        self.fake_ab = None

    def _build_placeholders(self):
        self.real_a = Variable(self.real_a)
        self.real_b = Variable(self.real_b)
        self.label = Variable(self.label)

    def _define_optimizers(self):
        η, β1 = self.opt.lr, self.opt.beta1
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=η, betas=(β1, 0.999))
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=η, betas=(β1, 0.999))

    def _define_criterions(self):
        self.criterion = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()
        self.λ = self.opt.lamb

    def cuda(self):
        self.generator = self.generator.cuda()
        self.discriminator = self.discriminator.cuda()
        self.criterion = self.criterion.cuda()
        self.criterion_l1 = self.criterion_l1.cuda()
        self.real_a = self.real_a.cuda()
        self.real_b = self.real_b.cuda()
        self.label = self.label.cuda()

    def detail(self):
        print(self.generator)
        print(self.discriminator)
