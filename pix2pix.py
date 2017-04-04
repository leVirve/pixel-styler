import argparse

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

from models import Discriminator, Generator

real_label, fake_label = 1, 0


class Pix2Pix():

    def __init__(self, opt, dataset):
        self.opt = opt
        self.dataset = dataset
        self.generator = Generator(opt.input_nc, opt.output_nc, opt.ngf)
        self.discriminator = Discriminator(opt.input_nc, opt.output_nc, opt.ndf)

    def setup(self):
        self._build_tensors()
        self._define_criterions()
        self._build_placeholders()
        self._define_optimizers()
        if self.opt.cuda:
            self.cuda()

    def train(self, data_loader):
        self.setup()
        num_batches = len(self.dataset) // self.opt.batchSize
        num_epochs = self.opt.epochs

        def stage_logging():
            print('Epoch [{:3d}/{}]({:02d}/{:02d}) =>'
                  ' Loss_D: {:.4f} Loss_G: {:.4f}'
                  ' D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
                        epoch + 1, num_epochs, i + 1, num_batches,
                        d_loss.data[0], g_loss.data[0], *d_f1, *d_f2))

        for epoch in range(num_epochs):
            for i, (img_a, img_b) in enumerate(data_loader):
                fake_b, real_ab, fake_ab = self.create_example(img_b, img_a)
                d_f1, d_loss = self.update_discriminator(real_ab, fake_ab)
                d_f2, g_loss = self.update_generator(fake_b, fake_ab)
                if (i + 1) % 3 == 0:
                    stage_logging()
            torchvision.utils.save_image(
                fake_b.data, './output/fake_samples_epoch%d.png' % (epoch + 1))
            torchvision.utils.save_image(
                self.real_a.data, './output/real_a_samples_epoch%d.png' % (epoch + 1))
            torchvision.utils.save_image(
                self.real_b.data, './output/real_b_samples_epoch%d.png' % (epoch + 1))

    def test(self, data_loader):
        import os
        generator = torch.load('./generator.pth')
        generator = generator.cuda() if self.opt.cuda else generator

        for i, (img_a, img_b) in enumerate(data_loader):
            input_img = Variable(img_b).cuda()
            out = generator(input_img)

            os.makedirs('result/facades/', exist_ok=True)
            torchvision.utils.save_image(
                out.data[0], './result/facades/%d.png' % i, nrow=1)

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
        output = self.discriminator(fake_ab)  # is fake_b really detach ?
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
        self.real_a.data.resize_(img_a.size()).copy_(img_a)
        self.real_b.data.resize_(img_b.size()).copy_(img_b)

        fake_b = self.generator(self.real_a)
        real_ab = torch.cat((self.real_a, self.real_b), 1)
        fake_ab = torch.cat((self.real_a, fake_b.detach()), 1)
        return fake_b, real_ab, fake_ab

    def save_model(self):
        torch.save(self.generator.state_dict(), './generator.pkl')
        torch.save(self.discriminator.state_dict(), './discriminator.pkl')
        torch.save(self.generator, './generator.pth')
        torch.save(self.discriminator, './discriminator.pth')

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


parser = argparse.ArgumentParser(description='pix2pix in PyTorch')
parser.add_argument('--phase', required=True, help='Pix2Pix in training / testing phase')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epochs', type=int, default=20, help='training epochs')
parser.add_argument('--batchSize', type=int, default=16, help='batch size of data')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--out', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
