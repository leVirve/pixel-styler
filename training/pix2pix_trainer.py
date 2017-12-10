import torch
from torch.autograd import Variable

from training.base_trainer import BaseTrainer
from training import networks
from util.image_pool import ImagePool


class Pix2PixTrainer(BaseTrainer):
    name = 'Pix2PixTrainer'

    def __init__(self, opt):
        BaseTrainer.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # load/define networks
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf,
            opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf,
                opt.which_model_netD,
                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.criterionGAN = networks.GANLoss(use_lsgan=True, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.schedulers = [
                networks.get_scheduler(optimizer, opt)
                for optimizer in (self.optimizer_G, self.optimizer_D)]

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    def get_image_paths(self, data):
        AtoB = self.opt.which_direction == 'AtoB'
        self.image_paths = data['A_paths' if AtoB else 'B_paths']
        return self.image_paths

    def optimize_parameters(self, data):
        AtoB = self.opt.which_direction == 'AtoB'
        data_A = data['A' if AtoB else 'B']
        data_B = data['B' if AtoB else 'A']

        self.input_A.resize_(data_A.size()).copy_(data_A)
        self.input_B.resize_(data_B.size()).copy_(data_B)
        real_A, real_B = Variable(self.input_A), Variable(self.input_B)
        # real_A, real_B = Variable(data_A), Variable(data_B)
        fake_B = self.netG(real_A)

        self.optimizer_D.zero_grad()
        fake_AB = torch.cat((real_A, fake_B), dim=1)
        real_AB = torch.cat((real_A, real_B), dim=1)
        loss_D_fake = self.criterionGAN(self.netD(fake_AB.detach()), False)
        loss_D_real = self.criterionGAN(self.netD(real_AB), True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        fake_AB = torch.cat((real_A, fake_B), dim=1)
        loss_G_GAN = self.criterionGAN(self.netD(fake_AB), True)
        loss_G_L1 = self.criterionL1(fake_B, real_B) * self.opt.lambda_A
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        self.optimizer_G.step()

        result = {'real_A': real_A.data,
                  'real_B': real_B.data,
                  'fake_B': fake_B.data}
        loss = {'G/adv': loss_G_GAN,
                'G/l1': loss_G_L1,
                'D/real': loss_D_real,
                'D/fake': loss_D_fake}
        return loss, result

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
