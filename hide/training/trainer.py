<<<<<<< HEAD:hide/training/trainer.py
import itertools
import os

import torch
from torch.autograd import Variable
import onegan.losses as L

from training import networks
from training.image_pool import ImagePool


class BaseTrainer:
    name = 'BaseTrainer'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def _prepare_data(self, data):
        self.input = data

    def test(self, data):
        pass

    def optimize_parameters(self, data):
        pass

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        print('learning rate =', self.schedulers[0].get_lr())


class Pix2PixTrainer(BaseTrainer):
    name = 'Pix2PixTrainer'

    def __init__(self, opt):
        super().__init__(opt)
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf,
            opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf,
                opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.localnetD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf,
                opt.which_model_netD, opt.n_layers_D - 1, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.schedulers = [
                networks.get_scheduler(optimizer, opt)
                for optimizer in (self.optimizer_G, self.optimizer_D)]

    def test(self, data):
        input_A, input_B = data
        real_A = Variable(input_A, volatile=True)
        real_B = Variable(input_B, volatile=True)
        fake_B = self.netG(real_A)
        return {'real_A': real_A.data, 'real_B': real_B.data, 'fake_B': fake_B.data}
=======
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
>>>>>>> 6d9e17390a4b2e186bc45427c37f4ff42409bb40:models/cycle_gan_model.py

    def optimize_parameters(self, data, update_g=False, update_d=True):
        input_A, input_B, mask = data
        real_A, real_B, mask = Variable(input_A), Variable(input_B), Variable(mask)
        fake_B = self.netG(real_A)

        self.optimizer_D.zero_grad()
        fake_AB = self.fake_AB_pool.query(torch.cat((real_A, fake_B), dim=1).data)
        real_AB = torch.cat((real_A, real_B), dim=1)
        loss_D_fake = L.adversarial_w_loss(self.netD(fake_AB), False)
        loss_D_real = L.adversarial_w_loss(self.netD(real_AB), True)
        loss_D_gp = L.gradient_penalty(self.netD, real_AB, fake_AB) * 10
        loss_D = (loss_D_fake + loss_D_real) * 0.5

<<<<<<< HEAD:hide/training/trainer.py
        fake_AB = Variable(torch.cat((real_A, fake_B * mask), dim=1).data)
        real_AB = torch.cat((real_A, real_B * mask), dim=1)
        loss_localD_fake = L.adversarial_w_loss(self.localnetD(fake_AB), False)
        loss_localD_real = L.adversarial_w_loss(self.localnetD(real_AB), True)
        loss_localD_gp = L.gradient_penalty(self.localnetD, real_AB, fake_AB) * 10
        loss_localD = (loss_localD_fake + loss_localD_real) * 0.5

        if update_d:
            loss_D.backward()
            loss_localD.backward()
            loss_D_gp.backward()
            loss_localD_gp.backward()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        fake_AB = torch.cat((real_A, fake_B), dim=1)
        loss_G_GAN = L.adversarial_w_loss(self.netD(fake_AB), True)
        fake_AB = torch.cat((real_A, fake_B * mask), dim=1)
        loss_G_localGAN = L.adversarial_w_loss(self.netD(fake_AB), True)
        loss_G_L1 = L.l1_loss(fake_B, real_B)
        loss_G = loss_G_GAN + loss_G_localGAN + loss_G_L1 * self.opt.lambda_A
        if update_g:
            loss_G.backward()
        self.optimizer_G.step()

        result = {'real_A': real_A.data, 'real_B': real_B.data, 'fake_B': fake_B.data}
        loss = {'G/adv': loss_G_GAN, 'G/l1': loss_G_L1, 'D/real': loss_D_real, 'D/fake': loss_D_fake, 'D/gp': loss_D_gp}
        loss.update({'G/adv_l': loss_G_localGAN, 'D/real_l': loss_localD_real, 'D/fake_l': loss_localD_fake, 'D/gp_l': loss_localD_gp})
        return loss, result

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)


class CycleGANTrainer(BaseTrainer):
    name = 'CycleGANTrainer'

    def __init__(self, opt):
        super().__init__(opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
=======
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True) 
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0,
                                help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
>>>>>>> 6d9e17390a4b2e186bc45427c37f4ff42409bb40:models/cycle_gan_model.py

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def _prepare_data(self, data):
        AtoB = self.opt.which_direction == 'AtoB'
<<<<<<< HEAD:hide/training/trainer.py
        input_A = data['A' if AtoB else 'B']
        input_B = data['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def test(self, data):
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data

        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.netG_B(real_B)
        self.rec_B = self.netG_A(fake_A).data
        self.fake_A = fake_A.data

=======
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

>>>>>>> 6d9e17390a4b2e186bc45427c37f4ff42409bb40:models/cycle_gan_model.py
    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def optimize_parameters(self,data):
        # forward
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
<<<<<<< HEAD:hide/training/trainer.py
        self.optimizer_D_B.step()

        loss = {'D_A': self.loss_D_A, 'G_A': self.loss_G_A, 'Cyc_A': self.loss_cycle_A,
                'D_B': self.loss_D_B, 'G_B': self.loss_G_B, 'Cyc_B':  self.loss_cycle_B}
        if self.opt.identity > 0.0:
            loss['idt_A'] = self.loss_idt_A
            loss['idt_B'] = self.loss_idt_B

        result = {'real_A': real_A.data, 'fake_B': fake_B.data, 'rec_A': rec_A.data,
                  'real_B': real_B.data, 'fake_A': fake_A.data, 'rec_B': rec_B.data}
        if self.opt.isTrain and self.opt.identity > 0.0:
            result['idt_A'] = self.idt_A.data
            result['idt_B'] = self.idt_B.data
        return loss, result

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)


class TestTrainer(BaseTrainer):
    name = 'TestTrainer'

    def __init__(self, opt):
        assert(not opt.isTrain)
        super().__init__(opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      opt.init_type,
                                      self.gpu_ids)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def _prepare_data(self, data):
        # we need to use single_dataset mode
        input_A = data['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)

    def test(self, data):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)

    def get_current_visuals(self):
        return {'real_A': real_A.data, 'fake_B': fake_B.data}
=======
        self.optimizer_D.step()


>>>>>>> 6d9e17390a4b2e186bc45427c37f4ff42409bb40:models/cycle_gan_model.py
