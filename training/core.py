import logging
from functools import partial

import torch
import onegan
import onegan.losses as L
from onegan.utils import to_var

from training import networks
from training.image_pool import ImagePool


def create_optimizer(models, args):
    return [
        torch.optim.Adam(m.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        for m in models
    ]


def create_scheduler(optimizers, args):
    return [
        networks.get_scheduler(optim, args)
        for optim in optimizers
    ]


def training_estimator(models, args):

    def _closure(g, d, data, volatile=False):
        AtoB = args.which_direction == 'AtoB'
        source = to_var(data['A' if AtoB else 'B'], volatile=volatile)
        target = to_var(data['B' if AtoB else 'A'], volatile=volatile)
        output = g(source)

        # fake
        fake = pair_pool.query(torch.cat((source, output), dim=1).data)
        pred_fake = d(fake.detach())
        loss_d_fake = gan_loss(pred_fake, 0)
        # real
        real = torch.cat((source, target), dim=1)
        pred_real = d(real)
        loss_d_real = gan_loss(pred_real, 1)
        yield {
            'loss/loss_d': (loss_d_fake + loss_d_real) * 0.5,
            'loss/d_fake': loss_d_fake,
            'loss/d_real': loss_d_real,
        }

        # generated
        fake = torch.cat([source, output], dim=1)
        pred_fake = d(fake)
        loss_g_gan = gan_loss(pred_fake, 1)
        loss_g_l1 = l1_loss(output, target)
        yield {
            'loss/loss_g': loss_g_gan * args.lambda_A + loss_g_l1,
            'loss/g_l1': loss_g_l1,
            'loss/g_gan': loss_g_gan,
        }

        accuracy = {'acc/psnr': psnr(output, target)}
        yield accuracy

        viz_results = {
            'input': source.data,
            'output': output.data,
            'target': target.data}
        tensorboard.image(
            viz_results,
            epoch=estimator.state['epoch'], prefix='val_' if volatile else 'train_')
        yield

    pair_pool = ImagePool(args.pool_size)

    gan_loss = L.adversarial_ce_loss if args.no_lsgan else L.adversarial_ls_loss
    l1_loss = torch.nn.functional.l1_loss

    log = logging.getLogger(f'pixsty.{args.name}')

    checkpoint = onegan.extension.GANCheckpoint(name=args.name, save_epochs=5)
    tensorboard = onegan.extension.TensorBoardLogger(name=args.name, max_num_images=30)
    psnr = onegan.metrics.psnr

    optimizers = create_optimizer(models, args)
    schedulers = create_scheduler(optimizers, args)

    args.pretrain_G = None
    args.pretrain_D = None
    if args.pretrain_G:
        log.info('Load pre-trained weight as initialization for G')
        checkpoint.apply(args.pretrain_G, models[0], remove_module=False)
    if args.pretrain_D:
        log.info('Load pre-trained weight as initialization for D')
        checkpoint.apply(args.pretrain_D, models[1], remove_module=False)

    log.info('Build training esimator')
    estimator = onegan.estimator.OneGANEstimator(
        models, optimizers,
        lr_scheduler=schedulers, logger=tensorboard, saver=checkpoint, name=args.name)

    return partial(estimator.run, update_fn=partial(_closure), inference_fn=partial(_closure, volatile=True))
