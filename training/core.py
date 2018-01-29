import logging
import itertools
from functools import partial

import torch
import onegan
import onegan.losses as L
import torch.nn.functional as F
from onegan.utils import to_var

from training import networks
from training.image_pool import ImagePool


def create_optimizer(models, args):
    return [
        torch.optim.Adam(m.parameters() if hasattr(m, 'parameters') else m,
                         lr=args.lr, betas=(args.beta1, 0.999))
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


def cyclegan_estimator(models, args):

    def _closure(models, data, volatile=False):
        ga, gb, da, db = models
        AtoB = args.which_direction == 'AtoB'
        real_a = to_var(data['A' if AtoB else 'B'], volatile=volatile)
        real_b = to_var(data['B' if AtoB else 'A'], volatile=volatile)

        if args.identity > 0:
            idt_a = ga(real_b)
            loss_idt_a = F.l1_loss(idt_a, real_b) * args.lambda_B * args.identity
            idt_b = gb(real_a)
            loss_idt_b = F.l1_loss(idt_b, real_a) * args.lambda_A * args.identity
        else:
            loss_idt_a = loss_idt_b = 0

        fake_a = gb(real_b)
        fake_b = ga(real_a)
        loss_g_a = gan_loss(da(fake_b), 1)
        loss_g_b = gan_loss(db(fake_a), 1)

        # forward cycle loss
        rec_a = gb(fake_b)
        loss_cycle_a = F.l1_loss(rec_a, real_a) * args.lambda_A

        # backward cycle loss
        rec_b = ga(fake_a)
        loss_cycle_b = F.l1_loss(rec_b, real_b) * args.lambda_B

        yield {
            'loss/g': loss_g_a + loss_g_b + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b,
            'loss/g_a': loss_g_a,
            'loss/g_b': loss_g_b,
            'loss/cycle_a': loss_cycle_a,
            'loss/cycle_b': loss_cycle_b,
        }, (optim_g, 'loss/g')

        def _d(d, real, fake):
            loss_d_real = gan_loss(d(real), 1)
            loss_d_fake = gan_loss(d(fake.detach()), 0)
            return (loss_d_real + loss_d_fake) * 0.5

        loss_d_a = _d(da, real_b, b_pool.query(fake_b.data))
        yield {'loss/d_a': loss_d_a}, (optim_da, 'loss/d_a')

        loss_d_b = _d(db, real_a, a_pool.query(fake_a.data))
        yield {'loss/d_b': loss_d_b}, (optim_db, 'loss/d_b')

        viz_results = {
            'realA': real_a.data,
            'realB': real_b.data,
            'fakeA': fake_a.data,
            'fakeB': fake_b.data,
            'recA': rec_a.data,
            'recB': rec_b.data}
        if args.identity > 0:
            viz_results['idtA'] = idt_a
            viz_results['idtB'] = idt_b
        tensorboard.image(
            viz_results,
            epoch=estimator.state['epoch'], prefix='val_' if volatile else 'train_')
        yield

    def _epoch_ending_fn(epoch):
        tensorboard.scalar(estimator.history.metric(), epoch)
        checkpoint._save(f'G_A-{epoch}.pth', model_ga, None, epoch)
        checkpoint._save(f'G_B-{epoch}.pth', model_gb, None, epoch)
        checkpoint._save(f'D_A-{epoch}.pth', model_d[0], None, epoch)
        checkpoint._save(f'D_B-{epoch}.pth', model_d[1], None, epoch)
        estimator.adjust_learning_rate(('loss/loss_g_val', 'loss/loss_d_a_val', 'loss/loss_d_b_val'))

    a_pool = ImagePool(args.pool_size)
    b_pool = ImagePool(args.pool_size)

    log = logging.getLogger(f'pixsty.{args.name}')

    gan_loss = L.adversarial_ce_loss if args.no_lsgan else L.adversarial_ls_loss
    checkpoint = onegan.extension.GANCheckpoint(name=args.name, save_epochs=5)
    tensorboard = onegan.extension.TensorBoardLogger(name=args.name, max_num_images=30)

    (model_ga, model_gb), model_d = models
    model_g = itertools.chain(model_ga.parameters(), model_gb.parameters())
    optim_g, optim_da, optim_db = create_optimizer([model_g, *model_d], args)
    schedulers = create_scheduler([optim_g, optim_da, optim_db], args)

    log.info('Build CycleGAN training esimator')
    estimator = onegan.estimator.OneGANEstimator(
        [model_ga, model_gb, *model_d], lr_scheduler=schedulers, name=args.name)

    return partial(
        estimator.dummy_run,
        update_fn=partial(_closure),
        inference_fn=partial(_closure, volatile=True),
        epoch_fn=_epoch_ending_fn)
