import logging

import torch.backends.cudnn as cudnn

from datasets.loader import CustomDataLoader
from training import core, networks

cudnn.benchmark = True


def create_model(args):

    def g_net():
        return networks.define_G(
            args.input_nc, args.output_nc, args.ngf,
            args.which_model_netG, args.norm, not args.no_dropout,
            args.init_type, args.gpu_ids)

    def d_net():
        return networks.define_D(
            args.input_nc + args.output_nc, args.ndf,
            args.which_model_netD, args.n_layers_D, args.norm, args.no_lsgan,
            args.init_type, args.gpu_ids) if args.isTrain else None

    return {
        'cyclegan': lambda: ((g_net(), g_net()), (d_net(), d_net())),
        'pix2pix': lambda: (g_net(), d_net()),
    }[args.model]()


def main():
    log = logging.getLogger('pixsty')

    from options import TrainOptions
    parser = TrainOptions()
    parser.parser.add_argument('--subjects', type=str, nargs='+')
    args = parser.parse()

    log.info('Create dataset')
    train_loader = CustomDataLoader(args, phase='train')
    val_loader = CustomDataLoader(args, phase='val')
    print('training images = %d' % len(train_loader.dataset))
    print('validation images = %d' % len(val_loader.dataset))

    print('===> Build model')
    models = create_model(args)

    core_fn = core.training_estimator(models, args)
    core_fn(train_loader, val_loader, epochs=50)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        handlers=[logging.StreamHandler(), ])
    main()
