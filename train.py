import tqdm
import torch
from onegan.extensions import Logger, History

from data.loader import CustomDataLoader
from options import TrainOptions
from training import create_trainer


def main(opt):

    def prepare_input(data):
        data_A = torch.cat((data['A' if AtoB else 'B'], data['binseg']), dim=1)
        data_B = data['B' if AtoB else 'A']
        mask = data['binseg']
        if opt.gpu_ids:
            data_A = data_A.cuda()
            data_B = data_B.cuda()
            mask = mask.cuda()
        return data_A, data_B, mask

    train_loader = CustomDataLoader(opt, phase='train')
    val_loader = CustomDataLoader(opt, phase='val')
    AtoB = opt.which_direction == 'AtoB'
    print('training images = %d' % len(train_loader.dataset))
    print('validation images = %d' % len(val_loader.dataset))

    trainer = create_trainer(opt)
    logger = Logger(name=opt.name, max_num_images=30)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        ''' train '''
        trainer.netG.train()
        history = History(length=len(train_loader))
        pbar = tqdm.tqdm(train_loader)
        pbar.set_description(f'Epoch#{epoch}')
        for i, data in enumerate(pbar, 1):
            loss_terms, images = trainer.optimize_parameters(prepare_input(data), update_g=i % 5 == 0, update_d=True)
            pbar.set_postfix(history.add(loss_terms, {}))
            logger.image(images, epoch=epoch, prefix='train_')
        logger.scalar(history.metric(), epoch)

        ''' validate '''
        trainer.netG.eval()
        history = History(length=len(val_loader))
        for data in val_loader:
            loss_terms, images = trainer.optimize_parameters(prepare_input(data), update_g=False, update_d=False)
            history.add(loss_terms, {}, log_suffix='_val')
            logger.image(images, epoch=epoch, prefix='val_')
        logger.scalar(history.metric(), epoch)

        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}')
            trainer.save('latest')
            trainer.save(epoch)
        trainer.update_learning_rate()


if __name__ == '__main__':
    parser = TrainOptions()
    parser.parser.add_argument('--subjects', type=str, nargs='+')
    main(parser.parse())
