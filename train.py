import tqdm
from onegan.utils import History, Logger
from onegan.extensions import ImageSummaryExtention, HistoryExtention

from data.loader import CustomDataLoader
from options import TrainOptions
from training import create_trainer


if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataloader = CustomDataLoader(opt)
    print('#training images = %d' % len(dataloader.dataset))

    trainer = create_trainer(opt)
    logger = Logger(name='dbg')
    summary_history = HistoryExtention(logger)
    summary_image = ImageSummaryExtention(logger, summary_num_images=30)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        history = History(length=len(dataloader))
        progress = tqdm.tqdm(dataloader)
        for data in progress:
            trainer.set_input(data)
            loss_terms, visual_result = trainer.optimize_parameters()
            progress.set_description(f'Epoch#{epoch}')
            progress.set_postfix(history.add(loss_terms, {}))
            summary_image(visual_result, epoch=epoch, prefix='train_')

        summary_history(history.metric(), epoch)
        trainer.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            print(f'saving the model at the end of epoch {epoch}')
            trainer.save('latest')
            trainer.save(epoch)
