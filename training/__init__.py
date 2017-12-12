
def create_trainer(opt):
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from training.trainer import CycleGANTrainer
        trainer = CycleGANTrainer
    elif opt.model == 'pix2pix':
        assert opt.dataset_mode in {'aligned', 'mscoco'}  # TODO: use some better checks
        from training.trainer import Pix2PixTrainer
        trainer = Pix2PixTrainer
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from training.trainer import TestTrainer
        trainer = TestTrainer
    else:
        raise ValueError('Model [%s] not recognized.' % opt.model)

    print(f'Trainer [{trainer.name}] was created')
    return trainer(opt)
