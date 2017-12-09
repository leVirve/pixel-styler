
def create_trainer(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cyclegan_trainer import CycleGANTrainer
        trainer = CycleGANTrainer()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_trainer import Pix2PixTrainer
        trainer = Pix2PixTrainer()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_trainer import TestTrainer
        trainer = TestTrainer()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    trainer.initialize(opt)
    print("Trainer [%s] was created" % (trainer.name()))
    return trainer
