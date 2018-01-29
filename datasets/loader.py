import torch


def create_dataset(opt):
    builder = None
    if opt.dataset_mode == 'aligned':
        from datasets.base_dataset import AlignedDataset
        builder = AlignedDataset
    elif opt.dataset_mode == 'unaligned':
        from datasets.base_dataset import UnalignedDataset
        builder = UnalignedDataset
    elif opt.dataset_mode == 'single':
        from datasets.base_dataset import SingleDataset
        builder = SingleDataset
    elif opt.dataset_mode == 'mscoco':
        from datasets.coco import CocoDataLoader
        builder = CocoDataLoader
    elif opt.dataset_mode == 'voc':
        from datasets.voc import VOC2010Loader
        builder = VOC2010Loader
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.dataset_mode)

    print('Dataset [%s] was created' % (builder.name))
    return builder(opt)


class CustomDataLoader():

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.dataset = self._create_dataset_wrapper(opt, phase)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def _create_dataset_wrapper(self, opt, phase):
        if phase == 'val':
            opt.isTrain = False
            opt.serial_batches = True
            val_dataset = create_dataset(opt)
            opt.isTrain = True
            opt.serial_batches = False
            return val_dataset
        return create_dataset(opt)

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size) // self.opt.batchSize

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
