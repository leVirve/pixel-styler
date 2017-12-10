import torch


def create_dataset(opt):
    builder = None
    if opt.dataset_mode == 'aligned':
        from data.base_dataset import AlignedDataset
        builder = AlignedDataset
    elif opt.dataset_mode == 'unaligned':
        from data.base_dataset import UnalignedDataset
        builder = UnalignedDataset
    elif opt.dataset_mode == 'single':
        from data.base_dataset import SingleDataset
        builder = SingleDataset
    elif opt.dataset_mode == 'bmvc_wei':
        from data.flower import BMVCFlower
        builder = BMVCFlower
    elif opt.dataset_mode == 'mscoco':
        from data.coco import CocoDataLoader
        builder = CocoDataLoader
    else:
        raise ValueError('Dataset [%s] not recognized.' % opt.dataset_mode)

    print('Dataset [%s] was created' % (builder.name))
    return builder(opt)


class CustomDataLoader():

    def __init__(self, opt):
        self.opt = opt
        self.dataset = create_dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size) // self.opt.batchSize

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
