import torch
import torch.backends.cudnn as cudnn

import pix2pix
from pix2pix.torch import datasets

cudnn.benchmark = True


def main():
    opt = pix2pix.parser.parse_args()

    print('===> Load datasets')

    if opt.folderA and opt.folderB:
        train_dataset = datasets.ImageMixFolderDatasets(
                dataset_a=opt.folderA, dataset_b=opt.folderB)
        test_dataset = train_dataset
    else:
        train_dataset = datasets.ImageFolderDataset(
            root='./datasets/%s/train' % opt.dataset)
        test_dataset = datasets.ImageFolderDataset(
            root='./datasets/%s/test' % opt.dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=opt.batchSize,
        num_workers=opt.workers,
        pin_memory=True,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=opt.workers,
        pin_memory=True)

    print('===> Build model')
    pix = pix2pix.torch.Pix2Pix(opt, train_dataset)
    pix.detail()

    if opt.phase == 'train':
        print('===> Train model')
        pix.train(train_loader)
        pix.save_model()
    else:
        pix.test(test_loader)


if __name__ == '__main__':
    main()
