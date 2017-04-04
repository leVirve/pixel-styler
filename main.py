import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import datasets as dset

import pix2pix

cudnn.benchmark = True


def main():
    parser = pix2pix.parser
    opt = parser.parse_args()

    tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('===> Load datasets')
    train_dataset = dset.ImageFolderDataset(root='./datasets/facades/train',
                                            transform=tf)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt.batchSize,
                                               num_workers=opt.workers,
                                               pin_memory=True,
                                               shuffle=True)
    test_dataset = dset.ImageFolderDataset(root='./datasets/facades/test',
                                           transform=tf)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              num_workers=opt.workers,
                                              pin_memory=True)
    print('===> Build model')
    pix = pix2pix.Pix2Pix(opt, train_dataset)
    pix.detail()

    if opt.phase == 'train':
        print('===> Train model')
        pix.train(train_loader)
        pix.save_model()
    else:
        pix.test(test_loader)

if __name__ == '__main__':
    main()
