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
    print('===> Build model')
    pix = pix2pix.Pix2Pix(opt, train_dataset)
    pix.detail()

    print('===> Train model')
    pix.train(train_loader)
    pix.save_model()

if __name__ == '__main__':
    main()
