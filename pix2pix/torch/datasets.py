import glob
import random

import torchvision.datasets as dset


class ImageFolderDataset(dset.ImageFolder):
    """ Custom Dataset compatible with torch.utils.data.DataLoader. """

    def __init__(self, root, transform=None,
                 loader=dset.folder.default_loader):
        """
        Args:
            root: image directory.
            transform: image transformer
        """
        self.root = root
        self.imgs = glob.glob('%s/*.jpg' % root)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """ Returns an image pair. """
        path = self.imgs[index]
        img = self.loader(path)

        ''' Apply transformations
            - Scale up
            - Perform transforms defined in `main()`
            - Normalization on images from [0, 1] -> [-1, 1]
            - Random croping to desired size
        '''
        img = resize(img, [286 * 2, 286])
        if self.transform is not None:
            img = self.transform(img)
        img = normalize(img)

        a, b = split_ab(img)
        a = random_crop(a, (256, 256))
        b = random_crop(b, (256, 256))

        return a, b


def resize(img, shape):
    return img.resize(shape)


def normalize(img):
    return img * 2 - 1


def split_ab(img):
    _, _, w = img.size()
    return img[:, :, :w // 2], img[:, :, w // 2:]


def random_crop(img, shape):
    _, h, w = img.size()
    th, tw = shape
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return img[:, x1:x1 + tw, y1:y1 + th]
