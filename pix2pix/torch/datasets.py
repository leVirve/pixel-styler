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
        img = img.resize([286 * 2, 286])

        if self.transform is not None:
            img = self.transform(img)

        img = img * 2 - 1

        _, _, dim = img.size()
        a, b = img[:, :, :dim // 2], img[:, :, dim // 2:]

        w = h = 286
        tw = th = 256
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        a = a[:, x1:x1 + tw, y1:y1 + th]
        b = b[:, x1:x1 + tw, y1:y1 + th]

        return a, b
