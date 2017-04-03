import glob

import torchvision.datasets as dset


class ImageFolderDataset(dset.ImageFolder):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, transform=None,
                 loader=dset.folder.default_loader):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            transform: image transformer
        """
        self.root = root
        self.imgs = glob.glob('%s/*.jpg' % root)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        """Returns one image pair."""
        path = self.imgs[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        _, _, w = img.size()
        a, b = img[:, :, :w // 2], img[:, :, w // 2:]

        return a, b
