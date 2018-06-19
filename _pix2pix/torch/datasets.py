import glob
import random

from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as dset

load_size = 286
fine_size = 256

transform = transforms.Compose([
    transforms.Scale(load_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


class ImageFolderDataset(dset.ImageFolder):
    """ Custom Dataset compatible with torch.utils.data.DataLoader. """

    def __init__(self, root, loader=dset.folder.default_loader):
        """
        Args:
            root: image directory.
            transform: image transformer
        """
        self.root = root
        self.imgs = find_images(root)
        self.loader = loader

    def load(self, path):
        img = self.loader(path)
        a, b = random_flip(split_ab(img))
        a, b = random_crop((transform(a), transform(b)),
                           size=(fine_size, fine_size))
        return a, b

    def __getitem__(self, index):
        """ Returns an image pair. """
        return self.load(self.imgs[index])


class ImageMixFolderDatasets(dset.ImageFolder):
    """ Custom Mix Datasets compatible with torch.utils.data.DataLoader. """

    def __init__(self, dataset_a, dataset_b, loader=dset.folder.default_loader):
        """
        Args:
            dataset_a: image_A dataset directory.
            dataset_b: image_B dataset directory.
            transform: image transformer
        """
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.img_as = find_images(dataset_a)
        self.img_bs = find_images(dataset_b)
        self.loader = loader

        assert len(self.img_as) == len(self.img_bs)
        print('Found A and B pairs:', len(self.img_as))

    def load(self, path):
        img = self.loader(path)
        return img

    def __getitem__(self, index):
        """ Returns an image pair. """
        a = self.load(self.img_as[index])
        b = self.load(self.img_bs[index])
        a, b = random_flip((a, b))
        a, b = random_crop((transform(a), transform(b)),
                           size=(fine_size, fine_size))
        return a, b

    def __len__(self):
        return len(self.img_as)


def split_ab(img):
    h, w = img.height, img.width
    return img.crop((0, 0, w//2, h)), img.crop((w // 2, 0, w, h))


def random_flip(imgs):
    is_flip = random.random() < 0.5
    return [img.transpose(Image.FLIP_LEFT_RIGHT)
            if is_flip else img for img in imgs]


def random_crop(imgs, size):
    _, h, w = imgs[0].size()
    th, tw = size
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return [img[:, y1:y1 + th, x1:x1 + tw] for img in imgs]


def find_images(folder):
    return [g for g in glob.glob(folder + '/*.*')
            if dset.folder.is_image_file(g)]
