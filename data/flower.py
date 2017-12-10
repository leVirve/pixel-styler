import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, make_dataset
from PIL import Image


class BMVCFlower(BaseDataset):
    name = 'BMVCFlower'

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = opt.dataroot

        self.A_paths = sorted(make_dataset(self.dir_AB + '/inputs/scribble/'))
        self.B_paths = sorted(make_dataset(self.dir_AB + '/gt/'))

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert('RGB')
        B = Image.open(self.B_paths[index]).convert('RGB')
        return {'A': self.transform(A), 'B': self.transform(B),
                'A_paths': self.A_paths[index], 'B_paths': self.B_paths[index]}

    def __len__(self):
        return len(self.A_paths)
