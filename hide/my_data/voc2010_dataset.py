import os
import pickle

import torch
import numpy as np
import skimage.io as io
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

from data.base_dataset import BaseDataset
from detail import Detail


class VOC2010Dataset(BaseDataset):

    def initialize(self, opt):
        self.target_size = (opt.fineSize, opt.fineSize)

        self.phase = 'train' if opt.isTrain else 'val'
        self.root = opt.dataroot
        self.img_root = os.path.join(self.root, 'VOCdevkit/VOC2010/JPEGImages')
        self.voc = self._cache_voc()
        self.categories = opt.subjects
        self.meta_images = self.voc.getImgs(cats=self.categories)
        self.transform = T.Compose([
            T.Resize(self.target_size, interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

        mask_file = Image.open('data/mask.png')
        mask_file = mask_file.convert('1')
        self.mask = np.array(mask_file)
        print('{} images: {}'.format(self.phase, len(self)))

    def _cache_voc(self):
        os.makedirs('./cache', exist_ok=True)
        filepath = f'./cache/voc_{self.phase}.cache'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        voc = Detail(os.path.join(self.root, 'trainval_merged.json'), self.img_root, self.phase)
        with open(filepath, 'wb') as f:
            pickle.dump(voc, f, pickle.HIGHEST_PROTOCOL)
        return voc

    def load_example(self, index):
        meta_img = self.meta_images[index]
        img = io.imread(os.path.join(self.img_root, meta_img['file_name']))

        ''' scribble masking '''
        if self.phase == 'val':
            np.random.seed(9487)

        mask = np.zeros(img.shape[:2], dtype='uint8')
        masked_img = img
        bboxes = self.voc.getBboxes(meta_img, self.categories)
        for bbox in bboxes:
            box = bbox['bbox']
            rand_y, rand_x = np.random.randint(self.mask.shape[0]), np.random.randint(self.mask.shape[1])
            if rand_y+box[3]-self.mask.shape[0] > 0: rand_y = self.mask.shape[0]-box[3]
            if rand_x+box[2]-self.mask.shape[1] > 0: rand_x = self.mask.shape[1]-box[2]
            submask = self.mask[rand_y:rand_y+box[3], rand_x:rand_x+box[2]]
            masked_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2], 0] *= submask
            masked_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2], 1] *= submask
            masked_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2], 2] *= submask
            mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] |= submask

        instance_masks = self.voc.getMask(meta_img, cat=self.categories)
        binseg = np.zeros((*img.shape[:2], 1), dtype='uint8')
        binseg[instance_masks != 0] = 1

        return img, masked_img, mask, binseg

    def __len__(self):
        return len(self.meta_images)

    def __getitem__(self, index):
        img, masked_img, mask, binseg = self.load_example(index)
        return {
            'A': self.transform(F.to_pil_image(masked_img)),
            'B': self.transform(F.to_pil_image(img)),
            'A_paths': '',
            'B_paths': '',
            'mask': torch.from_numpy(np.array(F.resize(F.to_pil_image(mask), self.target_size, interpolation=Image.NEAREST))).unsqueeze(0),
            'binseg': torch.from_numpy(np.array(F.resize(F.to_pil_image(binseg), self.target_size, interpolation=Image.NEAREST))).unsqueeze(0).float(),
        }

    def name(self):
        return 'VOC2010Dataset'

