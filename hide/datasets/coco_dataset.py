import math
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

from pycocotools.coco import COCO


class CocoDataLoader(BaseDataset):

    def initialize(self, opt):
        self.target_size = (opt.fineSize, opt.fineSize)

        self.phase = 'train' if opt.isTrain else 'val'
        self.root = opt.dataroot
        self.coco = self._cache_coco()
        self.category_ids = self.coco.getCatIds(catNms=opt.subjects)
        self.image_ids = sorted(self.coco.getImgIds(catIds=self.category_ids))
        self.transform = T.Compose([
            T.Resize(self.target_size, interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        print('{} images: {}'.format(self.phase, len(self)))

    def _cache_coco(self):
        os.makedirs('./cache', exist_ok=True)
        filepath = f'./cache/coco_{self.phase}.cache'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        coco = COCO(os.path.join(self.root, f'annotations/instances_{self.phase}2017.json'))
        with open(filepath, 'wb') as f:
            pickle.dump(coco, f, pickle.HIGHEST_PROTOCOL)
        return coco

    def load_example(self, index):
        example = self.coco.loadImgs([self.image_ids[index]])[0]

        img_path = os.path.join(self.root, f'{self.phase}2017', example['file_name'])
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=example['id'],
                                                             catIds=self.category_ids, iscrowd=None))
        img = np.array(Image.open(img_path).convert('RGB'))
        seg = np.array([self.coco.annToMask(ann) for ann in annotations])

        return img, seg

    def get_center_masked_pair(self, img, seg):
        origin_img = img.copy()
        masked_img = img.copy()
        mask = np.zeros((*img.shape[:2], 1), dtype='uint8')
        binseg = np.zeros((*img.shape[:2], 1), dtype='uint8')

        w, h, _ = img.shape
        min_x, max_x = w // 4, w // 4 * 3
        min_y, max_y = h // 4, h // 4 * 3

        masked_img[min_x:max_x, min_y:max_y, :] = 0
        mask[min_x:max_x, min_y:max_y] = 1

        for k in range(seg.shape[0]):
            binseg[seg[k] == 1] = 1

        return origin_img, masked_img, mask, binseg

    def get_masked_pair(self, img, seg):
        origin_img = img.copy()
        masked_img = img.copy()
        mask = np.zeros((*img.shape[:2], 1), dtype='uint8')
        binseg = np.zeros((*img.shape[:2], 1), dtype='uint8')

        if self.phase == 'val':
            np.random.seed(9487)

        for k in range(seg.shape[0]):

            mask_region=np.where(seg[k]==1)
            mask_region=np.array(mask_region)

            if len(mask_region[0]) == 0:
                continue

            half_x=(max(mask_region[0])-min(mask_region[0]))/2
            half_y=(max(mask_region[1])-min(mask_region[1]))/2
            rand_point=np.random.randint(mask_region[0].shape[0])
            center=[mask_region[0][rand_point], mask_region[1][rand_point]]

            max_x = math.floor(center[0]+half_x/2)
            min_x = math.floor(center[0]-half_x/2)
            max_y = math.floor(center[1]+half_y/2)
            min_y = math.floor(center[1]-half_y/2)

            if max_x>masked_img.shape[0]: max_x = masked_img.shape[0]
            if min_x<0: min_x = 0
            if max_y>masked_img.shape[1]: max_y = masked_img.shape[1]
            if min_y<0: min_y = 0

            masked_img[min_x:max_x, min_y:max_y, :] = 0
            mask[min_x:max_x, min_y:max_y] = 1
            binseg[seg[k] == 1] = 1

        return origin_img, masked_img, mask, binseg

    def __getitem__(self, index):
        img, seg = self.load_example(index)
        # origin, masked, mask, binseg = self.get_masked_pair(img, seg)
        origin, masked, mask, binseg = self.get_center_masked_pair(img, seg)
        return {
            'A': self.transform(F.to_pil_image(masked)),
            'B': self.transform(F.to_pil_image(origin)),
            'A_paths': '',
            'B_paths': '',
            'mask': torch.from_numpy(np.array(F.resize(F.to_pil_image(mask), self.target_size, interpolation=Image.NEAREST))).unsqueeze(0).float(),
            'binseg': torch.from_numpy(np.array(F.resize(F.to_pil_image(binseg), self.target_size, interpolation=Image.NEAREST))).unsqueeze(0).float(),
        }

    def __len__(self):
        return len(self.image_ids)

    def name(self):
        return 'CocoDataset'
