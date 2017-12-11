import math
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

import onegan
from pycocotools.coco import COCO


class CocoDataLoader(onegan.io.loader.BaseDastaset):
    name = 'CocoDataLoader'

    def __init__(self, opt):
        subjects = ['cat']
        target_size = (opt.fineSize, opt.fineSize)

        self.phase = 'train' if opt.isTrain else 'val'
        self.root = opt.dataroot
        self.coco = self._cache_coco(subjects)
        self.category_ids = self.coco.getCatIds(catNms=subjects)
        self.image_ids = sorted(self.coco.getImgIds(catIds=self.category_ids))
        self.transform = T.Compose([
            T.Resize(target_size, interpolation=Image.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        print('{} images: {}'.format(self.phase, len(self)))

    def _cache_coco(self, subjects):
        os.makedirs('./cache', exist_ok=True)
        filepath = f'./cache/coco_{self.phase}_{"-".join(subjects)}.cache'
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

    def get_masked_pair(self, img, seg):
        origin_img = img.copy()
        masked_img = img.copy()
        mask = np.zeros(img.shape, dtype='uint8')

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
            mask[min_x:max_x, min_y:max_y, :] = 1

        return origin_img, masked_img, mask

    def __getitem__(self, index):
        img, seg = self.load_example(index)
        origin, masked, mask = self.get_masked_pair(img, seg)
        return {
            'A': self.transform(F.to_pil_image(masked)),
            'B': self.transform(F.to_pil_image(origin)),
            'A_paths': '',
            'B_paths': '',
            'mask': torch.from_numpy(np.array(F.resize(F.to_pil_image(mask), (256, 256), interpolation=Image.NEAREST))).permute(2, 0, 1),
        }

    def __len__(self):
        return len(self.image_ids)


if __name__ == '__main__':
    dataloader = CocoDataLoader(data_folder='/archive/datasets/mscoco', subjects=['person'], target_size=(256, 256))

    img, seg = dataloader.load_example(80)
    origin_img, masked_img = dataloader.get_masked_pair(img, seg)
