import os
import random

import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator, load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator as DataGenerator


class ImageDataGenerator(DataGenerator):

    def flow_from_directory(self, directory,
                            target_size=(256, 256), color_mode='rgb',
                            batch_size=32, shuffle=True, seed=None,
                            follow_links=False):
        return ImageFolderIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            follow_links=follow_links)


class ImageFolderIterator(Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 follow_links=False):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        self.samples = 0
        self.files = []
        for ent in os.scandir(directory):
            is_valid = False
            for extension in white_list_formats:
                if ent.name.lower().endswith('.' + extension):
                    is_valid = True
                    break
            if is_valid:
                self.samples += 1
                self.files.append(ent.path)
        print('Found %d images.' % (self.samples))

        super().__init__(self.samples, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        batch_a = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        batch_b = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'

        # build batch of image data
        for i, j in enumerate(index_array):
            img = load_img(self.files[j],
                           grayscale=grayscale)

            img = resize(img, [286 * 2, 286])
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            x = normalize(x)

            a, b = split_ab(x)
            a = random_crop(a, self.target_size)
            b = random_crop(b, self.target_size)

            batch_a[i], batch_b[i] = a, b

        return batch_a, batch_b


def resize(img, shape):
    return img.resize(shape)


def normalize(img):
    return img * 2 - 1


def split_ab(img):
    _, w, _ = img.shape
    return img[:, :w // 2, :], img[:, w // 2:, :]


def random_crop(img, shape):
    h, w, _ = img.shape
    th, tw = shape
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return img[x1:x1 + tw, y1:y1 + th, :]
