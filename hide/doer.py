import sys
sys.path.append('../')  # noqa

import data
import options.train_options as option

opt = option.TrainOptions().parse()
opt.dataset_mode = 'coco'
dataset = data.create_dataset(opt)
