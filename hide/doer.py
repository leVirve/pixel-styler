import sys
sys.path.append('../')  # noqa

import data
import options.train_options as option

opt = option.TrainOptions().parse()
opt.dataset_mode = 'voc2010'

dataset = data.create_dataset(opt)
