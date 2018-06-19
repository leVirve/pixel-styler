import os
import scipy.misc

from data.loader import CustomDataLoader
from training import create_trainer
from options import TestOptions


if __name__ == '__main__':
    parser = TestOptions()
    parser.parser.add_argument('--subjects', type=str, nargs='+')
    opt = parser.parse()
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    AtoB = opt.which_direction == 'AtoB'

    dataloader = CustomDataLoader(opt)
    trainer = create_trainer(opt)
    output_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.subjects[0]}_{opt.which_epoch}')
    os.makedirs(output_dir, exist_ok=True)

    for i, data in enumerate(dataloader):
        if i >= opt.how_many:
            break
        visuals = trainer.test(data)
        img_path = data['A_paths' if AtoB else 'B_paths'][0] or f'{i}.png'
        print(f'process image... {img_path}')
        for name, img in visuals.items():
            scipy.misc.imsave(
                os.path.join(output_dir, name + img_path),
                img.squeeze().permute(1, 2, 0).cpu().numpy())
