train_binseg:
	python train.py --dataroot /archive/datasets/mscoco --which_direction AtoB --model pix2pix --which_model_netG unet_128 --lambda_A 10 --dataset_mode mscoco --norm batch --pool_size 0 --batchSize 32 --fineSize 128 --loadSize 187 --subjects cat --name cat-nc4-A --input_nc 4 --niter 200 --niter_decay 300 --gpu_ids 0,1
t87:
	python train.py --dataroot /archive/datasets/mscoco --which_direction AtoB --model pix2pix --which_model_netG unet_128 --lambda_A 10 --dataset_mode mscoco --norm batch --pool_size 0 --batchSize 32 --fineSize 128 --loadSize 187 --subjects cat --name d87 --input_nc 4 --niter 200 --niter_decay 300 --gpu_ids 0,1
test:
	python test.py  --dataroot /archive/datasets/mscoco --which_direction AtoB --model pix2pix --which_model_netG unet_128 --dataset_mode mscoco --norm batch --name coco_pix2pix_$(model) --subjects $(data) --how_many 9999999

# proper data for training
# scribble corruptions

dev:
	python main.py --dataroot /archive/datasets/mscoco --which_direction AtoB --model pix2pix --which_model_netG unet_128 --lambda_A 10 --dataset_mode mscoco --norm batch --pool_size 0 --batchSize 16 --fineSize 128 --loadSize 128 --subjects cat --name d87 --input_nc 3 --niter 200 --niter_decay 300 --gpu_ids 0,1

devc:
	python main.py --dataroot /archive/datasets/mscoco --which_direction AtoB --model cyclegan --which_model_netG unet_128 --lambda_A 10 --dataset_mode mscoco --norm batch --pool_size 0 --batchSize 16 --fineSize 128 --loadSize 128 --subjects cat --name d87c --input_nc 3 --niter 200 --niter_decay 300 --gpu_ids 0,1

dev_m:
	python main.py --dataroot ./data/maps --which_direction AtoB --model pix2pix --which_model_netG unet_128 --lambda_A 10 --dataset_mode aligned --norm batch --pool_size 0 --batchSize 1 --fineSize 128 --loadSize 256 --name maps --input_nc 3 --niter 200 --niter_decay 300 --gpu_ids 0,1
