CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=4321 train_PromptGIP.py \
--model mae_vit_large_patch16_input256 \
--input_size 256 --batch_size 6 --mask_ratio 0.75 \
--warmup_epochs 5 --epochs 50 --blr 1e-4 \
--save_ckpt_freq 5 \
--output_dir experiments/PromptGIP \
--data_path /nvme/liuyihao/DATA/ImageNet/images/train \
--data_path_val /nvme/liuyihao/DATA/Common528/Image_clean_256x256 | tee -a experiments/PromptGIP_log.txt
