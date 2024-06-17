import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import datasets
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae_PromptGIP_CNN_Head
from engine_pretrain import train_one_epoch
import dataset.lowlevel_PromtGIP_dataloader as lowlevel_prompt_dataloader
from evaluation_prompt import *


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--ckpt', help='resume from checkpoint')
    
    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--break_after_epoch', type=int, metavar='N', help='break training after X epochs, to tune hyperparams and avoid messing with training schedule')


    # Dataset parameters
    parser.add_argument('--data_path', default='/shared/yossi_gandelsman/arxiv/arxiv_data/', type=str,
                        help='dataset path')
    parser.add_argument('--data_path_val', default='/shared/yossi_gandelsman/arxiv/arxiv_data/', type=str,
                        help='val dataset path')
    parser.add_argument('--imagenet_percent', default=1, type=float)
    parser.add_argument('--subsample', action='store_true')
    parser.set_defaults(subsample=False)
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    args.second_input_size = 224
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    ITS_path = '/nvme/liuyihao/DATA/Dehaze/ITS'
    LOL_path = '/nvme/liuyihao/DATA/LOLdataset/our485'
    Rain13K_path = '/nvme/liuyihao/DATA/Derain/train/Rain13K'
    FiveK_path = '/nvme/liuyihao/DATA/MIT-fivek'
    LLF_path = '/nvme/liuyihao/DATA/MIT-fivek'

    dataset_train = lowlevel_prompt_dataloader.DatasetLowlevel_Train(dataset_path=args.data_path, 
                                                                     input_size=args.input_size,
                                                                     phase = 'train',
                                                                     ITS_path=ITS_path, 
                                                                     LOL_path=LOL_path,
                                                                     Rain13K_path=Rain13K_path,
                                                                     FiveK_path=FiveK_path,
                                                                     LLF_path=LLF_path
                                                                     )
    
    dataset_val = lowlevel_prompt_dataloader.DatasetLowlevel_Val(dataset_path=args.data_path_val,
                                                                 input_size=args.input_size)
    
    dataset_path_HQ_SOTS = '/nvme/liuyihao/DATA/Dehaze/SOTS/indoor/nyuhaze500/gt'
    dataset_path_LQ_SOTS = '/nvme/liuyihao/DATA/Dehaze/SOTS/indoor/nyuhaze500/hazy'
    dataset_val_SOTS = lowlevel_prompt_dataloader.DatasetLowlevel_Customized_Val(dataset_path_HQ=dataset_path_HQ_SOTS, 
                                                                                 dataset_path_LQ=dataset_path_LQ_SOTS, 
                                                                                 dataset_type='SOTS',
                                                                                 data_len=100)
    
    dataset_path_HQ_LOL = '/nvme/liuyihao/DATA/LOLdataset/eval15/high'
    dataset_path_LQ_LOL = '/nvme/liuyihao/DATA/LOLdataset/eval15/low'
    dataset_val_LOL = lowlevel_prompt_dataloader.DatasetLowlevel_Customized_Val(dataset_path_HQ=dataset_path_HQ_LOL, 
                                                                                dataset_path_LQ=dataset_path_LQ_LOL, 
                                                                                dataset_type='LOL',
                                                                                data_len=100)

    dataset_path_HQ_Test100 = '/nvme/liuyihao/DATA/Derain/test/Test100/target'
    dataset_path_LQ_Test100 = '/nvme/liuyihao/DATA/Derain/test/Test100/input'
    dataset_val_Test100 = lowlevel_prompt_dataloader.DatasetLowlevel_Customized_Val(dataset_path_HQ=dataset_path_HQ_Test100, 
                                                                                dataset_path_LQ=dataset_path_LQ_Test100, 
                                                                                dataset_type='Test100',
                                                                                data_len=100)
    
    
    dataset_path_HQ_LLF = '/nvme/liuyihao/DATA/MIT-fivek/expert_C_LLF_GT_test'
    dataset_path_LQ_LLF = '/nvme/liuyihao/DATA/MIT-fivek/expert_C_test'
    dataset_val_LLF = lowlevel_prompt_dataloader.DatasetLowlevel_Customized_Val(dataset_path_HQ=dataset_path_HQ_LLF, 
                                                                                dataset_path_LQ=dataset_path_LQ_LLF, 
                                                                                dataset_type='LLF',
                                                                                data_len=100)

    # single | mix | mismatch_single | UDC_Toled | UDC_Poled | Rain100L | Rain100H | Test100 | Test1200
    # SOTS | mismatch_mix
    test_folder = 'single' 
    data_path = os.path.join('/home/liuyihao/visual_prompting/prompt_generalization_testset_256', test_folder)
    dataset_val_single = lowlevel_prompt_dataloader.DatasetLowlevel_Customized_Test_DirectLoad_Triplet(data_path)
    
    test_folder = 'mix' 
    data_path = os.path.join('/home/liuyihao/visual_prompting/prompt_generalization_testset_256', test_folder)
    dataset_val_mix = lowlevel_prompt_dataloader.DatasetLowlevel_Customized_Test_DirectLoad_Triplet(data_path)
    
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    data_loader_val_SOTS = torch.utils.data.DataLoader(
        dataset_val_SOTS,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    data_loader_val_LOL = torch.utils.data.DataLoader(
        dataset_val_LOL,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    data_loader_val_Test100 = torch.utils.data.DataLoader(
        dataset_val_Test100,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    
    
    data_loader_val_LLF = torch.utils.data.DataLoader(
        dataset_val_LLF,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    
    data_loader_val_single = torch.utils.data.DataLoader(
        dataset_val_single,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    
    data_loader_val_mix = torch.utils.data.DataLoader(
        dataset_val_mix,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # define the model
    model = models_mae_PromptGIP_CNN_Head.__dict__[args.model]()
    
    if args.ckpt:
        print('Load pretrained model: ', args.ckpt)
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)

    model.to(device)
    epoch_size = len(dataset_train)
    print(f'epoch_size is {epoch_size}')
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    base_lr = (args.lr * 256 / eff_batch_size)
    print("base lr: %.2e" % base_lr)
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    for k, v in model_without_ddp.named_parameters():
        if 'vae' in k:
            v.requires_grad = False

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            epoch_size=epoch_size // eff_batch_size
        )

        val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val, epoch, args.output_dir, mode='val_test', patch_size=16)
        val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val_SOTS, epoch, args.output_dir, mode='val_test_SOTS', patch_size=16)
        val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val_LOL, epoch, args.output_dir, mode='val_test_LOL', patch_size=16)
        val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val_Test100, epoch, args.output_dir, mode='val_test_Test100', patch_size=16)
        val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val_LLF, epoch, args.output_dir, mode='val_test_LLF', patch_size=16)
        val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val_single, epoch, args.output_dir, mode='val_test_single', patch_size=16)
        val_on_master_CNN_Head(model_without_ddp.cuda(), data_loader_val_mix, epoch, args.output_dir, mode='val_test_mix', patch_size=16)
        
        if args.output_dir and (epoch % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # if misc.is_main_process():
    #     run.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
