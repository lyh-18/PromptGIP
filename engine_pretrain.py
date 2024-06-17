import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import util.misc as misc
import util.lr_sched as lr_sched

from loss import GANLoss

ls_gan_loss = GANLoss('lsgan', 1.0, 0.0)
L1_loss = nn.L1Loss()

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    epoch_size=1):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    data_loader_i = iter(data_loader)
    for data_iter_step in metric_logger.log_every(range(epoch_size), print_freq, header):
        (batch, _) = next(data_loader_i)
        # we use a per iteration (instead of per epoch) lr scheduler
        if isinstance(batch, tuple):
            samples, visual_tokens = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = visual_tokens.to(device, non_blocking=True)
        else: # hack for consistency
            samples = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = samples

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            loss_dict = model(samples, visual_tokens)[0]

        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def train_one_epoch_adversarial(model: torch.nn.Module, model_D: torch.nn.Module, model_F: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, optimizer_D: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, loss_scaler_D,
                    log_writer=None,
                    args=None,
                    epoch_size=1,
                    patch_size=16,
                    l_pix_w=0.1,
                    l_fea_w=1,
                    l_gan_w=5e-3):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    optimizer_D.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    data_loader_i = iter(data_loader)
    for data_iter_step in metric_logger.log_every(range(epoch_size), print_freq, header):
        (batch, _) = next(data_loader_i)
        # we use a per iteration (instead of per epoch) lr scheduler
        if isinstance(batch, tuple):
            samples, visual_tokens = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = visual_tokens.to(device, non_blocking=True)
        else: # hack for consistency
            samples = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = samples

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            loss_dict, pred, mask = model(samples, visual_tokens, mask_ratio=args.mask_ratio)

        
        # G
        for p in model_D.parameters():
            p.requires_grad = False
        # pix loss (in loss_dict)
        loss_dict['pix_loss'] = l_pix_w * loss_dict['pix_loss']
        
        # perceptual loss
        input_img1 = visual_tokens[:, 0, :, :, :] 
        target_img1 = visual_tokens[:, 1, :, :, :]
        input_img2 = visual_tokens[:, 2, :, :, :]
        target_img2 = visual_tokens[:, 3, :, :, :]
        
        
        input_img1_patch_num = input_img1.shape[2]//patch_size*input_img1.shape[3]//patch_size
        target_img1_patch_num = target_img1.shape[2]//patch_size*target_img1.shape[3]//patch_size
        input_img2_patch_num = input_img2.shape[2]//patch_size*input_img2.shape[3]//patch_size
        
        pred_target_img1 = pred[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num, :]
        pred_target_img1 = model.module.unpatchify(pred_target_img1).float()
        
        pred_target_img2 = pred[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:, :]
        pred_target_img2 = model.module.unpatchify(pred_target_img2).float()
        
        real_fea1 = model_F(target_img1).detach()
        fake_fea1 = model_F(pred_target_img1)
        l_g_fea1 = l_fea_w * L1_loss(real_fea1, fake_fea1)
        
        real_fea2 = model_F(target_img2).detach()
        fake_fea2 = model_F(pred_target_img2)
        l_g_fea2 = l_fea_w * L1_loss(real_fea2, fake_fea2)
        
        loss_dict['per_loss'] = l_g_fea1 + l_g_fea2
        
        # adversarial loss G
        pred_g_fake1 = model_D(pred_target_img1)
        l_g_gan1 = l_gan_w * ls_gan_loss(pred_g_fake1, True)
        pred_g_fake2 = model_D(pred_target_img2)
        l_g_gan2 = l_gan_w * ls_gan_loss(pred_g_fake2, True)
        
        loss_dict['gan_G'] = l_g_gan1 + l_g_gan2
        
        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            
            
        # D
        for p in model_D.parameters():
            p.requires_grad = True

        loss_D_dict = {}
        l_d_total = 0
        pred_d_real1 = model_D(target_img1)
        pred_d_fake1 = model_D(pred_target_img1.detach())  # detach to avoid BP to G

        l_d_real1 = ls_gan_loss(pred_d_real1, True)
        l_d_fake1 = ls_gan_loss(pred_d_fake1, False)
        l_d_total1 = l_d_real1 + l_d_fake1
        
        pred_d_real2 = model_D(target_img2)
        pred_d_fake2 = model_D(pred_target_img2.detach())  # detach to avoid BP to G

        l_d_real2 = ls_gan_loss(pred_d_real2, True)
        l_d_fake2 = ls_gan_loss(pred_d_fake2, False)
        l_d_total2 = l_d_real2 + l_d_fake2

        l_d_total = l_d_total1 + l_d_total2
        
        loss_D_dict['gan_D'] = l_d_total

        loss_D = torch.stack([loss_D_dict[l] for l in loss_D_dict if 'unscaled' not in l]).sum()
        loss_value_D = loss_D.item()

        if not math.isfinite(loss_value_D):
            print("Loss is {}, stopping training".format(loss_value_D))
            sys.exit(1)

        
        loss_D /= accum_iter
        loss_scaler_D(loss_D, optimizer_D, parameters=model_D.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_D.zero_grad()


        torch.cuda.synchronize()

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{k: v.item() for k, v in loss_D_dict.items()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_value_reduce_D = misc.all_reduce_mean(loss_value_D)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss_G', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_D', loss_value_reduce_D, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_adversarial_CNN_Head(model: torch.nn.Module, model_D: torch.nn.Module, model_F: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, optimizer_D: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, loss_scaler_D,
                    log_writer=None,
                    args=None,
                    epoch_size=1,
                    patch_size=16,
                    l_pix_w=0.1,
                    l_fea_w=1,
                    l_gan_w=5e-3,
                    lr_decay=True):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()
    optimizer_D.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    data_loader_i = iter(data_loader)
    for data_iter_step in metric_logger.log_every(range(epoch_size), print_freq, header):
        (batch, _) = next(data_loader_i)
        # we use a per iteration (instead of per epoch) lr scheduler
        if isinstance(batch, tuple):
            samples, visual_tokens = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = visual_tokens.to(device, non_blocking=True)
        else: # hack for consistency
            samples = batch
            samples = samples.to(device, non_blocking=True)
            visual_tokens = samples

        if data_iter_step % accum_iter == 0:
            if lr_decay:
                lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            else:
                pass
        with torch.cuda.amp.autocast():
            loss_dict, pred, mask, pred_target_img1, pred_target_img2 = model(samples, visual_tokens, mask_ratio=args.mask_ratio)

        
        # G
        for p in model_D.parameters():
            p.requires_grad = False
        # pix loss (in loss_dict)
        loss_dict['pix_loss'] = l_pix_w * loss_dict['pix_loss']
        
        # perceptual loss
        input_img1 = visual_tokens[:, 0, :, :, :] 
        target_img1 = visual_tokens[:, 1, :, :, :]
        input_img2 = visual_tokens[:, 2, :, :, :]
        target_img2 = visual_tokens[:, 3, :, :, :]
        
        pred_target_img1 = pred_target_img1.float()
        
        pred_target_img2 = pred_target_img2.float()
        
        real_fea1 = model_F(target_img1).detach()
        fake_fea1 = model_F(pred_target_img1)
        l_g_fea1 = l_fea_w * L1_loss(real_fea1, fake_fea1)
        
        real_fea2 = model_F(target_img2).detach()
        fake_fea2 = model_F(pred_target_img2)
        l_g_fea2 = l_fea_w * L1_loss(real_fea2, fake_fea2)
        
        loss_dict['per_loss'] = l_g_fea1 + l_g_fea2
        
        # adversarial loss G
        pred_g_fake1 = model_D(pred_target_img1)
        l_g_gan1 = l_gan_w * ls_gan_loss(pred_g_fake1, True)
        pred_g_fake2 = model_D(pred_target_img2)
        l_g_gan2 = l_gan_w * ls_gan_loss(pred_g_fake2, True)
        
        loss_dict['gan_G'] = l_g_gan1 + l_g_gan2
        
        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            
            
        # D
        for p in model_D.parameters():
            p.requires_grad = True

        loss_D_dict = {}
        l_d_total = 0
        pred_d_real1 = model_D(target_img1)
        pred_d_fake1 = model_D(pred_target_img1.detach())  # detach to avoid BP to G

        l_d_real1 = ls_gan_loss(pred_d_real1, True)
        l_d_fake1 = ls_gan_loss(pred_d_fake1, False)
        l_d_total1 = l_d_real1 + l_d_fake1
        
        pred_d_real2 = model_D(target_img2)
        pred_d_fake2 = model_D(pred_target_img2.detach())  # detach to avoid BP to G

        l_d_real2 = ls_gan_loss(pred_d_real2, True)
        l_d_fake2 = ls_gan_loss(pred_d_fake2, False)
        l_d_total2 = l_d_real2 + l_d_fake2

        l_d_total = l_d_total1 + l_d_total2
        
        loss_D_dict['gan_D'] = l_d_total

        loss_D = torch.stack([loss_D_dict[l] for l in loss_D_dict if 'unscaled' not in l]).sum()
        loss_value_D = loss_D.item()

        if not math.isfinite(loss_value_D):
            print("Loss is {}, stopping training".format(loss_value_D))
            sys.exit(1)

        
        loss_D /= accum_iter
        loss_scaler_D(loss_D, optimizer_D, parameters=model_D.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_D.zero_grad()


        torch.cuda.synchronize()

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{k: v.item() for k, v in loss_D_dict.items()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_value_reduce_D = misc.all_reduce_mean(loss_value_D)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss_G', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_D', loss_value_reduce_D, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate(model, data_loader, device, epoch, log_writer, args):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, visual_tokens = batch
        samples = samples.to(device, non_blocking=True)
        visual_tokens = visual_tokens.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss_dict, _, _ = model(samples, visual_tokens, mask_ratio=args.mask_ratio)

        loss = torch.stack([loss_dict[l] for l in loss_dict if 'unscaled' not in l]).sum()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats for val:", metric_logger)
    return {'val_' + k: meter.global_avg for k, meter in metric_logger.meters.items()}
