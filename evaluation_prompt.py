import torch
import numpy as np
import cv2
import os
import torch.distributed as dist
import copy
import math
import torch.nn.functional as F

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def val_on_master(model, data_loader_val, epoch, save_path, mode, patch_size):
    if is_main_process():
        val_model(model, data_loader_val, epoch, save_path, mode, patch_size)



def val_model(model, data_loader_val, epoch, save_path, mode='val_pretrain', patch_size=16):
    if mode == 'val_pretrain':
        mask_ratio = 0.75
        train_mode = True
    elif mode == 'val_test' or 'test' in mode:
        mask_ratio = 1
        train_mode = False
        
    idx = 0
    for batch, deg_type in data_loader_val:
        idx += 1
        #print('idx: ', idx, len(batch), deg_type)
        input_img1 = batch['input_query_img1'].cuda()
        target_img1 = batch['target_img1'].cuda()
        input_img2 = batch['input_query_img2'].cuda()
        #print(batch['target_img2'])
        if batch['target_img2'][0] != 'None':
            target_img2 = batch['target_img2'].cuda()
            target_img2_flag = True
        else:
            target_img2 = input_img2
            target_img2_flag = False
        input_sequence = [input_img1, target_img1, input_img2, target_img2]
        model.eval()
        loss, pred, mask = model(imgs=input_sequence, mask_ratio=mask_ratio, input_is_list=True, train_mode=train_mode)
        
        input_img1_patch_num = input_img1.shape[2]//patch_size*input_img1.shape[3]//patch_size
        target_img1_patch_num = target_img1.shape[2]//patch_size*target_img1.shape[3]//patch_size
        input_img2_patch_num = input_img2.shape[2]//patch_size*input_img2.shape[3]//patch_size
        target_img2_patch_num = target_img2.shape[2]//patch_size*target_img2.shape[3]//patch_size

        input_img1_mask = mask[:, 0:input_img1_patch_num]
        target_img1_mask = mask[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num]
        input_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num:input_img1_patch_num+target_img1_patch_num+input_img2_patch_num]
        target_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:]

        target_img1_mask = target_img1_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img1_mask = model.unpatchify(target_img1_mask)  # 1 is removing, 0 is keeping
        target_img1_mask = torch.einsum('nchw->nhwc', target_img1_mask).detach().cpu()
        target_img2_mask = target_img2_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img2_mask = model.unpatchify(target_img2_mask)  # 1 is removing, 0 is keeping
        target_img2_mask = torch.einsum('nchw->nhwc', target_img2_mask).detach().cpu()

        pred_target_img1 = pred[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num, :]
        pred_target_img1 = model.unpatchify(pred_target_img1)  # 1 is removing, 0 is keeping
        pred_target_img1 = torch.einsum('nchw->nhwc', pred_target_img1).detach().cpu()

        pred_target_img2 = pred[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:, :]
        pred_target_img2 = model.unpatchify(pred_target_img2)  # 1 is removing, 0 is keeping
        pred_target_img2 = torch.einsum('nchw->nhwc', pred_target_img2).detach().cpu()

        input_img1 = torch.einsum('nchw->nhwc', input_img1)
        target_img1 = torch.einsum('nchw->nhwc', target_img1)
        input_img2 = torch.einsum('nchw->nhwc', input_img2)
        target_img2 = torch.einsum('nchw->nhwc', target_img2)

        input_img1 = (torch.clip((input_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        input_img2 = (torch.clip((input_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)

        target_img1 = (torch.clip((target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img1 = target_img1 * (1 - target_img1_mask)
        target_img2 = (torch.clip((target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img2 = target_img2 * (1 - target_img2_mask)


        pred_target_img1 = (torch.clip((pred_target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img1 = pred_target_img1 * (target_img1_mask)
        pred_target_final_img1 = target_img1 * (1 - target_img1_mask) + pred_target_img1 * (target_img1_mask)
        pred_target_img2 = (torch.clip((pred_target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img2 = pred_target_img2 * (target_img2_mask)
        pred_target_final_img2 = target_img2 * (1 - target_img2_mask) + pred_target_img2 * (target_img2_mask)

        input_img1 = input_img1[0].numpy().astype(np.uint8)
        input_img2 = input_img2[0].numpy().astype(np.uint8)

        target_img1 = target_img1[0].numpy().astype(np.uint8)
        target_img2 = target_img2[0].numpy().astype(np.uint8)
        target_mask_img1 = target_mask_img1[0].numpy().astype(np.uint8)
        target_mask_img2 = target_mask_img2[0].numpy().astype(np.uint8)

        pred_target_img1 = pred_target_img1[0].numpy().astype(np.uint8)
        pred_target_mask_img1 = pred_target_mask_img1[0].numpy().astype(np.uint8)
        pred_target_final_img1 = pred_target_final_img1[0].numpy().astype(np.uint8)
        pred_target_img2 = pred_target_img2[0].numpy().astype(np.uint8)
        pred_target_mask_img2 = pred_target_mask_img2[0].numpy().astype(np.uint8)
        pred_target_final_img2 = pred_target_final_img2[0].numpy().astype(np.uint8)
        
        
        save_path_img = os.path.join(save_path, 'vis', mode, 'Epoch{}'.format(epoch))
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        
        
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img1_{}.png'.format(idx, deg_type[0])), input_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img1.png'.format(idx)), target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img1.png'.format(idx)), target_mask_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img1.png'.format(idx)), pred_target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img1.png'.format(idx)), pred_target_mask_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img1.png'.format(idx)), pred_target_final_img1)

        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img2_{}.png'.format(idx, deg_type[0])), input_img2)
        
        if target_img2_flag:
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img2.png'.format(idx)), target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img2.png'.format(idx)), target_mask_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img2.png'.format(idx)), pred_target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img2.png'.format(idx)), pred_target_mask_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img2.png'.format(idx)), pred_target_final_img2)


def val_on_master_CNN_Head(model, data_loader_val, epoch, save_path, mode, patch_size):
    if is_main_process():
        val_model_CNN_Head(model, data_loader_val, epoch, save_path, mode, patch_size)

def val_model_CNN_Head(model, data_loader_val, epoch, save_path, mode='val_pretrain', patch_size=16):
    if mode == 'val_pretrain':
        mask_ratio = 0.75
        train_mode = True
    elif mode == 'val_test' or 'test' in mode:
        mask_ratio = 1
        train_mode = False
        
    idx = 0
    for batch, deg_type in data_loader_val:
        idx += 1
        #print('idx: ', idx, len(batch), deg_type)
        input_img1 = batch['input_query_img1'].cuda()
        target_img1 = batch['target_img1'].cuda()
        input_img2 = batch['input_query_img2'].cuda()
        #print(batch['target_img2'])
        if batch['target_img2'][0] != 'None':
            target_img2 = batch['target_img2'].cuda()
            target_img2_flag = True
        else:
            target_img2 = input_img2
            target_img2_flag = False
        input_sequence = [input_img1, target_img1, input_img2, target_img2]
        model.eval()
        loss, pred, mask, pred_target_img1, pred_target_img2 = model(imgs=input_sequence, mask_ratio=mask_ratio, input_is_list=True, train_mode=train_mode)
        
        input_img1_patch_num = input_img1.shape[2]//patch_size*input_img1.shape[3]//patch_size
        target_img1_patch_num = target_img1.shape[2]//patch_size*target_img1.shape[3]//patch_size
        input_img2_patch_num = input_img2.shape[2]//patch_size*input_img2.shape[3]//patch_size
        target_img2_patch_num = target_img2.shape[2]//patch_size*target_img2.shape[3]//patch_size

        input_img1_mask = mask[:, 0:input_img1_patch_num]
        target_img1_mask = mask[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num]
        input_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num:input_img1_patch_num+target_img1_patch_num+input_img2_patch_num]
        target_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:]

        target_img1_mask = target_img1_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img1_mask = model.unpatchify(target_img1_mask)  # 1 is removing, 0 is keeping
        target_img1_mask = torch.einsum('nchw->nhwc', target_img1_mask).detach().cpu()
        target_img2_mask = target_img2_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img2_mask = model.unpatchify(target_img2_mask)  # 1 is removing, 0 is keeping
        target_img2_mask = torch.einsum('nchw->nhwc', target_img2_mask).detach().cpu()

        pred_target_img1 = torch.einsum('nchw->nhwc', pred_target_img1).detach().cpu()
        pred_target_img2 = torch.einsum('nchw->nhwc', pred_target_img2).detach().cpu()

        input_img1 = torch.einsum('nchw->nhwc', input_img1)
        target_img1 = torch.einsum('nchw->nhwc', target_img1)
        input_img2 = torch.einsum('nchw->nhwc', input_img2)
        target_img2 = torch.einsum('nchw->nhwc', target_img2)

        input_img1 = (torch.clip((input_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        input_img2 = (torch.clip((input_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)

        target_img1 = (torch.clip((target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img1 = target_img1 * (1 - target_img1_mask)
        target_img2 = (torch.clip((target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img2 = target_img2 * (1 - target_img2_mask)


        pred_target_img1 = (torch.clip((pred_target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img1 = pred_target_img1 * (target_img1_mask)
        pred_target_final_img1 = target_img1 * (1 - target_img1_mask) + pred_target_img1 * (target_img1_mask)
        pred_target_img2 = (torch.clip((pred_target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img2 = pred_target_img2 * (target_img2_mask)
        pred_target_final_img2 = target_img2 * (1 - target_img2_mask) + pred_target_img2 * (target_img2_mask)

        input_img1 = input_img1[0].numpy().astype(np.uint8)
        input_img2 = input_img2[0].numpy().astype(np.uint8)

        target_img1 = target_img1[0].numpy().astype(np.uint8)
        target_img2 = target_img2[0].numpy().astype(np.uint8)
        target_mask_img1 = target_mask_img1[0].numpy().astype(np.uint8)
        target_mask_img2 = target_mask_img2[0].numpy().astype(np.uint8)

        pred_target_img1 = pred_target_img1[0].numpy().astype(np.uint8)
        pred_target_mask_img1 = pred_target_mask_img1[0].numpy().astype(np.uint8)
        pred_target_final_img1 = pred_target_final_img1[0].numpy().astype(np.uint8)
        pred_target_img2 = pred_target_img2[0].numpy().astype(np.uint8)
        pred_target_mask_img2 = pred_target_mask_img2[0].numpy().astype(np.uint8)
        pred_target_final_img2 = pred_target_final_img2[0].numpy().astype(np.uint8)
        
        
        save_path_img = os.path.join(save_path, 'vis', mode, 'Epoch{}'.format(epoch))
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        
        
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img1_{}.png'.format(idx, deg_type[0])), input_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img1.png'.format(idx)), target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img1.png'.format(idx)), target_mask_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img1.png'.format(idx)), pred_target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img1.png'.format(idx)), pred_target_mask_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img1.png'.format(idx)), pred_target_final_img1)

        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img2_{}.png'.format(idx, deg_type[0])), input_img2)
        
        if target_img2_flag:
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img2.png'.format(idx)), target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img2.png'.format(idx)), target_mask_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img2.png'.format(idx)), pred_target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img2.png'.format(idx)), pred_target_mask_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img2.png'.format(idx)), pred_target_final_img2)

def val_model_CNN_Head_ExtraSaveOutput(model, data_loader_val, epoch, save_path, mode='val_pretrain', patch_size=16):
    if mode == 'val_pretrain':
        mask_ratio = 0.75
        train_mode = True
    elif mode == 'val_test' or 'test' in mode:
        mask_ratio = 1
        train_mode = False
        
    idx = 0
    for batch, deg_type in data_loader_val:
        idx += 1
        #print('idx: ', idx, len(batch), deg_type)
        input_img1 = batch['input_query_img1'].cuda()
        target_img1 = batch['target_img1'].cuda()
        input_img2 = batch['input_query_img2'].cuda()
        input_query_img2_path = batch['input_query_img2_path'][0]
   
        #print(batch['target_img2'])
        if batch['target_img2'][0] != 'None':
            target_img2 = batch['target_img2'].cuda()
            target_img2_flag = True
        else:
            target_img2 = input_img2
            target_img2_flag = False
        input_sequence = [input_img1, target_img1, input_img2, target_img2]
        model.eval()
        loss, pred, mask, pred_target_img1, pred_target_img2 = model(imgs=input_sequence, mask_ratio=mask_ratio, input_is_list=True, train_mode=train_mode)
        
        input_img1_patch_num = input_img1.shape[2]//patch_size*input_img1.shape[3]//patch_size
        target_img1_patch_num = target_img1.shape[2]//patch_size*target_img1.shape[3]//patch_size
        input_img2_patch_num = input_img2.shape[2]//patch_size*input_img2.shape[3]//patch_size
        target_img2_patch_num = target_img2.shape[2]//patch_size*target_img2.shape[3]//patch_size

        input_img1_mask = mask[:, 0:input_img1_patch_num]
        target_img1_mask = mask[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num]
        input_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num:input_img1_patch_num+target_img1_patch_num+input_img2_patch_num]
        target_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:]

        target_img1_mask = target_img1_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img1_mask = model.unpatchify(target_img1_mask)  # 1 is removing, 0 is keeping
        target_img1_mask = torch.einsum('nchw->nhwc', target_img1_mask).detach().cpu()
        target_img2_mask = target_img2_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img2_mask = model.unpatchify(target_img2_mask)  # 1 is removing, 0 is keeping
        target_img2_mask = torch.einsum('nchw->nhwc', target_img2_mask).detach().cpu()

        pred_target_img1 = torch.einsum('nchw->nhwc', pred_target_img1).detach().cpu()
        pred_target_img2 = torch.einsum('nchw->nhwc', pred_target_img2).detach().cpu()

        input_img1 = torch.einsum('nchw->nhwc', input_img1)
        target_img1 = torch.einsum('nchw->nhwc', target_img1)
        input_img2 = torch.einsum('nchw->nhwc', input_img2)
        target_img2 = torch.einsum('nchw->nhwc', target_img2)

        input_img1 = (torch.clip((input_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        input_img2 = (torch.clip((input_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)

        target_img1 = (torch.clip((target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img1 = target_img1 * (1 - target_img1_mask)
        target_img2 = (torch.clip((target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img2 = target_img2 * (1 - target_img2_mask)


        pred_target_img1 = (torch.clip((pred_target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img1 = pred_target_img1 * (target_img1_mask)
        pred_target_final_img1 = target_img1 * (1 - target_img1_mask) + pred_target_img1 * (target_img1_mask)
        pred_target_img2 = (torch.clip((pred_target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img2 = pred_target_img2 * (target_img2_mask)
        pred_target_final_img2 = target_img2 * (1 - target_img2_mask) + pred_target_img2 * (target_img2_mask)

        input_img1 = input_img1[0].numpy().astype(np.uint8)
        input_img2 = input_img2[0].numpy().astype(np.uint8)

        target_img1 = target_img1[0].numpy().astype(np.uint8)
        target_img2 = target_img2[0].numpy().astype(np.uint8)
        target_mask_img1 = target_mask_img1[0].numpy().astype(np.uint8)
        target_mask_img2 = target_mask_img2[0].numpy().astype(np.uint8)

        pred_target_img1 = pred_target_img1[0].numpy().astype(np.uint8)
        pred_target_mask_img1 = pred_target_mask_img1[0].numpy().astype(np.uint8)
        pred_target_final_img1 = pred_target_final_img1[0].numpy().astype(np.uint8)
        pred_target_img2 = pred_target_img2[0].numpy().astype(np.uint8)
        pred_target_mask_img2 = pred_target_mask_img2[0].numpy().astype(np.uint8)
        pred_target_final_img2 = pred_target_final_img2[0].numpy().astype(np.uint8)
        
        
        save_path_img = os.path.join(save_path, 'vis', mode, 'Epoch{}'.format(epoch))
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        
        
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img1_{}.png'.format(idx, deg_type[0])), input_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img1.png'.format(idx)), target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img1.png'.format(idx)), target_mask_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img1.png'.format(idx)), pred_target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img1.png'.format(idx)), pred_target_mask_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img1.png'.format(idx)), pred_target_final_img1)

        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img2_{}.png'.format(idx, deg_type[0])), input_img2)
        
        if target_img2_flag:
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img2.png'.format(idx)), target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img2.png'.format(idx)), target_mask_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img2.png'.format(idx)), pred_target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img2.png'.format(idx)), pred_target_mask_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img2.png'.format(idx)), pred_target_final_img2)
        
        save_path_img = os.path.join(save_path, 'vis', mode, epoch)
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        img_name = input_query_img2_path.split('/')[-1]
        cv2.imwrite(os.path.join(save_path_img, '{}'.format(img_name)), pred_target_final_img2)

def val_on_master_CNN_Head_ViT(model, data_loader_val, epoch, save_path, mode, patch_size):
    if is_main_process():
        val_model_CNN_Head_ViT(model, data_loader_val, epoch, save_path, mode, patch_size)

def val_model_CNN_Head_ViT(model, data_loader_val, epoch, save_path, mode, patch_size=16):
    idx = 0
    for batch, deg_type in data_loader_val:
        idx += 1
        #print('idx: ', idx, len(batch), deg_type)
        input_img1 = batch['input_query_img1'].cuda()
        
        target_img1 = batch['target_img1']
        if isinstance(target_img1, list):
            gt_flag = False
            target_img1 = input_img1
        else:
            gt_flag = True
            target_img1 = batch['target_img1'].cuda()
        
        input_sequence = [input_img1, target_img1]
        model.eval()
        loss, pred, pred_target_img1 = model(imgs=input_sequence, input_is_list=True)
        


        pred_target_img1 = torch.einsum('nchw->nhwc', pred_target_img1).detach().cpu()

        input_img1 = torch.einsum('nchw->nhwc', input_img1)
        target_img1 = torch.einsum('nchw->nhwc', target_img1)

        input_img1 = (torch.clip((input_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_img1 = (torch.clip((target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_img1 = (torch.clip((pred_target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        

        input_img1 = input_img1[0].numpy().astype(np.uint8)
        target_img1 = target_img1[0].numpy().astype(np.uint8)
        
        

        pred_target_img1 = pred_target_img1[0].numpy().astype(np.uint8)
        save_path_img = os.path.join(save_path, 'vis', mode, 'Epoch{}'.format(epoch))
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        
        
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img1_{}.png'.format(idx, deg_type[0])), input_img1)
        if gt_flag:
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img1.png'.format(idx)), target_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img1.png'.format(idx)), pred_target_img1)

        save_path_img = os.path.join(save_path, 'vis', mode, epoch)
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        input_query_img2_path = batch['input_query_img2_path'][0]
        img_name = input_query_img2_path.split('/')[-1]
        cv2.imwrite(os.path.join(save_path_img, '{}'.format(img_name)), pred_target_img1)
        
        
def fill_to_full_lyh(arr, total_patches):
    new_arr = copy.deepcopy(arr)
    if isinstance(new_arr, np.ndarray):
        new_arr = list(new_arr)
    for i in range(total_patches):
        if i not in new_arr:
            new_arr.append(i)
    return torch.tensor(new_arr)[np.newaxis, ]

def obtain_values_from_mask_lyh(mask):
    
    return list(mask.flatten().nonzero()[0])

# def generate_mask_for_evaluation_lyh(num_patches_row, total_patches):
#     mask = np.zeros((num_patches_row,num_patches_row))
#     mask[:num_patches_row//2] = 1
#     mask[:, :num_patches_row//2] = 1
#     mask = obtain_values_from_mask_lyh(mask)
#     len_keep = len(mask)
#     return fill_to_full_lyh(mask,total_patches=total_patches), len_keep

def generate_mask_for_evaluation_lyh(num_patches_row, total_patches):
    mask = np.zeros((num_patches_row,num_patches_row))
    mask[:num_patches_row//2] = 1
    mask[:, :num_patches_row//2] = 1
    mask = obtain_values_from_mask_lyh(mask)
    len_keep = len(mask)
    #return fill_to_full_lyh(mask,total_patches=total_patches), len_keep
    return mask, len_keep
        
def val_on_master_stitch(model, data_loader_val, epoch, save_path, mode, patch_size):
    if is_main_process():
        val_model_stitch(model, data_loader_val, epoch, save_path, mode, patch_size)



def val_model_stitch(model, data_loader_val, epoch, save_path, mode='val_pretrain', patch_size=16):
    if mode == 'val_pretrain':
        mask_ratio = 0.75
    elif mode == 'val_test':
        mask_ratio = 1
        
    idx = 0
    for batch, deg_type in data_loader_val:
        idx += 1
        #print('idx: ', idx, len(batch), deg_type)
        grid_stitch = batch['grid'].cuda()

        model.eval()
        if mode == 'val_pretrain':
            mask_ratio = 0.75
            loss, pred, mask = model(imgs=grid_stitch, mask_ratio=mask_ratio)
        elif mode == 'val_test':
            mask_ratio = 1
            img_size = grid_stitch.shape[2]
            ids_shuffle, len_keep = generate_mask_for_evaluation_lyh(num_patches_row=img_size//patch_size, total_patches=(img_size//patch_size)**2)
            preserve_list = ids_shuffle
            loss, pred, mask = model(imgs=grid_stitch, mask_ratio=mask_ratio, preserve_list=preserve_list)
        
        mask = mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        #print('mask: ', mask.shape)
        
        original_input = grid_stitch
        original_input = torch.einsum('nchw->nhwc', original_input).detach().cpu()
        original_input = (torch.clip((original_input[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        
        masked_input = original_input * (1-mask)
        
        pred_output = model.unpatchify(pred)
        pred_output = torch.einsum('nchw->nhwc', pred_output).detach().cpu()
        pred_output = (torch.clip((pred_output[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        
        
        pred_output_mask = pred_output * (mask)
        pred_output_fuse = masked_input * (1 - mask) + pred_output_mask * (mask)
        
        
        original_input = original_input[0].numpy().astype(np.uint8)
        masked_input  = masked_input[0].numpy().astype(np.uint8)
        pred_output_mask = pred_output_mask[0].numpy().astype(np.uint8)
        pred_output_fuse = pred_output_fuse[0].numpy().astype(np.uint8)
        
        

        
        
        save_path_img = os.path.join(save_path, 'vis', mode, 'Epoch{}'.format(epoch))
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        
        
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_ori_input_img_{}.png'.format(idx, deg_type[0])), original_input)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_masked_input_img.png'.format(idx)), masked_input)
        
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_img.png'.format(idx)), pred_output_mask)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_fuse_img.png'.format(idx)), pred_output_fuse)


def val_model_mismatched_prompt(model, data_loader_val, epoch, save_path, mode='val_pretrain', patch_size=16):
    if mode == 'val_pretrain':
        mask_ratio = 0.75
        train_mode = True
    elif mode == 'val_test':
        mask_ratio = 1
        train_mode = False
        
    idx = 0
    for batch, prompt_deg_type, test_deg_type in data_loader_val:
        idx += 1
        #print('idx: ', idx, len(batch), deg_type)
        input_img1 = batch['input_query_img1'].cuda()
        target_img1 = batch['target_img1'].cuda()
        input_img2 = batch['input_query_img2'].cuda()
        target_img2 = batch['target_img2'].cuda()
        input_sequence = [input_img1, target_img1, input_img2, target_img2]
        model.eval()
        loss, pred, mask = model(imgs=input_sequence, mask_ratio=mask_ratio, input_is_list=True, train_mode=train_mode)
        
        input_img1_patch_num = input_img1.shape[2]//patch_size*input_img1.shape[3]//patch_size
        target_img1_patch_num = target_img1.shape[2]//patch_size*target_img1.shape[3]//patch_size
        input_img2_patch_num = input_img2.shape[2]//patch_size*input_img2.shape[3]//patch_size
        target_img2_patch_num = target_img2.shape[2]//patch_size*target_img2.shape[3]//patch_size

        input_img1_mask = mask[:, 0:input_img1_patch_num]
        target_img1_mask = mask[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num]
        input_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num:input_img1_patch_num+target_img1_patch_num+input_img2_patch_num]
        target_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:]

        target_img1_mask = target_img1_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img1_mask = model.unpatchify(target_img1_mask)  # 1 is removing, 0 is keeping
        target_img1_mask = torch.einsum('nchw->nhwc', target_img1_mask).detach().cpu()
        target_img2_mask = target_img2_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img2_mask = model.unpatchify(target_img2_mask)  # 1 is removing, 0 is keeping
        target_img2_mask = torch.einsum('nchw->nhwc', target_img2_mask).detach().cpu()

        pred_target_img1 = pred[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num, :]
        pred_target_img1 = model.unpatchify(pred_target_img1)  # 1 is removing, 0 is keeping
        pred_target_img1 = torch.einsum('nchw->nhwc', pred_target_img1).detach().cpu()

        pred_target_img2 = pred[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:, :]
        pred_target_img2 = model.unpatchify(pred_target_img2)  # 1 is removing, 0 is keeping
        pred_target_img2 = torch.einsum('nchw->nhwc', pred_target_img2).detach().cpu()

        input_img1 = torch.einsum('nchw->nhwc', input_img1)
        target_img1 = torch.einsum('nchw->nhwc', target_img1)
        input_img2 = torch.einsum('nchw->nhwc', input_img2)
        target_img2 = torch.einsum('nchw->nhwc', target_img2)

        input_img1 = (torch.clip((input_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        input_img2 = (torch.clip((input_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)

        target_img1 = (torch.clip((target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img1 = target_img1 * (1 - target_img1_mask)
        target_img2 = (torch.clip((target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img2 = target_img2 * (1 - target_img2_mask)


        pred_target_img1 = (torch.clip((pred_target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img1 = pred_target_img1 * (target_img1_mask)
        pred_target_final_img1 = target_img1 * (1 - target_img1_mask) + pred_target_img1 * (target_img1_mask)
        pred_target_img2 = (torch.clip((pred_target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img2 = pred_target_img2 * (target_img2_mask)
        pred_target_final_img2 = target_img2 * (1 - target_img2_mask) + pred_target_img2 * (target_img2_mask)

        input_img1 = input_img1[0].numpy().astype(np.uint8)
        input_img2 = input_img2[0].numpy().astype(np.uint8)

        target_img1 = target_img1[0].numpy().astype(np.uint8)
        target_img2 = target_img2[0].numpy().astype(np.uint8)
        target_mask_img1 = target_mask_img1[0].numpy().astype(np.uint8)
        target_mask_img2 = target_mask_img2[0].numpy().astype(np.uint8)

        pred_target_img1 = pred_target_img1[0].numpy().astype(np.uint8)
        pred_target_mask_img1 = pred_target_mask_img1[0].numpy().astype(np.uint8)
        pred_target_final_img1 = pred_target_final_img1[0].numpy().astype(np.uint8)
        pred_target_img2 = pred_target_img2[0].numpy().astype(np.uint8)
        pred_target_mask_img2 = pred_target_mask_img2[0].numpy().astype(np.uint8)
        pred_target_final_img2 = pred_target_final_img2[0].numpy().astype(np.uint8)
        
        
        save_path_img = os.path.join(save_path, 'vis', mode, 'Epoch{}'.format(epoch))
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        
        
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img1_{}.png'.format(idx, prompt_deg_type[0])), input_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img1.png'.format(idx)), target_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img1.png'.format(idx)), target_mask_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img1.png'.format(idx)), pred_target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img1.png'.format(idx)), pred_target_mask_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img1.png'.format(idx)), pred_target_final_img1)

        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img2_{}.png'.format(idx, test_deg_type[0])), input_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img2.png'.format(idx)), target_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img2.png'.format(idx)), target_mask_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img2.png'.format(idx)), pred_target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img2.png'.format(idx)), pred_target_mask_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img2.png'.format(idx)), pred_target_final_img2)


### VQGAN mode ###
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def val_on_master_vqgan(model, data_loader_val, epoch, save_path, mode, patch_size):
    if is_main_process():
        val_model_vqgan(model, data_loader_val, epoch, save_path, mode, patch_size)

def val_model_vqgan(model, data_loader_val, epoch, save_path, mode='val_pretrain', patch_size=16):
    if mode == 'val_pretrain':
        mask_ratio = 0.75
        train_mode = True
    elif mode == 'val_test' or 'test' in mode:
        mask_ratio = 1
        train_mode = False
        
    idx = 0
    for batch, deg_type in data_loader_val:
        idx += 1
        #print('idx: ', idx, len(batch), deg_type)
        input_img1 = batch['input_query_img1'].cpu()
        target_img1 = batch['target_img1'].cpu()
        input_img2 = batch['input_query_img2'].cpu()
        
        # input_img1 = (input_img1 - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        # target_img1 = (target_img1 - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        # input_img2 = (input_img2 - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        
        input_img1 = input_img1.cuda().float()
        target_img1 = target_img1.cuda().float()
        input_img2 = input_img2.cuda().float()
        
        if batch['target_img2'][0] != 'None':
            target_img2 = batch['target_img2'].cuda()
            target_img2_flag = True
        else:
            target_img2 = input_img2
            target_img2_flag = False
        num_patches = input_img1.shape[2]//patch_size
        input_sequence = [input_img1, target_img1, input_img2, target_img2]
        model.eval()
        loss, pred, mask = model(imgs=input_sequence, mask_ratio=mask_ratio, input_is_list=True, train_mode=train_mode)
        
        input_img1_patch_num = input_img1.shape[2]//patch_size*input_img1.shape[3]//patch_size
        target_img1_patch_num = target_img1.shape[2]//patch_size*target_img1.shape[3]//patch_size
        input_img2_patch_num = input_img2.shape[2]//patch_size*input_img2.shape[3]//patch_size
        target_img2_patch_num = target_img2.shape[2]//patch_size*target_img2.shape[3]//patch_size

        input_img1_mask = mask[:, 0:input_img1_patch_num]
        target_img1_mask = mask[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num]
        input_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num:input_img1_patch_num+target_img1_patch_num+input_img2_patch_num]
        target_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:]

        target_img1_mask = target_img1_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img1_mask = model.unpatchify(target_img1_mask)  # 1 is removing, 0 is keeping
        target_img1_mask = torch.einsum('nchw->nhwc', target_img1_mask).detach().cpu()
        target_img2_mask = target_img2_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img2_mask = model.unpatchify(target_img2_mask)  # 1 is removing, 0 is keeping
        target_img2_mask = torch.einsum('nchw->nhwc', target_img2_mask).detach().cpu()

        # predict target img1
        pred_target_img1_latent = pred[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num, :]
        y = pred_target_img1_latent.argmax(dim=-1)
        y = model.vae.quantize.get_codebook_entry(y.reshape(-1),
                                              [y.shape[0], y.shape[-1] // num_patches, y.shape[-1] // num_patches, -1])
        y = model.vae.decode(y)
        y = F.interpolate(y, size=(input_img1.shape[2], input_img1.shape[3]), mode='bilinear').permute(0, 2, 3, 1)
        pred_target_img1 = y
        
        # predict target img2 
        pred_target_img2_latent = pred[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:, :]
        y = pred_target_img2_latent.argmax(dim=-1)
        y = model.vae.quantize.get_codebook_entry(y.reshape(-1),
                                              [y.shape[0], y.shape[-1] // num_patches, y.shape[-1] // num_patches, -1])
        y = model.vae.decode(y)
        y = F.interpolate(y, size=(input_img1.shape[2], input_img1.shape[3]), mode='bilinear').permute(0, 2, 3, 1)
        pred_target_img2 = y
        

        input_img1 = torch.einsum('nchw->nhwc', input_img1)
        target_img1 = torch.einsum('nchw->nhwc', target_img1)
        input_img2 = torch.einsum('nchw->nhwc', input_img2)
        target_img2 = torch.einsum('nchw->nhwc', target_img2)
        


        input_img1 = (torch.clip((input_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        input_img2 = (torch.clip((input_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)

        target_img1 = (torch.clip((target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img1 = target_img1 * (1 - target_img1_mask)
        target_img2 = (torch.clip((target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img2 = target_img2 * (1 - target_img2_mask)


        pred_target_img1 = (torch.clip((pred_target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img1 = pred_target_img1 * (target_img1_mask)
        pred_target_final_img1 = target_img1 * (1 - target_img1_mask) + pred_target_img1 * (target_img1_mask)
        #pred_target_img2 = (torch.clip((pred_target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_img2 = (torch.clip(((pred_target_img2[0].cpu().detach()-imagenet_mean)/imagenet_std) * 255, 0, 255).int()).unsqueeze(0)
        
        #pred_target_img2 = torch.clip((pred_target_img2[0].cpu().detach() * imagenet_std + imagenet_mean) * 255, 0, 255).int().numpy()
        
        pred_target_mask_img2 = pred_target_img2 * (target_img2_mask)
        pred_target_final_img2 = target_img2 * (1 - target_img2_mask) + pred_target_img2 * (target_img2_mask)

        input_img1 = input_img1[0].numpy().astype(np.uint8)
        input_img2 = input_img2[0].numpy().astype(np.uint8)

        target_img1 = target_img1[0].numpy().astype(np.uint8)
        target_img2 = target_img2[0].numpy().astype(np.uint8)
        target_mask_img1 = target_mask_img1[0].numpy().astype(np.uint8)
        target_mask_img2 = target_mask_img2[0].numpy().astype(np.uint8)

        pred_target_img1 = pred_target_img1[0].numpy().astype(np.uint8)
        pred_target_mask_img1 = pred_target_mask_img1[0].numpy().astype(np.uint8)
        pred_target_final_img1 = pred_target_final_img1[0].numpy().astype(np.uint8)
        pred_target_img2 = pred_target_img2[0].numpy().astype(np.uint8)
        pred_target_mask_img2 = pred_target_mask_img2[0].numpy().astype(np.uint8)
        pred_target_final_img2 = pred_target_final_img2[0].numpy().astype(np.uint8)
        
        
        save_path_img = os.path.join(save_path, 'vis', mode, 'Epoch{}'.format(epoch))
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        
        
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img1_{}.png'.format(idx, deg_type[0])), input_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img1.png'.format(idx)), target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img1.png'.format(idx)), target_mask_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img1.png'.format(idx)), pred_target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img1.png'.format(idx)), pred_target_mask_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img1.png'.format(idx)), pred_target_final_img1)

        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img2_{}.png'.format(idx, deg_type[0])), input_img2)
        if target_img2_flag:
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img2.png'.format(idx)), target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img2.png'.format(idx)), target_mask_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img2.png'.format(idx)), pred_target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img2.png'.format(idx)), pred_target_mask_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img2.png'.format(idx)), pred_target_final_img2)



def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def val_model_PSNR(model, data_loader_val, epoch, save_path, mode='val_pretrain', patch_size=16):
    if mode == 'val_pretrain':
        mask_ratio = 0.75
        train_mode = True
    elif mode == 'val_test' or 'test' in mode:
        mask_ratio = 1
        train_mode = False
    
    PSNR_list = []    
    idx = 0
    for batch, deg_type in data_loader_val:
        idx += 1
        #print('idx: ', idx, len(batch), deg_type)
        input_img1 = batch['input_query_img1'].cuda()
        target_img1 = batch['target_img1'].cuda()
        input_img2 = batch['input_query_img2'].cuda()
        if batch['target_img2'][0] != 'None':
            target_img2 = batch['target_img2'].cuda()
            target_img2_flag = True
        else:
            target_img2 = input_img2
            target_img2_flag = False
        input_sequence = [input_img1, target_img1, input_img2, target_img2]
        model.eval()
        loss, pred, mask = model(imgs=input_sequence, mask_ratio=mask_ratio, input_is_list=True, train_mode=train_mode)
        
        input_img1_patch_num = input_img1.shape[2]//patch_size*input_img1.shape[3]//patch_size
        target_img1_patch_num = target_img1.shape[2]//patch_size*target_img1.shape[3]//patch_size
        input_img2_patch_num = input_img2.shape[2]//patch_size*input_img2.shape[3]//patch_size
        target_img2_patch_num = target_img2.shape[2]//patch_size*target_img2.shape[3]//patch_size

        input_img1_mask = mask[:, 0:input_img1_patch_num]
        target_img1_mask = mask[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num]
        input_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num:input_img1_patch_num+target_img1_patch_num+input_img2_patch_num]
        target_img2_mask = mask[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:]

        target_img1_mask = target_img1_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img1_mask = model.unpatchify(target_img1_mask)  # 1 is removing, 0 is keeping
        target_img1_mask = torch.einsum('nchw->nhwc', target_img1_mask).detach().cpu()
        target_img2_mask = target_img2_mask.unsqueeze(-1).repeat(1, 1, patch_size ** 2 * 3)
        target_img2_mask = model.unpatchify(target_img2_mask)  # 1 is removing, 0 is keeping
        target_img2_mask = torch.einsum('nchw->nhwc', target_img2_mask).detach().cpu()

        pred_target_img1 = pred[:, input_img1_patch_num:input_img1_patch_num+target_img1_patch_num, :]
        pred_target_img1 = model.unpatchify(pred_target_img1)  # 1 is removing, 0 is keeping
        pred_target_img1 = torch.einsum('nchw->nhwc', pred_target_img1).detach().cpu()

        pred_target_img2 = pred[:, input_img1_patch_num+target_img1_patch_num+input_img2_patch_num:, :]
        pred_target_img2 = model.unpatchify(pred_target_img2)  # 1 is removing, 0 is keeping
        pred_target_img2 = torch.einsum('nchw->nhwc', pred_target_img2).detach().cpu()

        input_img1 = torch.einsum('nchw->nhwc', input_img1)
        target_img1 = torch.einsum('nchw->nhwc', target_img1)
        input_img2 = torch.einsum('nchw->nhwc', input_img2)
        target_img2 = torch.einsum('nchw->nhwc', target_img2)

        input_img1 = (torch.clip((input_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        input_img2 = (torch.clip((input_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)

        target_img1 = (torch.clip((target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img1 = target_img1 * (1 - target_img1_mask)
        target_img2 = (torch.clip((target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        target_mask_img2 = target_img2 * (1 - target_img2_mask)


        pred_target_img1 = (torch.clip((pred_target_img1[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img1 = pred_target_img1 * (target_img1_mask)
        pred_target_final_img1 = target_img1 * (1 - target_img1_mask) + pred_target_img1 * (target_img1_mask)
        pred_target_img2 = (torch.clip((pred_target_img2[0].cpu().detach()) * 255, 0, 255).int()).unsqueeze(0)
        pred_target_mask_img2 = pred_target_img2 * (target_img2_mask)
        pred_target_final_img2 = target_img2 * (1 - target_img2_mask) + pred_target_img2 * (target_img2_mask)

        input_img1 = input_img1[0].numpy().astype(np.uint8)
        input_img2 = input_img2[0].numpy().astype(np.uint8)

        target_img1 = target_img1[0].numpy().astype(np.uint8)
        target_img2 = target_img2[0].numpy().astype(np.uint8)
        target_mask_img1 = target_mask_img1[0].numpy().astype(np.uint8)
        target_mask_img2 = target_mask_img2[0].numpy().astype(np.uint8)

        pred_target_img1 = pred_target_img1[0].numpy().astype(np.uint8)
        pred_target_mask_img1 = pred_target_mask_img1[0].numpy().astype(np.uint8)
        pred_target_final_img1 = pred_target_final_img1[0].numpy().astype(np.uint8)
        pred_target_img2 = pred_target_img2[0].numpy().astype(np.uint8)
        pred_target_mask_img2 = pred_target_mask_img2[0].numpy().astype(np.uint8)
        pred_target_final_img2 = pred_target_final_img2[0].numpy().astype(np.uint8)
        
        
        save_path_img = os.path.join(save_path, mode)
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)
        
        
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img1_{}.png'.format(idx, deg_type[0])), input_img1)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img1.png'.format(idx)), target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img1.png'.format(idx)), target_mask_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img1.png'.format(idx)), pred_target_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img1.png'.format(idx)), pred_target_mask_img1)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img1.png'.format(idx)), pred_target_final_img1)

        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_input_img2_{}.png'.format(idx, deg_type[0])), input_img2)
        if target_img2_flag:
            cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_img2.png'.format(idx)), target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_target_mask_img2.png'.format(idx)), target_mask_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_img2.png'.format(idx)), pred_target_img2)
        #cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_mask_img2.png'.format(idx)), pred_target_mask_img2)
        cv2.imwrite(os.path.join(save_path_img, 'TestID{:04d}_pred_target_fuse_img2.png'.format(idx)), pred_target_final_img2)


        # calculate PSNR
        psnr = calculate_psnr(pred_target_final_img2, target_img2)
        PSNR_list.append(psnr)
        
        
    print('AVG PSNR = {:.4f}    Total image: {}'.format(np.mean(PSNR_list), len(PSNR_list)))