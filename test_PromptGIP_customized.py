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
import dataset.lowlevel_PromtGIP_dataloader as lowlevel_prompt_dataloader
from evaluation_prompt import *
from sklearn.metrics import mean_squared_error

def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output_dir/')
    parser.add_argument('--data_path_HQ')
    parser.add_argument('--data_path_LQ')
    parser.add_argument('--dataset_type')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tta_option', default=0, type=int)
    parser.add_argument('--ckpt', help='resume from checkpoint')
    
    parser.add_argument('--prompt_id', default=-1, type=int)
    parser.add_argument('--degradation_type', default='GaussianNoise', type=str)

    parser.set_defaults(autoregressive=False)
    return parser


def main(args):
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    
    
    
    degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'Resize', 
                                 'Rain', 'SPNoise', 'LowLight', 'PoissonNoise', 'Ringing',
                                 'r_l', 'Inpainting']
    
    degradation_type = args.degradation_type
    if degradation_type in degradation_type_list:
        dataset_path_HQ = '/nvme/liuyihao/DATA/Common528/Image_clean_256x256'
        dataset_path_LQ = '/nvme/liuyihao/DATA/Common528/{}'.format(degradation_type)
    elif degradation_type == 'Test100':
        dataset_path_HQ = '/nvme/liuyihao/DATA/Test100_256/target'
        dataset_path_LQ = '/nvme/liuyihao/DATA/Test100_256/input'
    elif degradation_type == 'SOTS':
        dataset_path_HQ = '/nvme/liuyihao/DATA/SOTS_256/gt'
        dataset_path_LQ = '/nvme/liuyihao/DATA/SOTS_256/hazy'
    elif degradation_type == 'LOL':
        dataset_path_HQ = '/nvme/liuyihao/DATA/LOL_256/eval15/high'
        dataset_path_LQ = '/nvme/liuyihao/DATA/LOL_256/eval15/low'
    elif degradation_type == 'Canny':
        dataset_path_HQ = '/nvme/liuyihao/DATA/Common528/Canny'
        dataset_path_LQ = '/nvme/liuyihao/DATA/Common528/Image_clean_256x256'
    elif degradation_type == 'LLF':
        dataset_path_HQ = '/nvme/liuyihao/DATA/LLF_256/gt'
        dataset_path_LQ = '/nvme/liuyihao/DATA/LLF_256/input'
    elif degradation_type == 'Laplacian':
        dataset_path_HQ = '/nvme/liuyihao/DATA/Common528/Laplacian'
        dataset_path_LQ = '/nvme/liuyihao/DATA/Common528/Image_clean_256x256'
    elif degradation_type == 'L0_smooth':
        dataset_path_HQ = '/nvme/liuyihao/DATA/Common528/L0_smooth'
        dataset_path_LQ = '/nvme/liuyihao/DATA/Common528/Image_clean_256x256'
    
    dataset_val = lowlevel_prompt_dataloader.DatasetLowlevel_Customized_Val_original_size(dataset_path_HQ=dataset_path_HQ, 
                                                                                 dataset_path_LQ=dataset_path_LQ, 
                                                                                 dataset_type=degradation_type,
                                                                                 data_len=None,
                                                                                 prompt_id = args.prompt_id)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # build model
    print('Model: ', models_mae_PromptGIP_CNN_Head)
    model = getattr(models_mae_PromptGIP_CNN_Head, args.model)()
    # load model
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=True)
    print(msg)
    model.to(device)



    print('degradation_type: ', degradation_type)    
    val_model_CNN_Head_ExtraSaveOutput(model.cuda(), data_loader_val, degradation_type, args.output_dir, mode='val_test', patch_size=16)
    metric_msg = compute_metrics(os.path.join(args.output_dir, 'vis', 'val_test', degradation_type), dataset_path_HQ)
    with open(os.path.join(args.output_dir, degradation_type+'_result.txt'), 'a') as f:
        f.write('prompt id: {}'.format(args.prompt_id)+'  '+metric_msg+'\n')

      
def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border=0,
                   input_order='HWC',
                   test_y_channel=False):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]


    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()

def compute_metrics(input_path, GT_path):
    input_fname_list = os.listdir(input_path)
    input_fname_list.sort()
    input_path_list = [os.path.join(input_path, fname) for fname in input_fname_list]

    GT_fname_list = os.listdir(GT_path)
    GT_fname_list.sort()
    GT_path_list = [os.path.join(GT_path, fname) for fname in GT_fname_list]

    #assert len(input_path_list) == len(GT_path_list)
    print(len(input_path_list))

    crop_border = 0

    psnr_list = []
    ssim_list = []
    mae_list = []
    mse_list = []
    for i in range(len(input_path_list)):
        #assert input_fname_list[i].split('.')[0] == GT_fname_list[i].split('.')[0]
        img1 = cv2.imread(input_path_list[i], cv2.IMREAD_COLOR)
        
        if 'SOTS' in GT_path:
            HQ_name = input_path_list[i].split('/')[-1].split('_')[0]
            HQ_path = os.path.join(GT_path, '{}.png'.format(HQ_name))
            img2 = cv2.imread(HQ_path, cv2.IMREAD_COLOR)
        else:
            img2 = cv2.imread(GT_path_list[i], cv2.IMREAD_COLOR)
        
            
        if crop_border == 0:
            img1 = img1
            img2 = img2
        else:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border, :]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]
        
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        psnr = calculate_psnr(img1_rgb, img2_rgb)
        ssim = calculate_ssim(img1_rgb, img2_rgb)
        
        mae = np.mean(np.abs(img1_rgb.astype(np.float64) - img2_rgb.astype(np.float64)))
        mse = np.mean((img1_rgb.astype(np.float64) - img2_rgb.astype(np.float64))**2)
        
        #print('img: {}  PSNR: {:.4f}  SSIM: {:.4f}'.format(input_fname_list[i].split('.')[0], psnr, ssim))
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        mae_list.append(mae)
        mse_list.append(mse)

        
    print('Prompt ID: ', args.prompt_id)
    msg = 'Average PSNR: {:.4f}  SSIM: {:.4f}  MAE: {:.4f}  MSE: {:.4f}  Total image: {}'.format(np.mean(psnr_list), np.mean(ssim_list), np.mean(mae_list), np.mean(mse_list), len(psnr_list))
    print(msg)
    
    
    return msg

if __name__ == '__main__':
    args = get_args()
    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
