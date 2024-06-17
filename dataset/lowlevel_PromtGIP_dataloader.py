import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from evaluate.add_degradation_various import *
from dataset.image_operators import *

import dataset.util as util


def add_degradation_two_images(img_HQ1, img_HQ2, deg_type):
    if deg_type == 'GaussianNoise':
        level = random.uniform(10, 50)
        img_LQ1 = add_Gaussian_noise(img_HQ1.copy(), level=level)
        level = random.uniform(10, 50)
        img_LQ2 = add_Gaussian_noise(img_HQ2.copy(), level=level)
    elif deg_type == 'GaussianBlur':
        sigma = random.uniform(2, 4)
        img_LQ1 = iso_GaussianBlur(img_HQ1.copy(), window=15, sigma=sigma)
        sigma = random.uniform(2, 4)
        img_LQ2 = iso_GaussianBlur(img_HQ2.copy(), window=15, sigma=sigma)
    elif deg_type == 'JPEG':
        level = random.randint(10, 40)
        img_LQ1 = add_JPEG_noise(img_HQ1.copy(), level=level)
        level = random.randint(10, 40)
        img_LQ2 = add_JPEG_noise(img_HQ2.copy(), level=level)
    elif deg_type == 'Resize':
        img_LQ1 = add_resize(img_HQ1.copy())
        img_LQ2 = add_resize(img_HQ2.copy())
    elif deg_type == 'Rain':
        value = random.uniform(40, 200)
        img_LQ1 = add_rain(img_HQ1.copy(), value=value)
        value = random.uniform(40, 200)
        img_LQ2 = add_rain(img_HQ2.copy(), value=value)
    elif deg_type == 'SPNoise':
        img_LQ1 = add_sp_noise(img_HQ1.copy())
        img_LQ2 = add_sp_noise(img_HQ2.copy())
    elif deg_type == 'LowLight':
        lum_scale = random.uniform(0.3, 0.4)
        img_LQ1 = low_light(img_HQ1.copy(), lum_scale=lum_scale)
        img_LQ2 = low_light(img_HQ2.copy(), lum_scale=lum_scale)
    elif deg_type == 'PoissonNoise':
        img_LQ1 = add_Poisson_noise(img_HQ1.copy(), level=2)
        img_LQ2 = add_Poisson_noise(img_HQ2.copy(), level=2)
    elif deg_type == 'Ringing':
        img_LQ1 = add_ringing(img_HQ1.copy())
        img_LQ2 = add_ringing(img_HQ2.copy())
    elif deg_type == 'r_l':
        img_LQ1 = r_l(img_HQ1.copy())
        img_LQ2 = r_l(img_HQ2.copy())
    elif deg_type == 'Inpainting':
        l_num = random.randint(5, 10)
        l_thick = random.randint(5, 10)
        img_LQ1 = inpainting(img_HQ1.copy(), l_num=l_num, l_thick=l_thick)
        img_LQ2 = inpainting(img_HQ2.copy(), l_num=l_num, l_thick=l_thick)
    elif deg_type == 'gray':
        img_LQ1 = cv2.cvtColor(img_HQ1.copy(), cv2.COLOR_BGR2GRAY)
        img_LQ1 = np.expand_dims(img_LQ1, axis=2)
        img_LQ1 = np.concatenate((img_LQ1, img_LQ1, img_LQ1), axis=2)
        img_LQ2 = cv2.cvtColor(img_HQ2.copy(), cv2.COLOR_BGR2GRAY)
        img_LQ2 = np.expand_dims(img_LQ2, axis=2)
        img_LQ2 = np.concatenate((img_LQ2, img_LQ2, img_LQ2), axis=2)
    elif deg_type == '':
        img_LQ1 = img_HQ1
        img_LQ2 = img_HQ2
    elif deg_type == 'Laplacian':
        img_LQ1 = img_HQ1.copy()
        img_HQ1 = Laplacian_edge_detector(img_HQ1.copy())
        img_LQ2 = img_HQ2.copy()
        img_HQ2 = Laplacian_edge_detector(img_HQ2.copy())
    elif deg_type == 'Canny':
        img_LQ1 = img_HQ1.copy()
        img_HQ1 = Canny_edge_detector(img_HQ1.copy())
        img_LQ2 = img_HQ2.copy()
        img_HQ2 = Canny_edge_detector(img_HQ2.copy())
    elif deg_type == 'L0_smooth':
        img_LQ1 = img_HQ1.copy()
        img_HQ1 = L0_smooth(img_HQ1.copy())
        img_LQ2 = img_HQ2.copy()
        img_HQ2 = L0_smooth(img_HQ2.copy())
    else:
        print('Error!')
        exit()

    img_LQ1 = np.clip(img_LQ1*255, 0, 255).round().astype(np.uint8)
    img_LQ1 = img_LQ1.astype(np.float32)/255.0
    
    img_LQ2 = np.clip(img_LQ2*255, 0, 255).round().astype(np.uint8)
    img_LQ2 = img_LQ2.astype(np.float32)/255.0
    
    img_HQ1 = np.clip(img_HQ1*255, 0, 255).round().astype(np.uint8)
    img_HQ1 = img_HQ1.astype(np.float32)/255.0
    
    img_HQ2 = np.clip(img_HQ2*255, 0, 255).round().astype(np.uint8)
    img_HQ2 = img_HQ2.astype(np.float32)/255.0

    return img_LQ1, img_LQ2, img_HQ1, img_HQ2

class DatasetLowlevel_Train(Dataset):
    def __init__(self, dataset_path, input_size, phase, ITS_path=None, LOL_path=None, Rain13K_path=None,
                 GoPro_path=None, FiveK_path=None, LLF_path=None, RealOld_path=None):
        
        np.random.seed(5)
        self.HQ_size = input_size
        self.phase = phase
        
        # base dataset
        self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path)
        self.paths_base_len = len(self.paths_HQ)
        
        # dehaze dataset (ITS)
        if ITS_path is not None:
            self.dataset_path_HQ_ITS = os.path.join(ITS_path, 'clear')
            self.dataset_path_LQ_ITS = os.path.join(ITS_path, 'hazy')
            self.paths_HQ_ITS, self.sizes_HQ_ITS = util.get_image_paths('img', self.dataset_path_HQ_ITS)
            self.paths_LQ_ITS, self.sizes_LQ_ITS = util.get_image_paths('img', self.dataset_path_LQ_ITS)
            self.paths_ITS_len = len(self.paths_LQ_ITS)
        
        # low-light enhancement dataset (LOL)
        if LOL_path is not None:
            self.dataset_path_HQ_LOL = os.path.join(LOL_path, 'high')
            self.dataset_path_LQ_LOL = os.path.join(LOL_path, 'low')
            self.paths_HQ_LOL, self.sizes_HQ_LOL = util.get_image_paths('img', self.dataset_path_HQ_LOL)
            self.paths_LQ_LOL, self.sizes_LQ_LOL = util.get_image_paths('img', self.dataset_path_LQ_LOL)
            self.paths_LOL_len = len(self.paths_LQ_LOL)
            sorted(self.paths_HQ_LOL)
            sorted(self.paths_LQ_LOL)
            
        # Derain dataset (Rain13K)
        if Rain13K_path is not None:
            self.dataset_path_HQ_Rain13K = os.path.join(Rain13K_path, 'target')
            self.dataset_path_LQ_Rain13K = os.path.join(Rain13K_path, 'input')
            self.paths_HQ_Rain13K, self.sizes_HQ_Rain13K = util.get_image_paths('img', self.dataset_path_HQ_Rain13K)
            self.paths_LQ_Rain13K, self.sizes_LQ_Rain13K = util.get_image_paths('img', self.dataset_path_LQ_Rain13K)
            self.paths_Rain13K_len = len(self.paths_LQ_Rain13K)
            sorted(self.paths_HQ_Rain13K)
            sorted(self.paths_LQ_Rain13K)
            
        # Motion Deblur dataset (GoPro)
        if GoPro_path is not None:
            self.dataset_path_HQ_GoPro = os.path.join(GoPro_path, 'groundtruth')
            self.dataset_path_LQ_GoPro = os.path.join(GoPro_path, 'input')
            self.paths_HQ_GoPro, self.sizes_HQ_GoPro = util.get_image_paths('img', self.dataset_path_HQ_GoPro)
            self.paths_LQ_GoPro, self.sizes_LQ_GoPro = util.get_image_paths('img', self.dataset_path_LQ_GoPro)
            self.paths_GoPro_len = len(self.paths_LQ_GoPro)
            sorted(self.paths_HQ_GoPro)
            sorted(self.paths_LQ_GoPro)
            
        # Image Retouching dataset (MIT-Adobe FiveK)
        if FiveK_path is not None:
            self.dataset_path_HQ_FiveK = os.path.join(FiveK_path, 'expert_C_train')
            self.dataset_path_LQ_FiveK = os.path.join(FiveK_path, 'raw_input_train_png')
            self.paths_HQ_FiveK, self.sizes_HQ_FiveK = util.get_image_paths('img', self.dataset_path_HQ_FiveK)
            self.paths_LQ_FiveK, self.sizes_LQ_FiveK = util.get_image_paths('img', self.dataset_path_LQ_FiveK)
            self.paths_FiveK_len = len(self.paths_LQ_FiveK)
            sorted(self.paths_HQ_FiveK)
            sorted(self.paths_LQ_FiveK)
            
        
        # Local Laplacian Filter dataset (MIT-Adobe FiveK - LLF)
        if LLF_path is not None:
            self.dataset_path_HQ_LLF = os.path.join(LLF_path, 'expert_C_LLF_GT_train')
            self.dataset_path_LQ_LLF = os.path.join(LLF_path, 'expert_C_train')
            self.paths_HQ_LLF, self.sizes_HQ_LLF = util.get_image_paths('img', self.dataset_path_HQ_LLF)
            self.paths_LQ_LLF, self.sizes_LQ_LLF = util.get_image_paths('img', self.dataset_path_LQ_LLF)
            self.paths_LLF_len = len(self.paths_LQ_LLF)
            sorted(self.paths_HQ_LLF)
            sorted(self.paths_LQ_LLF)
            
        # RealOld dataset
        if RealOld_path is not None:
            self.dataset_path_HQ_RealOld = os.path.join(RealOld_path, 'HQ')
            self.dataset_path_LQ_RealOld = os.path.join(RealOld_path, 'LQ')
            self.paths_HQ_RealOld, self.sizes_HQ_RealOld = util.get_image_paths('img', self.dataset_path_HQ_RealOld)
            self.paths_LQ_RealOld, self.sizes_LQ_RealOld = util.get_image_paths('img', self.dataset_path_LQ_RealOld)
            self.paths_RealOld_len = len(self.paths_LQ_RealOld)
            sorted(self.paths_HQ_RealOld)
            sorted(self.paths_LQ_RealOld)
        
        #self.dataset_list = ['Base', 'ITS', 'LOL', 'Rain13K', 'RealOld', 'FiveK', 'LLF']
        self.dataset_list = ['Base', 'ITS', 'LOL', 'Rain13K', 'LLF']
        
        #self.dataset_list = ['Base', 'ITS']
        print('dataset list: ', self.dataset_list)
        
        self.degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'LowLight',
                                    'Rain', 'SPNoise', 'PoissonNoise', 'Ringing',
                                    'r_l', 'Inpainting', 'Laplacian', 'Canny']
        
        print('degradation_type list: ', self.degradation_type_list)
        
    def __len__(self):
        return len(self.paths_HQ) #+ len(self.paths_LQ_ITS) + len(self.paths_HQ_LOL) + len(self.paths_HQ_Rain13K)


    def __getitem__(self, idx):
        #dataset_choice = np.random.choice(self.dataset_list, p=[14/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20])
        #dataset_choice = np.random.choice(self.dataset_list, p=[1/2, 1/2])
        dataset_choice = np.random.choice(self.dataset_list, p=[12/20, 2/20, 2/20, 2/20, 2/20])
        
        if dataset_choice == 'Base':
            random_index1 = random.randint(0, self.paths_base_len-1)
            HQ1_path = self.paths_HQ[random_index1]
            random_index2 = random.randint(0, self.paths_base_len-1)
            HQ2_path = self.paths_HQ[random_index2]
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_HQ2 = util.read_img(None, HQ2_path, None)
            
            if self.phase == 'train':
                # if the image size is too small
                #print(HQ_size)
                H, W, _ = img_HQ1.shape
                #print(H, W)
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ1 = cv2.resize(np.copy(img_HQ1), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_HQ2.shape
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ2 = cv2.resize(np.copy(img_HQ2), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                if img_HQ1.ndim == 2:
                    img_HQ1 = np.expand_dims(img_HQ1, axis=2)
                    img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
                if img_HQ2.ndim == 2:
                    img_HQ2 = np.expand_dims(img_HQ2, axis=2)
                    img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
                    
                if img_HQ1.shape[2] !=3:
                    img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
                if img_HQ2.shape[2] !=3:
                    img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
            
            
            # add degradation to HQ
            
            
            degradation_type_1 = ['LowLight', '']
            degradation_type_2 = ['GaussianBlur', 'Ringing', 'r_l', '']
            degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', '']
            degradation_type_4 = ['JPEG', '']
            degradation_type_5 = ['Inpainting', 'Rain', '']
            
            round_select = np.random.choice(['1', 'Single'], p=[4/5, 1/5])
            #round_select = np.random.choice(['1', 'Single'], p=[0, 1])
            
            if round_select == '1':
                # 1 round
                deg_type1 = random.choice(degradation_type_1)
                deg_type2 = random.choice(degradation_type_2)
                deg_type3 = random.choice(degradation_type_3)
                deg_type4 = random.choice(degradation_type_4)
                deg_type5 = random.choice(degradation_type_5)
                img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_HQ1), np.copy(img_HQ2), deg_type1)
                img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type2)
                img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type3)
                img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type4)
                img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type5)
                deg_type = 'R1_'+deg_type1+'_'+deg_type2+'_'+deg_type3+'_'+deg_type4+'_'+deg_type5
                # print(deg_type)
                # print(img_HQ1.shape)
                # print(img_LQ1.shape)
                # print(img_HQ2.shape)
                # print(img_LQ2.shape)
            elif round_select == 'Single':
                deg_type1 = random.choice(self.degradation_type_list)
                img_LQ1, img_LQ2, img_HQ1, img_HQ2 = add_degradation_two_images(img_HQ1, img_HQ2, deg_type1)
                deg_type = deg_type1
                
            
   
            
        elif dataset_choice == 'ITS':
            random_index1 = random.randint(0, self.paths_ITS_len-1)
            LQ1_path = self.paths_LQ_ITS[random_index1]
            HQ1_name = LQ1_path.split('/')[-1].split('_')[0]
            HQ1_path = os.path.join(self.dataset_path_HQ_ITS, '{}.png'.format(HQ1_name))
            
            random_index2 = random.randint(0, self.paths_ITS_len-1)
            LQ2_path = self.paths_LQ_ITS[random_index2]
            HQ2_name = LQ2_path.split('/')[-1].split('_')[0]
            HQ2_path = os.path.join(self.dataset_path_HQ_ITS, '{}.png'.format(HQ2_name))
            deg_type = 'ITS'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
            # H_GT, W_GT, _ = img_HQ1.shape
            # H_LQ, W_LQ, _ = img_LQ1.shape
            
            # crop_size_H = np.abs(H_LQ-H_GT)//2
            # crop_size_W = np.abs(W_LQ-W_GT)//2
            # img_HQ1 = img_HQ1[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
            # img_HQ2 = img_HQ2[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
        
        elif dataset_choice == 'LOL':
            random_index1 = random.randint(0, self.paths_LOL_len-1)
            LQ1_path = self.paths_LQ_LOL[random_index1]
            HQ1_path = self.paths_HQ_LOL[random_index1]
            
            random_index2 = random.randint(0, self.paths_LOL_len-1)
            LQ2_path = self.paths_LQ_LOL[random_index2]
            HQ2_path = self.paths_HQ_LOL[random_index2]
            
            deg_type = 'LOL'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
        
        elif dataset_choice == 'GoPro':
            random_index1 = random.randint(0, self.paths_GoPro_len-1)
            LQ1_path = self.paths_LQ_GoPro[random_index1]
            HQ1_path = self.paths_HQ_GoPro[random_index1]
            
            random_index2 = random.randint(0, self.paths_GoPro_len-1)
            LQ2_path = self.paths_LQ_GoPro[random_index2]
            HQ2_path = self.paths_HQ_GoPro[random_index2]
            
            deg_type = 'GoPro'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
        
        elif dataset_choice == 'FiveK':
            random_index1 = random.randint(0, self.paths_FiveK_len-1)
            LQ1_path = self.paths_LQ_FiveK[random_index1]
            HQ1_path = self.paths_HQ_FiveK[random_index1]
            
            random_index2 = random.randint(0, self.paths_FiveK_len-1)
            LQ2_path = self.paths_LQ_FiveK[random_index2]
            HQ2_path = self.paths_HQ_FiveK[random_index2]
            
            deg_type = 'FiveK'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
        
        elif dataset_choice == 'LLF':
            random_index1 = random.randint(0, self.paths_LLF_len-1)
            LQ1_path = self.paths_LQ_LLF[random_index1]
            HQ1_path = self.paths_HQ_LLF[random_index1]
            
            random_index2 = random.randint(0, self.paths_LLF_len-1)
            LQ2_path = self.paths_LQ_LLF[random_index2]
            HQ2_path = self.paths_HQ_LLF[random_index2]
            
            deg_type = 'LLF'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
        elif dataset_choice == 'Rain13K':
            random_index1 = random.randint(0, self.paths_Rain13K_len-1)
            LQ1_path = self.paths_LQ_Rain13K[random_index1]
            HQ1_path = self.paths_HQ_Rain13K[random_index1]
            
            random_index2 = random.randint(0, self.paths_Rain13K_len-1)
            LQ2_path = self.paths_LQ_Rain13K[random_index2]
            HQ2_path = self.paths_HQ_Rain13K[random_index2]
            
            deg_type = 'Rain13K'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
            if self.phase == 'train':
                # if the image size is too small
                #print(HQ_size)
                H, W, _ = img_HQ1.shape
                #print(H, W)
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ1 = cv2.resize(np.copy(img_HQ1), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_HQ2.shape
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ2 = cv2.resize(np.copy(img_HQ2), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_LQ1.shape
                #print(H, W)
                if H < self.HQ_size or W < self.HQ_size:
                    img_LQ1 = cv2.resize(np.copy(img_LQ1), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_LQ2.shape
                if H < self.HQ_size or W < self.HQ_size:
                    img_LQ2 = cv2.resize(np.copy(img_LQ2), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
        
        elif dataset_choice == 'RealOld':
            random_index1 = random.randint(0, self.paths_RealOld_len-1)
            LQ1_path = self.paths_LQ_RealOld[random_index1]
            HQ1_path = self.paths_HQ_RealOld[random_index1]
            
            random_index2 = random.randint(0, self.paths_RealOld_len-1)
            LQ2_path = self.paths_LQ_RealOld[random_index2]
            HQ2_path = self.paths_HQ_RealOld[random_index2]
            
            deg_type = 'RealOld'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
            if self.phase == 'train':
                # if the image size is too small
                #print(HQ_size)
                H, W, _ = img_HQ1.shape
                #print(H, W)
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ1 = cv2.resize(np.copy(img_HQ1), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_HQ2.shape
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ2 = cv2.resize(np.copy(img_HQ2), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_LQ1.shape
                #print(H, W)
                if H < self.HQ_size or W < self.HQ_size:
                    img_LQ1 = cv2.resize(np.copy(img_LQ1), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_LQ2.shape
                if H < self.HQ_size or W < self.HQ_size:
                    img_LQ2 = cv2.resize(np.copy(img_LQ2), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
        
        else:
            print('Error! Undefined dataset: {}'.format(dataset_choice))
            exit()
        
        if self.phase == 'train':
            scale = 1
            # randomly crop to designed size
            H1, W1, C = img_LQ1.shape
            LQ_size = self.HQ_size // scale
            rnd_h = random.randint(0, max(0, H1 - LQ_size))
            rnd_w = random.randint(0, max(0, W1 - LQ_size))
            img_LQ1 = img_LQ1[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_HQ, rnd_w_HQ = int(rnd_h * scale), int(rnd_w * scale)
            img_HQ1 = img_HQ1[rnd_h_HQ:rnd_h_HQ + self.HQ_size, rnd_w_HQ:rnd_w_HQ + self.HQ_size, :]
            
            H2, W2, C = img_LQ2.shape
            LQ_size = self.HQ_size // scale
            rnd_h = random.randint(0, max(0, H2 - LQ_size))
            rnd_w = random.randint(0, max(0, W2 - LQ_size))
            img_LQ2 = img_LQ2[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_HQ, rnd_w_HQ = int(rnd_h * scale), int(rnd_w * scale)
            img_HQ2 = img_HQ2[rnd_h_HQ:rnd_h_HQ + self.HQ_size, rnd_w_HQ:rnd_w_HQ + self.HQ_size, :]

            # augmentation - flip, rotate
            img_LQ1, img_LQ2, img_HQ1, img_HQ2 = util.augment([img_LQ1, img_LQ2, img_HQ1, img_HQ2], hflip=True, rot=True)
        
    
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        target_img2 = img_HQ2
        
        batch = torch.stack([input_query_img1, target_img1, input_query_img2, target_img2], dim=0)

        return batch, deg_type

class DatasetLowlevel_Train_onlyrestore(Dataset):
    def __init__(self, dataset_path, input_size, phase, ITS_path=None, LOL_path=None, Rain13K_path=None,
                 GoPro_path=None, FiveK_path=None, LLF_path=None, RealOld_path=None):
        
        np.random.seed(5)
        self.HQ_size = input_size
        self.phase = phase
        
        # base dataset
        self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path)
        self.paths_base_len = len(self.paths_HQ)
        
        # dehaze dataset (ITS)
        if ITS_path is not None:
            self.dataset_path_HQ_ITS = os.path.join(ITS_path, 'clear')
            self.dataset_path_LQ_ITS = os.path.join(ITS_path, 'hazy')
            self.paths_HQ_ITS, self.sizes_HQ_ITS = util.get_image_paths('img', self.dataset_path_HQ_ITS)
            self.paths_LQ_ITS, self.sizes_LQ_ITS = util.get_image_paths('img', self.dataset_path_LQ_ITS)
            self.paths_ITS_len = len(self.paths_LQ_ITS)
        
        # low-light enhancement dataset (LOL)
        if LOL_path is not None:
            self.dataset_path_HQ_LOL = os.path.join(LOL_path, 'high')
            self.dataset_path_LQ_LOL = os.path.join(LOL_path, 'low')
            self.paths_HQ_LOL, self.sizes_HQ_LOL = util.get_image_paths('img', self.dataset_path_HQ_LOL)
            self.paths_LQ_LOL, self.sizes_LQ_LOL = util.get_image_paths('img', self.dataset_path_LQ_LOL)
            self.paths_LOL_len = len(self.paths_LQ_LOL)
            sorted(self.paths_HQ_LOL)
            sorted(self.paths_LQ_LOL)
            
        # Derain dataset (Rain13K)
        if Rain13K_path is not None:
            self.dataset_path_HQ_Rain13K = os.path.join(Rain13K_path, 'target')
            self.dataset_path_LQ_Rain13K = os.path.join(Rain13K_path, 'input')
            self.paths_HQ_Rain13K, self.sizes_HQ_Rain13K = util.get_image_paths('img', self.dataset_path_HQ_Rain13K)
            self.paths_LQ_Rain13K, self.sizes_LQ_Rain13K = util.get_image_paths('img', self.dataset_path_LQ_Rain13K)
            self.paths_Rain13K_len = len(self.paths_LQ_Rain13K)
            sorted(self.paths_HQ_Rain13K)
            sorted(self.paths_LQ_Rain13K)
            
        # Motion Deblur dataset (GoPro)
        if GoPro_path is not None:
            self.dataset_path_HQ_GoPro = os.path.join(GoPro_path, 'groundtruth')
            self.dataset_path_LQ_GoPro = os.path.join(GoPro_path, 'input')
            self.paths_HQ_GoPro, self.sizes_HQ_GoPro = util.get_image_paths('img', self.dataset_path_HQ_GoPro)
            self.paths_LQ_GoPro, self.sizes_LQ_GoPro = util.get_image_paths('img', self.dataset_path_LQ_GoPro)
            self.paths_GoPro_len = len(self.paths_LQ_GoPro)
            sorted(self.paths_HQ_GoPro)
            sorted(self.paths_LQ_GoPro)
            
        # Image Retouching dataset (MIT-Adobe FiveK)
        if FiveK_path is not None:
            self.dataset_path_HQ_FiveK = os.path.join(FiveK_path, 'expert_C_train')
            self.dataset_path_LQ_FiveK = os.path.join(FiveK_path, 'raw_input_train_png')
            self.paths_HQ_FiveK, self.sizes_HQ_FiveK = util.get_image_paths('img', self.dataset_path_HQ_FiveK)
            self.paths_LQ_FiveK, self.sizes_LQ_FiveK = util.get_image_paths('img', self.dataset_path_LQ_FiveK)
            self.paths_FiveK_len = len(self.paths_LQ_FiveK)
            sorted(self.paths_HQ_FiveK)
            sorted(self.paths_LQ_FiveK)
            
        
        # Local Laplacian Filter dataset (MIT-Adobe FiveK - LLF)
        if LLF_path is not None:
            self.dataset_path_HQ_LLF = os.path.join(LLF_path, 'expert_C_LLF_GT_train')
            self.dataset_path_LQ_LLF = os.path.join(LLF_path, 'expert_C_train')
            self.paths_HQ_LLF, self.sizes_HQ_LLF = util.get_image_paths('img', self.dataset_path_HQ_LLF)
            self.paths_LQ_LLF, self.sizes_LQ_LLF = util.get_image_paths('img', self.dataset_path_LQ_LLF)
            self.paths_LLF_len = len(self.paths_LQ_LLF)
            sorted(self.paths_HQ_LLF)
            sorted(self.paths_LQ_LLF)
            
        # RealOld dataset
        if RealOld_path is not None:
            self.dataset_path_HQ_RealOld = os.path.join(RealOld_path, 'HQ')
            self.dataset_path_LQ_RealOld = os.path.join(RealOld_path, 'LQ')
            self.paths_HQ_RealOld, self.sizes_HQ_RealOld = util.get_image_paths('img', self.dataset_path_HQ_RealOld)
            self.paths_LQ_RealOld, self.sizes_LQ_RealOld = util.get_image_paths('img', self.dataset_path_LQ_RealOld)
            self.paths_RealOld_len = len(self.paths_LQ_RealOld)
            sorted(self.paths_HQ_RealOld)
            sorted(self.paths_LQ_RealOld)
        
        #self.dataset_list = ['Base', 'ITS', 'LOL', 'Rain13K', 'RealOld', 'FiveK', 'LLF']
        self.dataset_list = ['Base', 'ITS', 'Rain13K']
        
        #self.dataset_list = ['Base', 'ITS']
        print('dataset list: ', self.dataset_list)
        
        self.degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG',
                                    'Rain', 'SPNoise', 'PoissonNoise', 'Ringing',
                                    'r_l', 'Inpainting']
        
        print('degradation_type list: ', self.degradation_type_list)
        
    def __len__(self):
        return len(self.paths_HQ) #+ len(self.paths_LQ_ITS) + len(self.paths_HQ_LOL) + len(self.paths_HQ_Rain13K)


    def __getitem__(self, idx):
        #dataset_choice = np.random.choice(self.dataset_list, p=[14/20, 1/20, 1/20, 1/20, 1/20, 1/20, 1/20])
        #dataset_choice = np.random.choice(self.dataset_list, p=[1/2, 1/2])
        dataset_choice = np.random.choice(self.dataset_list, p=[9/11, 1/11, 1/11])
        
        if dataset_choice == 'Base':
            random_index1 = random.randint(0, self.paths_base_len-1)
            HQ1_path = self.paths_HQ[random_index1]
            random_index2 = random.randint(0, self.paths_base_len-1)
            HQ2_path = self.paths_HQ[random_index2]
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_HQ2 = util.read_img(None, HQ2_path, None)
            
            if self.phase == 'train':
                # if the image size is too small
                #print(HQ_size)
                H, W, _ = img_HQ1.shape
                #print(H, W)
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ1 = cv2.resize(np.copy(img_HQ1), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_HQ2.shape
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ2 = cv2.resize(np.copy(img_HQ2), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                if img_HQ1.ndim == 2:
                    img_HQ1 = np.expand_dims(img_HQ1, axis=2)
                    img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
                if img_HQ2.ndim == 2:
                    img_HQ2 = np.expand_dims(img_HQ2, axis=2)
                    img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
                    
                if img_HQ1.shape[2] !=3:
                    img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
                if img_HQ2.shape[2] !=3:
                    img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
            
            
            # add degradation to HQ
            
            
            degradation_type_1 = ['LowLight', '']
            degradation_type_2 = ['GaussianBlur', 'Ringing', 'r_l', '']
            degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', '']
            degradation_type_4 = ['JPEG', '']
            degradation_type_5 = ['Inpainting', 'Rain', '']
            
            round_select = np.random.choice(['1', 'Single'], p=[4/5, 1/5])
            #round_select = np.random.choice(['1', 'Single'], p=[0, 1])
            
            if round_select == '1':
                # 1 round
                deg_type1 = random.choice(degradation_type_1)
                deg_type2 = random.choice(degradation_type_2)
                deg_type3 = random.choice(degradation_type_3)
                deg_type4 = random.choice(degradation_type_4)
                deg_type5 = random.choice(degradation_type_5)
                img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_HQ1), np.copy(img_HQ2), deg_type1)
                img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type2)
                img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type3)
                img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type4)
                img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type5)
                deg_type = 'R1_'+deg_type1+'_'+deg_type2+'_'+deg_type3+'_'+deg_type4+'_'+deg_type5
                # print(deg_type)
                # print(img_HQ1.shape)
                # print(img_LQ1.shape)
                # print(img_HQ2.shape)
                # print(img_LQ2.shape)
            elif round_select == 'Single':
                deg_type1 = random.choice(self.degradation_type_list)
                img_LQ1, img_LQ2, img_HQ1, img_HQ2 = add_degradation_two_images(img_HQ1, img_HQ2, deg_type1)
                deg_type = deg_type1
                
            
   
            
        elif dataset_choice == 'ITS':
            random_index1 = random.randint(0, self.paths_ITS_len-1)
            LQ1_path = self.paths_LQ_ITS[random_index1]
            HQ1_name = LQ1_path.split('/')[-1].split('_')[0]
            HQ1_path = os.path.join(self.dataset_path_HQ_ITS, '{}.png'.format(HQ1_name))
            
            random_index2 = random.randint(0, self.paths_ITS_len-1)
            LQ2_path = self.paths_LQ_ITS[random_index2]
            HQ2_name = LQ2_path.split('/')[-1].split('_')[0]
            HQ2_path = os.path.join(self.dataset_path_HQ_ITS, '{}.png'.format(HQ2_name))
            deg_type = 'ITS'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
            # H_GT, W_GT, _ = img_HQ1.shape
            # H_LQ, W_LQ, _ = img_LQ1.shape
            
            # crop_size_H = np.abs(H_LQ-H_GT)//2
            # crop_size_W = np.abs(W_LQ-W_GT)//2
            # img_HQ1 = img_HQ1[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
            # img_HQ2 = img_HQ2[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
        
        elif dataset_choice == 'LOL':
            random_index1 = random.randint(0, self.paths_LOL_len-1)
            LQ1_path = self.paths_LQ_LOL[random_index1]
            HQ1_path = self.paths_HQ_LOL[random_index1]
            
            random_index2 = random.randint(0, self.paths_LOL_len-1)
            LQ2_path = self.paths_LQ_LOL[random_index2]
            HQ2_path = self.paths_HQ_LOL[random_index2]
            
            deg_type = 'LOL'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
        
        elif dataset_choice == 'GoPro':
            random_index1 = random.randint(0, self.paths_GoPro_len-1)
            LQ1_path = self.paths_LQ_GoPro[random_index1]
            HQ1_path = self.paths_HQ_GoPro[random_index1]
            
            random_index2 = random.randint(0, self.paths_GoPro_len-1)
            LQ2_path = self.paths_LQ_GoPro[random_index2]
            HQ2_path = self.paths_HQ_GoPro[random_index2]
            
            deg_type = 'GoPro'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
        
        elif dataset_choice == 'FiveK':
            random_index1 = random.randint(0, self.paths_FiveK_len-1)
            LQ1_path = self.paths_LQ_FiveK[random_index1]
            HQ1_path = self.paths_HQ_FiveK[random_index1]
            
            random_index2 = random.randint(0, self.paths_FiveK_len-1)
            LQ2_path = self.paths_LQ_FiveK[random_index2]
            HQ2_path = self.paths_HQ_FiveK[random_index2]
            
            deg_type = 'FiveK'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
        
        elif dataset_choice == 'LLF':
            random_index1 = random.randint(0, self.paths_LLF_len-1)
            LQ1_path = self.paths_LQ_LLF[random_index1]
            HQ1_path = self.paths_HQ_LLF[random_index1]
            
            random_index2 = random.randint(0, self.paths_LLF_len-1)
            LQ2_path = self.paths_LQ_LLF[random_index2]
            HQ2_path = self.paths_HQ_LLF[random_index2]
            
            deg_type = 'LLF'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
        elif dataset_choice == 'Rain13K':
            random_index1 = random.randint(0, self.paths_Rain13K_len-1)
            LQ1_path = self.paths_LQ_Rain13K[random_index1]
            HQ1_path = self.paths_HQ_Rain13K[random_index1]
            
            random_index2 = random.randint(0, self.paths_Rain13K_len-1)
            LQ2_path = self.paths_LQ_Rain13K[random_index2]
            HQ2_path = self.paths_HQ_Rain13K[random_index2]
            
            deg_type = 'Rain13K'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
            if self.phase == 'train':
                # if the image size is too small
                #print(HQ_size)
                H, W, _ = img_HQ1.shape
                #print(H, W)
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ1 = cv2.resize(np.copy(img_HQ1), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_HQ2.shape
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ2 = cv2.resize(np.copy(img_HQ2), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_LQ1.shape
                #print(H, W)
                if H < self.HQ_size or W < self.HQ_size:
                    img_LQ1 = cv2.resize(np.copy(img_LQ1), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_LQ2.shape
                if H < self.HQ_size or W < self.HQ_size:
                    img_LQ2 = cv2.resize(np.copy(img_LQ2), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
        
        elif dataset_choice == 'RealOld':
            random_index1 = random.randint(0, self.paths_RealOld_len-1)
            LQ1_path = self.paths_LQ_RealOld[random_index1]
            HQ1_path = self.paths_HQ_RealOld[random_index1]
            
            random_index2 = random.randint(0, self.paths_RealOld_len-1)
            LQ2_path = self.paths_LQ_RealOld[random_index2]
            HQ2_path = self.paths_HQ_RealOld[random_index2]
            
            deg_type = 'RealOld'
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
            if self.phase == 'train':
                # if the image size is too small
                #print(HQ_size)
                H, W, _ = img_HQ1.shape
                #print(H, W)
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ1 = cv2.resize(np.copy(img_HQ1), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_HQ2.shape
                if H < self.HQ_size or W < self.HQ_size:
                    img_HQ2 = cv2.resize(np.copy(img_HQ2), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_LQ1.shape
                #print(H, W)
                if H < self.HQ_size or W < self.HQ_size:
                    img_LQ1 = cv2.resize(np.copy(img_LQ1), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
                    
                H, W, _ = img_LQ2.shape
                if H < self.HQ_size or W < self.HQ_size:
                    img_LQ2 = cv2.resize(np.copy(img_LQ2), (self.HQ_size, self.HQ_size),
                                        interpolation=cv2.INTER_LINEAR)
        
        else:
            print('Error! Undefined dataset: {}'.format(dataset_choice))
            exit()
        
        if self.phase == 'train':
            scale = 1
            # randomly crop to designed size
            H1, W1, C = img_LQ1.shape
            LQ_size = self.HQ_size // scale
            rnd_h = random.randint(0, max(0, H1 - LQ_size))
            rnd_w = random.randint(0, max(0, W1 - LQ_size))
            img_LQ1 = img_LQ1[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_HQ, rnd_w_HQ = int(rnd_h * scale), int(rnd_w * scale)
            img_HQ1 = img_HQ1[rnd_h_HQ:rnd_h_HQ + self.HQ_size, rnd_w_HQ:rnd_w_HQ + self.HQ_size, :]
            
            H2, W2, C = img_LQ2.shape
            LQ_size = self.HQ_size // scale
            rnd_h = random.randint(0, max(0, H2 - LQ_size))
            rnd_w = random.randint(0, max(0, W2 - LQ_size))
            img_LQ2 = img_LQ2[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_HQ, rnd_w_HQ = int(rnd_h * scale), int(rnd_w * scale)
            img_HQ2 = img_HQ2[rnd_h_HQ:rnd_h_HQ + self.HQ_size, rnd_w_HQ:rnd_w_HQ + self.HQ_size, :]

            # augmentation - flip, rotate
            img_LQ1, img_LQ2, img_HQ1, img_HQ2 = util.augment([img_LQ1, img_LQ2, img_HQ1, img_HQ2], hflip=True, rot=True)
        
    
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        target_img2 = img_HQ2
        
        batch = torch.stack([input_query_img1, target_img1, input_query_img2, target_img2], dim=0)

        return batch, deg_type
    
class DatasetLowlevel_Val(Dataset):
    def __init__(self, dataset_path, input_size=320):
        self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path)
        self.input_size = input_size

    def __len__(self):
        return len(self.paths_HQ)-1


    def __getitem__(self, idx):
        HQ1_path = self.paths_HQ[idx]
        random_index = random.randint(0, len(self.paths_HQ)-1)
        HQ2_path = self.paths_HQ[random_index]
        
        img_HQ1 = util.read_img(None, HQ1_path, None)
        img_HQ2 = util.read_img(None, HQ2_path, None)
        
        #img_HQ1 = util.modcrop(img_HQ1, 16)
        #img_HQ2 = util.modcrop(img_HQ2, 16)
        
        img_HQ1 = cv2.resize(np.copy(img_HQ1), (self.input_size, self.input_size),
                                    interpolation=cv2.INTER_LINEAR)
        img_HQ2 = cv2.resize(np.copy(img_HQ2), (self.input_size, self.input_size),
                                    interpolation=cv2.INTER_LINEAR)
        
        if img_HQ1.ndim == 2:
            img_HQ1 = np.expand_dims(img_HQ1, axis=2)
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.ndim == 2:
            img_HQ2 = np.expand_dims(img_HQ2, axis=2)
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
            
        if img_HQ1.shape[2] !=3:
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.shape[2] !=3:
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        
        # add degradation to HQ
        degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'Resize', 
                                    'Rain', 'SPNoise', 'LowLight', 'PoissonNoise', 'Ringing',
                                    'r_l', 'Inpainting', 'Laplacian', 'Canny']
        
        degradation_type_1 = ['Resize', 'LowLight', '']
        degradation_type_2 = ['GaussianBlur', '']
        degradation_type_3 = ['GaussianNoise', 'SPNoise', 'PoissonNoise', '']
        degradation_type_4 = ['JPEG', 'Ringing', 'r_l', '']
        degradation_type_5 = ['Inpainting', '']
        degradation_type_6 = ['Rain', '']
        
        round_select = np.random.choice(['1', 'Single'], p=[4/5, 1/5])
            
        if round_select == '1':
            # 1 round
            deg_type1 = random.choice(degradation_type_1)
            deg_type2 = random.choice(degradation_type_2)
            deg_type3 = random.choice(degradation_type_3)
            deg_type4 = random.choice(degradation_type_4)
            deg_type5 = random.choice(degradation_type_5)
            deg_type6 = random.choice(degradation_type_6)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_HQ1), np.copy(img_HQ2), deg_type1)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type2)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type3)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type4)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type5)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(np.copy(img_LQ1), np.copy(img_LQ2), deg_type6)
            deg_type = 'R1_'+deg_type1+'_'+deg_type2+'_'+deg_type3+'_'+deg_type4+'_'+deg_type5+'_'+deg_type6
            # print(deg_type)
            # print(img_HQ1.shape)
            # print(img_LQ1.shape)
            # print(img_HQ2.shape)
            # print(img_LQ2.shape)
        elif round_select == 'Single':
            deg_type1 = random.choice(degradation_type_list)
            img_LQ1, img_LQ2, img_HQ1, img_HQ2 = add_degradation_two_images(img_HQ1, img_HQ2, deg_type1)
            deg_type = deg_type1
            
        elif round_select == '2':
            # 2 round (currently does not take into account, 1 round is enough)
            deg_type1 = random.choice(degradation_type_1)
            deg_type2 = random.choice(degradation_type_2)
            deg_type3 = random.choice(degradation_type_3)
            deg_type4 = random.choice(degradation_type_4)
            deg_type5 = random.choice(degradation_type_5)
            deg_type6 = random.choice(degradation_type_6)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_HQ1, img_HQ2, deg_type1)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type2)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type3)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type4)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type5)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type6)
            
            deg_type7 = random.choice(degradation_type_1)
            deg_type8 = random.choice(degradation_type_2)
            deg_type9 = random.choice(degradation_type_3)
            deg_type10 = random.choice(degradation_type_4)
            deg_type11 = random.choice(degradation_type_5)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type7)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type8)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type9)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type10)
            img_LQ1, img_LQ2, _, _ = add_degradation_two_images(img_LQ1, img_LQ2, deg_type11)
            
            deg_type = 'R2_'+deg_type1+'_'+deg_type2+'_'+deg_type3+'_'+deg_type4+'_'+deg_type5+'_'+deg_type6+'_'+ \
                        deg_type7+'_'+deg_type8+'_'+deg_type9+'_'+deg_type10+'_'+deg_type11

       
        
        
        # print(HQ1_path)
        # print(HQ2_path) 
        # print(deg_type)
        # print(img_HQ1.shape)
        # print(img_LQ1.shape)
        # print(img_HQ2.shape)
        # print(img_LQ2.shape)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        # if img_HQ1.shape[2] == 3:
        #     img_HQ1 = img_HQ1[:, :, [2, 1, 0]]
        #     img_HQ2 = img_HQ2[:, :, [2, 1, 0]]
        #     img_LQ1 = img_LQ1[:, :, [2, 1, 0]]
        #     img_LQ2 = img_LQ2[:, :, [2, 1, 0]]
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        target_img2 = img_HQ2
        
        batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                 'input_query_img2': input_query_img2, 'target_img2': target_img2}
        return batch, deg_type
    

class DatasetLowlevel_Mismatched_Val(Dataset):
    def __init__(self, dataset_path):
        self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path)


    def __len__(self):
        return len(self.paths_HQ)-1


    def __getitem__(self, idx):
        HQ1_path = self.paths_HQ[idx]
        random_index = random.randint(0, len(self.paths_HQ)-1)
        HQ2_path = self.paths_HQ[random_index]
        
        img_HQ1 = util.read_img(None, HQ1_path, None)
        img_HQ2 = util.read_img(None, HQ2_path, None)
        
        #img_HQ1 = util.modcrop(img_HQ1, 16)
        #img_HQ2 = util.modcrop(img_HQ2, 16)
        
        img_HQ1 = cv2.resize(np.copy(img_HQ1), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        img_HQ2 = cv2.resize(np.copy(img_HQ2), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        
        if img_HQ1.ndim == 2:
            img_HQ1 = np.expand_dims(img_HQ1, axis=2)
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.ndim == 2:
            img_HQ2 = np.expand_dims(img_HQ2, axis=2)
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
            
        if img_HQ1.shape[2] !=3:
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.shape[2] !=3:
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        
        # add degradation to HQ
        degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'Resize', 
                                 'Rain', 'SPNoise', 'LowLight', 'PoissonNoise', 'Ringing',
                                 'r_l', 'Inpainting']
        prompt_deg_type = random.choice(degradation_type_list)
        if prompt_deg_type == 'GaussianNoise':
            level = random.uniform(10, 50)
            img_LQ1 = add_Gaussian_noise(img_HQ1.copy(), level=level)
        elif prompt_deg_type == 'GaussianBlur':
            sigma = random.uniform(1, 4)
            img_LQ1 = iso_GaussianBlur(img_HQ1.copy(), window=15, sigma=sigma)
        elif prompt_deg_type == 'JPEG':
            level = random.randint(20, 95)
            img_LQ1 = add_JPEG_noise(img_HQ1.copy(), level=level)
        elif prompt_deg_type == 'Resize':
            img_LQ1 = add_resize(img_HQ1.copy())
        elif prompt_deg_type == 'Rain':
            value = random.uniform(40, 200)
            img_LQ1 = add_rain(img_HQ1.copy(), value=value)
        elif prompt_deg_type == 'SPNoise':
            img_LQ1 = add_sp_noise(img_HQ1.copy())
        elif prompt_deg_type == 'LowLight':
            lum_scale = random.uniform(0.1, 0.6)
            img_LQ1 = low_light(img_HQ1.copy(), lum_scale=lum_scale)
        elif prompt_deg_type == 'PoissonNoise':
            img_LQ1 = add_Poisson_noise(img_HQ1.copy(), level=2)
        elif prompt_deg_type == 'Ringing':
            img_LQ1 = add_ringing(img_HQ1.copy())
        elif prompt_deg_type == 'r_l':
            img_LQ1 = r_l(img_HQ1.copy())
        elif prompt_deg_type == 'Inpainting':
            l_num = random.randint(5, 10)
            l_thick = random.randint(5, 10)
            img_LQ1 = inpainting(img_HQ1.copy(), l_num=l_num, l_thick=l_thick)
        else:
            print('Error!')
            exit()
       
        test_deg_type = random.choice(degradation_type_list)
        if test_deg_type == 'GaussianNoise':
            level = random.uniform(10, 50)
            img_LQ2 = add_Gaussian_noise(img_HQ2.copy(), level=level)
        elif test_deg_type == 'GaussianBlur':
            sigma = random.uniform(1, 4)
            img_LQ2 = iso_GaussianBlur(img_HQ2.copy(), window=15, sigma=sigma)
        elif test_deg_type == 'JPEG':
            level = random.randint(20, 95)
            img_LQ2 = add_JPEG_noise(img_HQ2.copy(), level=level)
        elif test_deg_type == 'Resize':
            img_LQ2 = add_resize(img_HQ2.copy())
        elif test_deg_type == 'Rain':
            value = random.uniform(40, 200)
            img_LQ2 = add_rain(img_HQ2.copy(), value=value)
        elif test_deg_type == 'SPNoise':
            img_LQ2 = add_sp_noise(img_HQ2.copy())
        elif test_deg_type == 'LowLight':
            lum_scale = random.uniform(0.1, 0.6)
            img_LQ2 = low_light(img_HQ2.copy(), lum_scale=lum_scale)
        elif test_deg_type == 'PoissonNoise':
            img_LQ2 = add_Poisson_noise(img_HQ2.copy(), level=2)
        elif test_deg_type == 'Ringing':
            img_LQ2 = add_ringing(img_HQ2.copy())
        elif test_deg_type == 'r_l':
            img_LQ2 = r_l(img_HQ2.copy())
        elif test_deg_type == 'Inpainting':
            l_num = random.randint(5, 10)
            l_thick = random.randint(5, 10)
            img_LQ2 = inpainting(img_HQ2.copy(), l_num=l_num, l_thick=l_thick)
        else:
            print('Error!')
            exit()
        
        # print(HQ1_path)
        # print(HQ2_path) 
        # print(deg_type)
        # print(img_HQ1.shape)
        # print(img_LQ1.shape)
        # print(img_HQ2.shape)
        # print(img_LQ2.shape)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        # if img_HQ1.shape[2] == 3:
        #     img_HQ1 = img_HQ1[:, :, [2, 1, 0]]
        #     img_HQ2 = img_HQ2[:, :, [2, 1, 0]]
        #     img_LQ1 = img_LQ1[:, :, [2, 1, 0]]
        #     img_LQ2 = img_LQ2[:, :, [2, 1, 0]]
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        target_img2 = img_HQ2
        
        batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                 'input_query_img2': input_query_img2, 'target_img2': target_img2}
        return batch, prompt_deg_type, test_deg_type
    
class DatasetLowlevel_Mix_Val(Dataset):
    def __init__(self, dataset_path):
        self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path)


    def __len__(self):
        return len(self.paths_HQ)-1


    def __getitem__(self, idx):
        HQ1_path = self.paths_HQ[idx]
        random_index = random.randint(0, len(self.paths_HQ)-1)
        HQ2_path = self.paths_HQ[random_index]
        
        img_HQ1 = util.read_img(None, HQ1_path, None)
        img_HQ2 = util.read_img(None, HQ2_path, None)
        
        #img_HQ1 = util.modcrop(img_HQ1, 16)
        #img_HQ2 = util.modcrop(img_HQ2, 16)
        
        img_HQ1 = cv2.resize(np.copy(img_HQ1), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        img_HQ2 = cv2.resize(np.copy(img_HQ2), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        
        if img_HQ1.ndim == 2:
            img_HQ1 = np.expand_dims(img_HQ1, axis=2)
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.ndim == 2:
            img_HQ2 = np.expand_dims(img_HQ2, axis=2)
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
            
        if img_HQ1.shape[2] !=3:
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.shape[2] !=3:
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        
        # add degradation to HQ
        degradation_type_list = ['GaussianBlur+GaussianNoise', 'GaussianBlur+SPNoise', 'GaussianBlur+LowLight', 
                                 'GaussianBlur+Rain',  'Ringing+GaussianNoise',
                                 'r_l+GaussianNoise', 'GaussianNoise+Inpainting']
        deg_type = random.choice(degradation_type_list)
        if deg_type == 'GaussianBlur+GaussianNoise':
            sigma = random.uniform(1, 4)
            img_LQ1 = iso_GaussianBlur(img_HQ1.copy(), window=15, sigma=sigma)
            img_LQ2 = iso_GaussianBlur(img_HQ2.copy(), window=15, sigma=sigma)
            level = random.uniform(10, 50)
            img_LQ1 = add_Gaussian_noise(img_LQ1.copy(), level=level)
            img_LQ2 = add_Gaussian_noise(img_LQ2.copy(), level=level)
        elif deg_type == 'GaussianBlur+SPNoise':
            sigma = random.uniform(1, 4)
            img_LQ1 = iso_GaussianBlur(img_HQ1.copy(), window=15, sigma=sigma)
            img_LQ2 = iso_GaussianBlur(img_HQ2.copy(), window=15, sigma=sigma)
            img_LQ1 = add_sp_noise(img_LQ1.copy())
            img_LQ2 = add_sp_noise(img_LQ2.copy())
        elif deg_type == 'GaussianBlur+Rain':
            sigma = random.uniform(1, 4)
            img_LQ1 = iso_GaussianBlur(img_HQ1.copy(), window=15, sigma=sigma)
            img_LQ2 = iso_GaussianBlur(img_HQ2.copy(), window=15, sigma=sigma)
            value = random.uniform(40, 200)
            img_LQ1 = add_rain(img_LQ1.copy(), value=value)
            img_LQ2 = add_rain(img_LQ2.copy(), value=value)
        elif deg_type == 'SPNoise':
            img_LQ1 = add_sp_noise(img_HQ1.copy())
            img_LQ2 = add_sp_noise(img_HQ2.copy())
        elif deg_type == 'GaussianBlur+LowLight':
            sigma = random.uniform(1, 4)
            img_LQ1 = iso_GaussianBlur(img_HQ1.copy(), window=15, sigma=sigma)
            img_LQ2 = iso_GaussianBlur(img_HQ2.copy(), window=15, sigma=sigma)
            lum_scale = random.uniform(0.1, 0.6)
            img_LQ1 = low_light(img_LQ1.copy(), lum_scale=lum_scale)
            img_LQ2 = low_light(img_LQ2.copy(), lum_scale=lum_scale)
        elif deg_type == 'PoissonNoise':
            img_LQ1 = add_Poisson_noise(img_HQ1.copy(), level=2)
            img_LQ2 = add_Poisson_noise(img_HQ2.copy(), level=2)
        elif deg_type == 'Ringing+GaussianNoise':
            img_LQ1 = add_ringing(img_HQ1.copy())
            img_LQ2 = add_ringing(img_HQ2.copy())
            level = random.uniform(10, 50)
            img_LQ1 = add_Gaussian_noise(img_LQ1.copy(), level=level)
            img_LQ2 = add_Gaussian_noise(img_LQ2.copy(), level=level)
        elif deg_type == 'r_l+GaussianNoise':
            img_LQ1 = r_l(img_HQ1.copy())
            img_LQ2 = r_l(img_HQ2.copy())
            level = random.uniform(10, 50)
            img_LQ1 = add_Gaussian_noise(img_LQ1.copy(), level=level)
            img_LQ2 = add_Gaussian_noise(img_LQ2.copy(), level=level)
        elif deg_type == 'GaussianNoise+Inpainting':
            level = random.uniform(10, 50)
            img_LQ1 = add_Gaussian_noise(img_HQ1.copy(), level=level)
            img_LQ2 = add_Gaussian_noise(img_HQ2.copy(), level=level)
            l_num = random.randint(5, 10)
            l_thick = random.randint(5, 10)
            img_LQ1 = inpainting(img_LQ1.copy(), l_num=l_num, l_thick=l_thick)
            img_LQ2 = inpainting(img_LQ2.copy(), l_num=l_num, l_thick=l_thick)
        else:
            print('Error!')
            exit()
       
        
        
        # print(HQ1_path)
        # print(HQ2_path) 
        # print(deg_type)
        # print(img_HQ1.shape)
        # print(img_LQ1.shape)
        # print(img_HQ2.shape)
        # print(img_LQ2.shape)
        
        # BGR to RGB, HWC to CHW, numpy to tensor
        # if img_HQ1.shape[2] == 3:
        #     img_HQ1 = img_HQ1[:, :, [2, 1, 0]]
        #     img_HQ2 = img_HQ2[:, :, [2, 1, 0]]
        #     img_LQ1 = img_LQ1[:, :, [2, 1, 0]]
        #     img_LQ2 = img_LQ2[:, :, [2, 1, 0]]
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        target_img2 = img_HQ2
        
        batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                 'input_query_img2': input_query_img2, 'target_img2': target_img2}
        return batch, deg_type
    
class DatasetLowlevel_Customized_Val(Dataset):
    def __init__(self, dataset_path_HQ, dataset_path_LQ, dataset_type='SOTS', data_len=None):
        self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path_HQ)
        self.paths_LQ, self.sizes_LQ = util.get_image_paths('img', dataset_path_LQ)
        self.dataset_path_HQ = dataset_path_HQ
        self.dataset_path_LQ = dataset_path_LQ
        sorted(self.paths_HQ)
        sorted(self.paths_LQ)
        self.dataset_type = dataset_type
        self.data_len = data_len
        
        if self.data_len is not None:
            self.paths_HQ = self.paths_HQ[0:self.data_len]
            self.paths_LQ = self.paths_LQ[0:self.data_len]

    def __len__(self):
        
        return len(self.paths_LQ)


    def __getitem__(self, idx):
        if self.dataset_type == 'SOTS':
            LQ1_path = self.paths_LQ[idx]
            HQ1_name = LQ1_path.split('/')[-1].split('_')[0]
            HQ1_path = os.path.join(self.dataset_path_HQ, '{}.png'.format(HQ1_name))
            
            random_index = random.randint(0, len(self.paths_HQ)-1)
            LQ2_path = self.paths_LQ[random_index]
            HQ2_name = LQ2_path.split('/')[-1].split('_')[0]
            HQ2_path = os.path.join(self.dataset_path_HQ, '{}.png'.format(HQ2_name))
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
            H_GT, W_GT, _ = img_HQ1.shape
            H_LQ, W_LQ, _ = img_LQ1.shape
            
            crop_size_H = np.abs(H_LQ-H_GT)//2
            crop_size_W = np.abs(W_LQ-W_GT)//2
            img_HQ1 = img_HQ1[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
            img_HQ2 = img_HQ2[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
        else:
            random_index = random.randint(0, len(self.paths_HQ)-1)
            HQ1_path = self.paths_HQ[random_index]
            LQ1_path = self.paths_LQ[random_index]
            
            HQ2_path = self.paths_HQ[idx]
            LQ2_path = self.paths_LQ[idx]
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
        
        
        deg_type = self.dataset_type
        #img_HQ1 = util.modcrop(img_HQ1, 16)
        #img_HQ2 = util.modcrop(img_HQ2, 16)
        
        img_HQ1 = cv2.resize(np.copy(img_HQ1), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        img_LQ1 = cv2.resize(np.copy(img_LQ1), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        img_HQ2 = cv2.resize(np.copy(img_HQ2), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        img_LQ2 = cv2.resize(np.copy(img_LQ2), (256, 256),
                                    interpolation=cv2.INTER_LINEAR)
        
        if img_HQ1.ndim == 2:
            img_HQ1 = np.expand_dims(img_HQ1, axis=2)
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.ndim == 2:
            img_HQ2 = np.expand_dims(img_HQ2, axis=2)
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
            
        if img_HQ1.shape[2] !=3:
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.shape[2] !=3:
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        
        
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        target_img2 = img_HQ2
        
        batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                 'input_query_img2': input_query_img2, 'target_img2': target_img2,
                 'input_query_img2_path': LQ2_path}
        return batch, deg_type

class DatasetLowlevel_Customized_Val_original_size(Dataset):
    def __init__(self, dataset_path_HQ, dataset_path_LQ, dataset_type='SOTS', data_len=None, prompt_id=-1):
        self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path_HQ)
        self.paths_LQ, self.sizes_LQ = util.get_image_paths('img', dataset_path_LQ)
        self.dataset_path_HQ = dataset_path_HQ
        self.dataset_path_LQ = dataset_path_LQ
        sorted(self.paths_HQ)
        sorted(self.paths_LQ)
        self.dataset_type = dataset_type
        self.data_len = data_len
        self.prompt_id = prompt_id
        
        print(self.prompt_id, self.paths_LQ[self.prompt_id])
        
        if self.data_len is not None:
            self.paths_HQ = self.paths_HQ[0:self.data_len]
            self.paths_LQ = self.paths_LQ[0:self.data_len]

    def __len__(self):
        
        return len(self.paths_LQ)


    def __getitem__(self, idx):
        if self.dataset_type == 'SOTS':
            if self.prompt_id == -1:
                random_index = random.randint(0, len(self.paths_LQ)-1)
                LQ1_path = self.paths_LQ[random_index]
            else:
                LQ1_path = self.paths_LQ[self.prompt_id]
            
            
            HQ1_name = LQ1_path.split('/')[-1].split('_')[0]
            HQ1_path = os.path.join(self.dataset_path_HQ, '{}.png'.format(HQ1_name))
            
            LQ2_path = self.paths_LQ[idx]
            HQ2_name = LQ2_path.split('/')[-1].split('_')[0]
            HQ2_path = os.path.join(self.dataset_path_HQ, '{}.png'.format(HQ2_name))
            
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
            H_GT, W_GT, _ = img_HQ1.shape
            H_LQ, W_LQ, _ = img_LQ1.shape
            
            if H_GT != H_LQ or W_GT != W_LQ:
                crop_size_H = np.abs(H_LQ-H_GT)//2
                crop_size_W = np.abs(W_LQ-W_GT)//2
                img_HQ1 = img_HQ1[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
                img_HQ2 = img_HQ2[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
        else:
            if self.prompt_id == -1:
                random_index = random.randint(0, len(self.paths_HQ)-1)
                HQ1_path = self.paths_HQ[random_index]
                LQ1_path = self.paths_LQ[random_index]
            else:
                
                HQ1_path = self.paths_HQ[self.prompt_id]
                LQ1_path = self.paths_LQ[self.prompt_id]
                
                # print(self.prompt_id, HQ1_path)
                # print(self.prompt_id, LQ1_path)
            
            HQ2_path = self.paths_HQ[idx]
            LQ2_path = self.paths_LQ[idx]
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
        
        
        deg_type = self.dataset_type
        #img_HQ1 = util.modcrop(img_HQ1, 16)
        #img_HQ2 = util.modcrop(img_HQ2, 16)
        
        # img_size = 320
        # img_HQ1 = cv2.resize(np.copy(img_HQ1), (img_size, img_size),
        #                             interpolation=cv2.INTER_LINEAR)
        # img_LQ1 = cv2.resize(np.copy(img_LQ1), (img_size, img_size),
        #                             interpolation=cv2.INTER_LINEAR)
        # img_HQ2 = cv2.resize(np.copy(img_HQ2), (img_size, img_size),
        #                             interpolation=cv2.INTER_LINEAR)
        # img_LQ2 = cv2.resize(np.copy(img_LQ2), (img_size, img_size),
        #                             interpolation=cv2.INTER_LINEAR)
        
        if img_HQ1.ndim == 2:
            img_HQ1 = np.expand_dims(img_HQ1, axis=2)
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.ndim == 2:
            img_HQ2 = np.expand_dims(img_HQ2, axis=2)
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
            
        if img_HQ1.shape[2] !=3:
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_HQ2.shape[2] !=3:
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        
        
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        target_img2 = img_HQ2
        
        batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                 'input_query_img2': input_query_img2, 'target_img2': target_img2,
                 'input_query_img2_path': LQ2_path}
        return batch, deg_type
    
class DatasetLowlevel_Customized_Test(Dataset):
    def __init__(self, dataset_path_HQ, dataset_path_LQ, dataset_type='SOTS'):
        self.paths_HQ, self.sizes_HQ = util.get_image_paths('img', dataset_path_HQ)
        self.paths_LQ, self.sizes_LQ = util.get_image_paths('img', dataset_path_LQ)
        self.dataset_path_HQ = dataset_path_HQ
        self.dataset_path_LQ = dataset_path_LQ
        sorted(self.paths_HQ)
        sorted(self.paths_LQ)
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.paths_LQ)


    def __getitem__(self, idx):
        if self.dataset_type == 'SOTS':
            random_index = random.randint(0, len(self.paths_HQ)-1)
            LQ1_path = self.paths_LQ[random_index]
            HQ1_name = LQ1_path.split('/')[-1].split('_')[0]
            HQ1_path = os.path.join(self.dataset_path_HQ, '{}.png'.format(HQ1_name))
            
            
            LQ2_path = self.paths_LQ[idx]
            HQ2_name = LQ2_path.split('/')[-1].split('_')[0]
            HQ2_path = os.path.join(self.dataset_path_HQ, '{}.png'.format(HQ2_name))
            
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
            
            H_GT, W_GT, _ = img_HQ1.shape
            H_LQ, W_LQ, _ = img_LQ1.shape
            
            crop_size_H = np.abs(H_LQ-H_GT)//2
            crop_size_W = np.abs(W_LQ-W_GT)//2
            img_HQ1 = img_HQ1[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
            img_HQ2 = img_HQ2[crop_size_H:-crop_size_H, crop_size_W:-crop_size_W, :]
        
        elif self.dataset_type == 'Syn':
            HQ1_path = self.paths_HQ[2]
            img_HQ1 = util.read_img(None, HQ1_path, None)
            
            # add degradation to HQ
            degradation_type_list = ['GaussianNoise', 'GaussianBlur', 'JPEG', 'Resize', 
                                    'Rain', 'SPNoise', 'LowLight', 'PoissonNoise', 'Ringing',
                                    'r_l', 'Inpainting']
            deg_type = 'Inpainting'
            if deg_type == 'GaussianNoise':
                level = 50
                img_LQ1 = add_Gaussian_noise(img_HQ1.copy(), level=level)
            elif deg_type == 'GaussianBlur':
                sigma = 3.2739
                img_LQ1 = iso_GaussianBlur(img_HQ1.copy(), window=15, sigma=sigma)
            elif deg_type == 'JPEG':
                level = random.randint(20, 95)
                img_LQ1 = add_JPEG_noise(img_HQ1.copy(), level=level)
            elif deg_type == 'Resize':
                img_LQ1 = add_resize(img_HQ1.copy())
            elif deg_type == 'Rain':
                value = 100
                img_LQ1 = add_rain(img_HQ1.copy(), value=value)
            elif deg_type == 'SPNoise':
                img_LQ1 = add_sp_noise(img_HQ1.copy())
            elif deg_type == 'LowLight':
                lum_scale = 0.3
                img_LQ1 = low_light(img_HQ1.copy(), lum_scale=lum_scale)
            elif deg_type == 'PoissonNoise':
                img_LQ1 = add_Poisson_noise(img_HQ1.copy(), level=2)
            elif deg_type == 'Ringing':
                img_LQ1 = add_ringing(img_HQ1.copy())
            elif deg_type == 'r_l':
                img_LQ1 = r_l(img_HQ1.copy())
            elif deg_type == 'Inpainting':
                l_num = random.randint(5, 10)
                l_thick = random.randint(5, 10)
                img_LQ1 = inpainting(img_HQ1.copy(), l_num=l_num, l_thick=l_thick)
            else:
                print('Error!')
                exit()
            
            
            HQ2_path = self.paths_HQ[idx]
            LQ2_path = self.paths_LQ[idx]
            
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
        
        else:
            HQ1_path = self.paths_HQ[idx]
            LQ1_path = self.paths_LQ[idx]
            random_index = random.randint(0, len(self.paths_HQ)-1)
            HQ2_path = self.paths_HQ[random_index]
            LQ2_path = self.paths_LQ[random_index]
            img_HQ1 = util.read_img(None, HQ1_path, None)
            img_LQ1 = util.read_img(None, LQ1_path, None) 
            img_HQ2 = util.read_img(None, HQ2_path, None)
            img_LQ2 = util.read_img(None, LQ2_path, None)
        
        
        deg_type = self.dataset_type
        #img_HQ1 = util.modcrop(img_HQ1, 16)
        #img_HQ2 = util.modcrop(img_HQ2, 16)
        

        
        
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        target_img2 = img_HQ2
        
        batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                 'input_query_img2': input_query_img2, 'target_img2': target_img2}
        return batch, deg_type
    
    
class DatasetLowlevel_Customized_Test_DirectLoad_Triplet(Dataset):
    def __init__(self, dataset_path_root):
        self.dataset_path_root = dataset_path_root
        self.dataset_paths = os.listdir(self.dataset_path_root)
        sorted(self.dataset_paths)

    def __len__(self):
        return len(self.dataset_paths)


    def __getitem__(self, idx):
        LQ1_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'prompt_input_img1.png')
        HQ1_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'prompt_target_img1.png')
        LQ2_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_input_img2.png')
        if os.path.exists(os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')):
            HQ2_path = os.path.join(self.dataset_path_root, self.dataset_paths[idx], 'query_target_img2.png')
        else:
            HQ2_path = None
        
        img_HQ1 = util.read_img(None, HQ1_path, None)
        img_LQ1 = util.read_img(None, LQ1_path, None)
        
        if HQ2_path:
            img_HQ2 = util.read_img(None, HQ2_path, None)
        
        img_LQ2 = util.read_img(None, LQ2_path, None)
        
        
        
        if img_HQ1.ndim == 2:
            img_HQ1 = np.expand_dims(img_HQ1, axis=2)
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_LQ1.ndim == 2:
            img_LQ1 = np.expand_dims(img_LQ1, axis=2)
            img_LQ1 = np.concatenate((img_LQ1, img_LQ1, img_LQ1), axis=2)
        if HQ2_path and img_HQ2.ndim == 2:
            img_HQ2 = np.expand_dims(img_HQ2, axis=2)
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        if img_LQ2.ndim == 2:
            img_LQ2 = np.expand_dims(img_LQ2, axis=2)
            img_LQ2 = np.concatenate((img_LQ2, img_LQ2, img_LQ2), axis=2)
            
        if img_HQ1.shape[2] !=3:
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=2)
        if img_LQ1.shape[2] !=3:
            img_LQ1 = np.concatenate((img_LQ1, img_LQ1, img_LQ1), axis=2)
        if HQ2_path and img_HQ2.shape[2] !=3:
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=2)
        if img_LQ2.shape[2] !=3:
            img_LQ2 = np.concatenate((img_LQ2, img_LQ2, img_LQ2), axis=2)
        
        
        # img_HQ1 = cv2.resize(np.copy(img_HQ1), (112, 112),
        #                             interpolation=cv2.INTER_LINEAR)
        # img_LQ1 = cv2.resize(np.copy(img_LQ1), (112, 112),
        #                             interpolation=cv2.INTER_LINEAR)
        # if HQ2_path:
        #     img_HQ2 = cv2.resize(np.copy(img_HQ2), (112, 112),
        #                             interpolation=cv2.INTER_LINEAR)
        # img_LQ2 = cv2.resize(np.copy(img_LQ2), (112, 112),
        #                             interpolation=cv2.INTER_LINEAR)
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        if HQ2_path:
            img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        
        input_query_img1 = img_LQ1
        target_img1 = img_HQ1
        input_query_img2 = img_LQ2
        if HQ2_path:
            target_img2 = img_HQ2
        else:
            target_img2 = 'None'
        
        batch = {'input_query_img1': input_query_img1, 'target_img1': target_img1,
                 'input_query_img2': input_query_img2, 'target_img2': target_img2,
                 'input_query_img2_path': LQ2_path}
        
        deg_type = self.dataset_paths[idx]
        
        return batch, deg_type