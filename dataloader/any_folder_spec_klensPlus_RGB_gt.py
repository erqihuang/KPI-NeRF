import os

import torch
import numpy as np
from tqdm import tqdm
import imageio
import cv2

import matplotlib.pylab as plt

from dataloader.with_colmap import resize_imgs
from utils.basis_func import DCT_bf

import glob


def load_spec_imgs(image_dir, load_sorted, load_img, spec_CH, bf_num, ls_factor, ls_name, cam_sen_name):
    img_paths = glob.glob( os.path.join( image_dir, '*.exr' ) ) # all .exr image names
    gt_names = glob.glob( os.path.join( image_dir, '*.npy' ) ) # gt

    if not load_sorted:
        np.random.shuffle( img_paths )

    N_imgs = len(img_paths)
    img_list = []
    if load_img == True:
        for p in tqdm( img_paths ) :
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED) # .exr (H, W, 3) np.float32 [0, 1] cv2.IMREAD_UNCHANGED very important
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_list.append( img )

            ## visulization for testing
            # img_uint8 = (img * 255 *10).astype(np.uint8)
            # write_image(f"img_v{idx+1}.png", img_uint8)
            # plt.figure()
            # plt.imshow(img)
            
        img_list = np.stack( img_list )  # (N, H, W, 3)
        img_list = torch.from_numpy( img_list ).float()
        _, H, W, rgb_C = img_list.shape

        # light source
        ls_path = os.path.join(os.path.dirname(image_dir), f'../environment/light_source/{ls_name}.npy')
        ls_list = np.load( ls_path ) # ( C, 1 )

        # camera sensitivity
        cam_sens_path = os.path.join(os.path.dirname(image_dir), f'../environment/camera_sensitivity/{cam_sen_name}.npy')
        cam_sens_list = np.load( cam_sens_path ) # (N, C, 3)

        wl = np.linspace(420, 700, 29)  # 29 channels from 420nm to 700nm
        new_wl = np.linspace(420, 700, spec_CH)  # 29 channels from 420nm to 700nm

        cam_sens_list_p = np.zeros( ( cam_sens_list.shape[0], spec_CH, rgb_C ), dtype=np.float32 ) # (N, C, 3)
        for n in range( N_imgs ):  # for each view
            for i in range( rgb_C ):
                    cam_sens_list_p[n, :, i] = np.interp( new_wl, wl, cam_sens_list[n, :, i] ) * np.interp( new_wl, wl, ls_list )
        cam_sens_list_p = torch.from_numpy( cam_sens_list_p * ls_factor ) # (N, C, 3)

        # DCT version
        bf = DCT_bf(bf_num, spec_CH)
        bf_inv = bf.T

        bf = torch.from_numpy( bf )
        bf_inv = torch.from_numpy( bf_inv ) 

        # gt
        gt = torch.from_numpy( np.load( gt_names[0] ) ) # [N, H, W, C]
        if gt.ndim == 3:
            gt = gt.unsqueeze(0).repeat( N_imgs, 1, 1, 1 )

        # least square method
        _sens_nrngnb = cam_sens_list_p.permute( [2, 0, 1] ).reshape( N_imgs * 3, -1)  # ( N, C, 3 ) -> ( 3 x N, C )   [r..., g..., b...]
        PHI = _sens_nrngnb @ bf.T # ( 3 x N, C ) * ( C, N_BF ) -> ( 3 x N, N_BF )
        Minv = torch.linalg.inv( PHI.T @ PHI ) @ PHI.T  # ( N_BF, 3*N ) * ( 3*N, N_BF ) * ( N_BF, 3*N ) -> ( N_BF, 3 x N )

        _img = img_list.permute([1, 2, 3, 0]).reshape( H, W, 3 * N_imgs )   # ( N, H, W, 3 ) -> ( H, W, 3 x N )
        coeff = _img @ Minv.T   # ( H, W, 3 x N ) x ( 3 x N, N_BF ) -> ( H, W, N_BF )

        # lsm precision test
        _sens_nrgb = cam_sens_list_p.permute( [0, 2, 1] ).reshape( N_imgs * 3, -1 )  # ( N, C, 3 ) -> ( N x 3, C )   [rgb, rgb, rgb...]
        _rgb = coeff @ bf @ _sens_nrgb.T   # (H, W, N_BF ) x ( N_BF, C ) x ( C, N x 3 ) -> ( H, W, N x 3 ) 
        rgb = _rgb.reshape( H, W, N_imgs, 3 ).permute( [2, 0, 1, 3] )    # (N, H, W, 3)
        mse = torch.mean( ( img_list - rgb ) ** 2 )
        pnsr = -10 * torch.log10(mse)
        print( pnsr )

    else:
        img = imageio.imread(img_paths[0])   # load one image to get H, W
        H, W = img.shape[0], img.shape[1]

    results = {
        'imgs': img_list,  # (N, H, W) torch.float32
        'N_imgs': N_imgs,   # (N, )   
        'H': H,
        'W': W,
        'rgb_sens': cam_sens_list_p,   # (N, N_BF, 3)
        'bf': bf,   # ( N_BF, C ) 
        'bf_inv': bf_inv,   # ( C, N_BF )
        'gt': gt,   # [N, H, W, C]
        'Minv': Minv,   # ( N_BF, 3 x N )
    }

    return results


class SpecDataLoaderAnyFolder:
    """
    Most useful fields:
        self.c2ws:          (N_imgs, 4, 4)      torch.float32
        self.imgs           (N_imgs, H, W, 4)   torch.float32
        self.ray_dir_cam    (H, W, 3)           torch.float32
        self.H              scalar
        self.W              scalar
        self.N_imgs         scalar
    """
    def __init__(self, base_dir, scene_name, res_ratio, load_sorted, load_img, channels, bf_num, ls_factor, ls_name, cam_sen_name):
        """
        :param base_dir:
        :param scene_name:
        :param res_ratio:       int [1, 2, 4] etc to resize images to a lower resolution.
        :param start/end/skip:  control frame loading in temporal domain.
        :param load_sorted:     True/False.
        :param load_img:        True/False. If set to false: only count number of images, get H and W,
                                but do not load imgs. Useful when vis poses or debug etc.
        """
        self.base_dir = base_dir
        self.scene_name = scene_name
        self.res_ratio = res_ratio
        self.load_sorted = load_sorted
        self.load_img = load_img
        self.channels = channels
        self.bf_num = bf_num 
        self.ls_factor = ls_factor
        self.ls_name = ls_name # xenon_c29.npy
        self.cam_sen_name = cam_sen_name # xenon_c29.npy


        self.imgs_dir = os.path.join(self.base_dir, self.scene_name)

        image_data = load_spec_imgs(self.imgs_dir, self.load_sorted, self.load_img, 
                                    self.channels, self.bf_num, self.ls_factor, ls_name, cam_sen_name)
        
        self.imgs = image_data['imgs']  # (N, H, W, 9) torch.float32
        self.N_imgs = image_data['N_imgs']
        self.ori_H = image_data['H']
        self.ori_W = image_data['W']
        self.rgb_sens = image_data['rgb_sens']
        self.bf = image_data['bf']  # ( N_BF, C ) 
        self.bf_inv = image_data['bf_inv']  # ( C, N_BF )
        self.gt = image_data['gt']  # ( N, H, W, C )
        self.Minv = image_data['Minv']  # ( N_BF, 3 x N )
        self.g_coeff = torch.tensor(0)    #  (H, W, N_BF)

        # always use ndc
        self.near = 0.0
        self.far = 1.0

        if self.res_ratio > 1:
            self.H = self.ori_H // self.res_ratio
            self.W = self.ori_W // self.res_ratio
        else:
            self.H = self.ori_H
            self.W = self.ori_W

        if self.load_img:
            self.imgs = resize_imgs(self.imgs, self.H, self.W)  # (N, H, W, 3) torch.float32
            self.gt = resize_imgs(self.gt, self.H, self.W)  # (N, H, W, C) torch.float32


def write_image(file_name, image_array, bin=50):
    import cv2
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_check = os.path.join(current_directory, 'img')
    if not os.path.isdir(path_check): os.makedirs(path_check)
    imageio.imwrite( os.path.join( path_check, file_name ), image_array )

    from matplotlib import pyplot as plt
    plt.figure()
    plt.hist( image_array.flatten(), bins=bin, color='blue', alpha=0.7 )
    plt.xlabel( 'Brightness' )
    plt.ylabel( 'Frequency' )
    plt.title( 'Histogram of Brightness Values' )
    path_hist = os.path.join( current_directory, 'hist' )
    if not os.path.isdir( path_hist ): os.makedirs( path_hist )
    plt.savefig( os.path.join( path_hist, file_name ), dpi=300, bbox_inches='tight', pad_inches=0 )


if __name__ == '__main__':
    base_dir = '/home/registration/MLF-Nerf/data/'
    scene_name = '20250909/exr_lego_GT_para_0_rad_0.1'
    resize_ratio = 3
    num_img_to_load = -1
    load_sorted = True
    load_img = True
    channels = 29
    bf_num = 3
    ls_factor = 1
    ls_name = 'xenon_c29'
    cam_sen_name= 'camera_sensitivity_c29'
    gt_name = 'lego_GT'


    scene = SpecDataLoaderAnyFolder(base_dir=base_dir,
                                    scene_name=scene_name,
                                    res_ratio=resize_ratio,
                                    num_img_to_load=num_img_to_load,
                                    load_sorted=load_sorted,
                                    load_img=load_img,
                                    channels=channels,
                                    bf_num=bf_num,
                                    ls_factor=ls_factor,
                                    ls_name=ls_name,
                                    cam_sen_name=cam_sen_name,
                                    gt_name=gt_name,
                                    )
    