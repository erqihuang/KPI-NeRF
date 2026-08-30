import os

import torch
import numpy as np
from tqdm import tqdm
import imageio
import cv2

import matplotlib.pylab as plt

from dataloader.with_colmap import resize_imgs
from utils.basis_func import DCT_bf


def load_spec_imgs(image_dir, num_img_to_load, start, end, skip, load_sorted, load_img, spec_CH, bf_num, ls_factor, ls_file, cam_sen_file):
    img_names = np.array(os.listdir(image_dir))  # all image names
    
    # down sample frames in temporal domain
    if end == -1:
        img_names = img_names[start::skip]
    else:
        img_names = img_names[start:end:skip]

    if not load_sorted:
        np.random.shuffle(img_names)

    # load images after down sampled
    if num_img_to_load > len(img_names):
        print('Asked for {0:6d} images but only {1:6d} available. Exit.'.format(num_img_to_load, len(img_names)))
        exit()
    elif num_img_to_load == -1:
        print('Loading all available {0:6d} images'.format(len(img_names)))
    else:
        print('Loading {0:6d} images out of {1:6d} images.'.format(num_img_to_load, len(img_names)))
        img_names = img_names[:num_img_to_load]

    img_paths = [os.path.join(image_dir, n) for n in img_names]
    N_imgs = len(img_paths)

    img_list = []
    if load_img == True:
        for idx, p in enumerate(tqdm(img_paths)):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED) # .exr (H, W, 3) np.float32 [0, 1] cv2.IMREAD_UNCHANGED very important
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # if minus:
            #     img -= 0.5
            img_list.append(img)

            ## visulization for testing
            # img_uint8 = (img * 255 *10).astype(np.uint8)
            # write_image(f"img_v{idx+1}.png", img_uint8, bin=50)
            # plt.figure()
            # plt.imshow(img)
            
        img_list = np.stack(img_list)  # (N, H, W, 3)
        img_list = torch.from_numpy(img_list).float()
        _, H, W, _ = img_list.shape

        # light source
        ls_path = os.path.join(os.path.dirname(image_dir), f'../environment/light_source/{ls_file}.npy')
        ls_list = np.load( ls_path ) # ( C, 1 )
        
        # camera sensitivity
        cam_sens_path = os.path.join(os.path.dirname(image_dir), f'../environment/camera_sensitivity/{cam_sen_file}.npy')
        cam_sens_list = np.load( cam_sens_path ) # (V, C, 3)

        channel_rgb = cam_sens_list.shape[2]

        wl = np.linspace(420, 700, 29)  # 29 channels from 420nm to 700nm
        new_wl = np.linspace(420, 700, spec_CH)  # 29 channels from 420nm to 700nm

        cam_sens_list_p = np.zeros( ( cam_sens_list.shape[0], spec_CH, channel_rgb ), dtype=np.float32 ) # (V, C, 3)
        for i in range( cam_sens_list.shape[0] ):  # for each view
            for j in range( channel_rgb ):
                    cam_sens_list_p[i, :, j] = np.interp( new_wl, wl, cam_sens_list[i, :, j] ) * np.interp( new_wl, wl, ls_list )

        cam_sens_list_p = torch.from_numpy( cam_sens_list_p * ls_factor ) # (V, C, 3)


        # DCT version -- dimension reduction of  rgb_sens_list_p
        bf = DCT_bf(bf_num, spec_CH)
        bf_inv = bf.transpose(1, 0)

        bf = torch.from_numpy( bf )
        bf_inv = torch.from_numpy( bf_inv ) 

    else:
        img = imageio.imread(img_paths[0])   # load one image to get H, W
        H, W = img.shape[0], img.shape[1]

    results = {
        'imgs': img_list,  # (N, H, W) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,   # (N, )   
        'H': H,
        'W': W,
        'rgb_sens': cam_sens_list_p,   # (V, N_BF, 3)
        'bf': bf,   # ( bf_num, spec_CH ) 
        'bf_inv': bf_inv,   # ( spec_CH, bf_num )
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
    def __init__(self, base_dir, scene_name, res_ratio, num_img_to_load, start, end, skip, load_sorted, load_img, channels, bf_num, ls_factor, ls_file, cam_sen_file):
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
        self.num_img_to_load = num_img_to_load
        self.start = start
        self.end = end
        self.skip = skip
        self.load_sorted = load_sorted
        self.load_img = load_img
        self.channels = channels
        self.bf_num = bf_num 
        self.ls_factor = ls_factor
        self.ls_file = ls_file # xenon_c29.npy
        self.cam_sen_file = cam_sen_file # xenon_c29.npy


        self.imgs_dir = os.path.join(self.base_dir, self.scene_name)

        image_data = load_spec_imgs(self.imgs_dir, self.num_img_to_load, self.start, self.end, self.skip, 
                                    self.load_sorted, self.load_img, self.channels, self.bf_num, self.ls_factor, ls_file, cam_sen_file)
        
        self.imgs = image_data['imgs']  # (N, H, W, 9) torch.float32
        self.img_names = image_data['img_names']  # (N, )
        self.N_imgs = image_data['N_imgs']
        self.ori_H = image_data['H']
        self.ori_W = image_data['W']
        self.rgb_sens = image_data['rgb_sens']
        self.bf_inv = image_data['bf_inv']  # ( bf_nums, spec_CH )

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
            self.imgs = resize_imgs(self.imgs, self.H, self.W)  # (N, H, W, 1) torch.float32


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
    scene_name = '20250526/DSC_0007_exr'
    resize_ratio = 3
    num_img_to_load = -1
    start = 0
    end = -1
    skip = 1
    load_sorted = True
    load_img = True

    scene = SpecDataLoaderAnyFolder(base_dir=base_dir,
                                    scene_name=scene_name,
                                    res_ratio=resize_ratio,
                                    num_img_to_load=num_img_to_load,
                                    start=start,
                                    end=end,
                                    skip=skip,
                                    load_sorted=load_sorted,
                                    load_img=load_img,
                                    channels=29,
                                    minus=False)
