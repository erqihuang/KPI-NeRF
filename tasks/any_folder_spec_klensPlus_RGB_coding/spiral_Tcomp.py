import sys
import os
import argparse
from pathlib import Path
from matplotlib import pyplot as plt

import torch
import numpy as np
from tqdm import tqdm
import imageio
import cv2

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.any_folder_spec_klensPlus_RGB_Tcomp import SpecDataLoaderAnyFolder
from utils.training_utils import set_randomness, load_ckpt_to_net, mse2psnr
from utils.pose_utils import create_spiral_poses
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from utils.lie_group_helper import convert3x4_4x4
from models.mlf_nerf_models import MLF_Nerf
from tasks.any_folder_spec_klensPlus_RGB_coding.train_Tcomp import model_render_image
from models.intrinsics import LearnFocal
from models.poses import LearnPose, synthetic_poses
from models.basis import BasisFunction

from utils.visualization import generate_filter, generate_bayer_filter

import torch.nn.functional as F

from utils.evaluate_metrics import evaluate_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, action='store_true')
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data')
    parser.add_argument('--scene_name', type=str, default='any_folder_demo/desk')

    parser.add_argument('--learn_focal', default=False, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])

    parser.add_argument('--learn_R', default=False, type=bool)
    parser.add_argument('--learn_t', default=False, type=bool)

    parser.add_argument('--resize_ratio', type=int, default=4, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=128, help='network hidden unit dimensions')
    parser.add_argument('--spec_chnls', type=int, default=9, help='number of spectral channles')
    parser.add_argument('--num_sample', type=int, default=128, help='number samples along a ray')

    parser.add_argument('--bf_num', type=int, default=3, help='number of freqs for basis function')
    
    parser.add_argument('--pos_enc_levels', type=int, default=10, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train')
    parser.add_argument('--train_load_sorted', type=bool, default=True)
    parser.add_argument('--train_start', type=int, default=0, help='inclusive')
    parser.add_argument('--train_end', type=int, default=-1, help='exclusive, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--spiral_mag_percent', type=float, default=50, help='for np.percentile')
    parser.add_argument('--spiral_axis_scale', type=float, default=[1.0, 1.0, 1.0], nargs=3,
                        help='applied on top of percentile, useful in zoom in motion')
    parser.add_argument('--N_img_per_circle', type=int, default=60)
    parser.add_argument('--N_circle_traj', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=2.2)
    # parser.add_argument('--val_scale', type=float, default=1)

    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--spiral', action="store_true", help="spiral or not")
    parser.add_argument('--stare', action="store_true", help="stare or not")
    parser.add_argument('--act', type=str, default='linear', help="activation function")
    parser.add_argument('--init_basis', type=str, default='DCT', help="basis function")
    parser.add_argument('--ls_factor', type=float, default=1, help="light source power factor")
    parser.add_argument('--ls_file', type=str, default='xenon_c29', help="xenon_c29 / sun_c29")
    parser.add_argument('--cam_sen_file', type=str, default='camera_sensitivity_c29', help="camera_sensitivity_c29")


    return parser.parse_args()


def test_one_epoch(H, W, focal_net, c2ws, basis_net,
                   near, far, model, my_devices, rgb_sens, args):
    model.eval()
    focal_net.eval()
    basis_net.eval()

    fxfy = focal_net(0)
    bf = basis_net(0)
    ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(near, far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img = c2ws.shape[0]

    rendered_rgb_list = []
    rendered_spec_list = []
    rendered_depth_list = []

    for i in tqdm( range(N_img) ):
        c2w = c2ws[i].to(my_devices)  # (4, 4)
        
        if args.act == 'linear':
            spec_act_fn = lambda x : x  # x=x
        elif args.act == 'sigmoid':
            spec_act_fn = torch.sigmoid # sigmoid
        elif args.act == 'exponential':
            spec_act_fn = torch.exp # exponential
        elif args.act == 'relu':    
            spec_act_fn = torch.relu    # relu
        elif args.act == 'softplus':
            spec_act_fn = torch.nn.Softplus()    # softplus

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_rgb = []
        rendered_spec = []
        rendered_depth = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, near, far, H, W, fxfy,
                                               model, False, 0.0, args, spec_act_fn, rgb_sens, bf)
            spec_rendered_rows = render_result['spec']  # (num_rows_eval_img, W)
            rgb_rendered_rows = render_result['rgb']   # (V, num_rows_eval_img, W, 3)
            depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

            rendered_spec.append(spec_rendered_rows)
            rendered_rgb.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)

        # combine rows to an image
        rendered_spec = torch.cat(rendered_spec, dim=0)  # (H, W, C)
        rendered_rgb = torch.cat(rendered_rgb, dim=1)  # (V, H, W, 3)
        rendered_depth = torch.cat(rendered_depth, dim=0)  # (H, W)

        # for vis
        rendered_spec_list.append(rendered_spec)
        rendered_rgb_list.append(rendered_rgb)
        rendered_depth_list.append(rendered_depth)

    rendered_rgb_list = torch.stack(rendered_rgb_list)  # (N, V, H, W, 3)
    rendered_spec_list = torch.stack(rendered_spec_list)  # (N, H, W, C)
    rendered_depth_list = torch.stack(rendered_depth_list)  # (N, H, W, 3)

    result = {
        'rgb': rendered_rgb_list,
        'spec': rendered_spec_list,
        'depths': rendered_depth_list,
    }
    return result
 

def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    test_dir = Path( os.path.join( args.ckpt_dir, 'render_spiral' + '_gamma_' + str(args.gamma) ) )
    rgb_out_dir = Path(os.path.join(test_dir, 'rgb_out'))
    depth_out_dir = Path(os.path.join(test_dir, 'depth_out'))
    video_out_dir = Path(os.path.join(test_dir, 'video_out'))
    mlf_depth_out_dir = Path(os.path.join(test_dir, 'mlf_depth_out'))
    spec_out_dir = Path(os.path.join(test_dir, 'spec_out'))
    test_dir.mkdir(parents=True, exist_ok=True)
    rgb_out_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir.mkdir(parents=True, exist_ok=True)
    mlf_depth_out_dir.mkdir(parents=True, exist_ok=True)
    spec_out_dir.mkdir(parents=True, exist_ok=True)

    '''Load scene meta'''
    scene_train = SpecDataLoaderAnyFolder(base_dir=args.base_dir,
                                          scene_name=args.scene_name,
                                          res_ratio=args.resize_ratio,
                                          num_img_to_load=args.train_img_num,
                                          start=args.train_start,
                                          end=args.train_end,
                                          skip=args.train_skip,
                                          load_sorted=args.train_load_sorted,
                                          load_img=True,
                                          channels=args.spec_chnls,
                                          bf_num=args.bf_num,
                                          ls_factor=args.ls_factor,
                                          ls_file=args.ls_file,
                                          cam_sen_file=args.cam_sen_file,
                                          )

    print('H: {0:4d}, W: {1:4d}.'.format(scene_train.H, scene_train.W))
    print('near: {0:.1f}, far: {1:.1f}.'.format(scene_train.near, scene_train.far))

    '''Model Loading'''
    pos_enc_in_dims = (2 * args.pos_enc_levels + int(args.pos_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    if args.use_dir_enc:
        dir_enc_in_dims = (2 * args.dir_enc_levels + int(args.dir_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    else:
        dir_enc_in_dims = 0

    model = MLF_Nerf(pos_enc_in_dims, dir_enc_in_dims, args.hidden_dims, args.bf_num )
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)
    model = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_nerf.pth'), model, map_location=my_devices)

    focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)
    focal_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_focal.pth'), focal_net, map_location=my_devices)

    pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)
    if args.multi_gpu:
        pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
    else:
        pose_param_net = pose_param_net.to(device=my_devices)
    pose_param_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_pose.pth'), pose_param_net, map_location=my_devices)

    basis_net = BasisFunction(args.bf_num, args.spec_chnls, req_grad=True, init_basis=args.init_basis)
    if args.multi_gpu:
        basis_net = torch.nn.DataParallel(basis_net).to(device=my_devices)
    else:
        basis_net = basis_net.to(device=my_devices)
    basis_net = load_ckpt_to_net(os.path.join(args.ckpt_dir, 'latest_basis.pth'), basis_net, map_location=my_devices)


    # learned_poses = torch.stack([synthetic_poses(i).to(my_devices)for i in range(scene_train.N_imgs)])
    learned_poses = torch.stack( [pose_param_net(i) for i in range(scene_train.N_imgs)] )

    '''Generate camera traj'''
    # This spiral camera traj code is modified from https://github.com/kwea123/nerf_pl.
    # hardcoded, this is numerically close to the formula given in the original repo. Mathematically if near=1
    # and far=infinity, then this number will converge to 4. Borrowed from https://github.com/kwea123/nerf_pl
    N_novel_imgs = args.N_img_per_circle * args.N_circle_traj
    focus_depth = 4
    radii = np.percentile( np.abs(learned_poses.cpu().numpy()[:, :3, 3]), args.spiral_mag_percent, axis=0 )  # (3,)
    radii *= np.array(args.spiral_axis_scale)
    c2ws = create_spiral_poses(radii, focus_depth, n_poses=N_novel_imgs, n_circle=args.N_circle_traj,)
    c2ws = torch.from_numpy(c2ws).float()  # (N, 3, 4)
    c2ws = convert3x4_4x4(c2ws)  # (N, 4, 4)
    rgb_sens = scene_train.rgb_sens.to(my_devices)  # (V, C, 3)

    ## evaluation
    gamma = args.gamma
    GT = scene_train.imgs.to(my_devices)  # (N, H, W, 3)

    # camera parameters
    fxfy = focal_net(0)
    print( f'learned fx: {fxfy[0].item():.2f}, fy: {fxfy[1].item():.2f}' ) 
    # print( 'learned fx: {1:.2f}, fy: {2:.2f}'.format( fxfy[0].item(), fxfy[1].item() ) )

    # basis function
    bf = basis_net(0)
    bf_inv = bf.transpose(1, 0)

    # pseudo multispectral rgb image
    spec = torch.zeros( scene_train.H, scene_train.W, args.spec_chnls, device=my_devices)  # [H, W, C]
    N_Filter = 9
    pseu_ms_rgb = torch.zeros( scene_train.H * rgb_sens.shape[0], scene_train.W * N_Filter, 3, device=my_devices )    # [H * V, W * N_Filter, 3]
    pseu_filter = torch.from_numpy( generate_filter() ).to( spec.device )   # [N_Filter, C, 3]

    # pseudo bayer filter
    pseu_bayer_filter = torch.from_numpy( generate_bayer_filter() ).to( spec.device )   # [N_Filter, C, 3]

    '''Spiral'''
    if args.spiral:
        spec_spiral = torch.zeros( N_novel_imgs, scene_train.H, scene_train.W, args.spec_chnls, device=my_devices)  # [N_view, H, W, C]
        results = test_one_epoch( scene_train.H, scene_train.W, focal_net, c2ws, basis_net,
                                 scene_train.near, scene_train.far, model, my_devices, rgb_sens, args )
        
        spec_spiral = torch.einsum( "cb,nhwb->nhwc", bf_inv, results['spec'] ) # [C, 2xBF+1] x [N, H, W, 2xBF+1] -> [N, H, W, C]
        spec_spiral[ spec_spiral < 0 ], spec_spiral[ spec_spiral > 1 ] = 0, 1
        # for sc in range( bf_inv.shape[0] ):
        #     spec_spiral[:, :, :, sc] = torch.sum( bf_inv[sc] * results['spec'], dim=3 )   # [C, 2xBF+1] x [N, H, W, 2xBF+1] -> [N, H, W, C]
        # spec_spiral = torch.clamp(spec_spiral, 0, 1)
        
        for s in range( rgb_sens.shape[0] ):
            rgb = results['rgb'][:, s, :, :, :]  # [N, V, H, W, 3]
            
            ## Multispectral Ligh-field Pseudo RGB image
            _pseu_rgb = torch.einsum('vhwc, nck->vnhwk', spec_spiral, pseu_filter)[:, s, ...] # [N_view, H, W, C] x [N_Filter, C, 3] ->  [N_view, N_Filter, H, W, 3]
        
            '''Write to folder'''           
            rgb = ( np.clip( rgb.cpu().numpy() ** ( 1 / gamma ), 0, 1 ) * 255 ).astype(np.uint8)
            pseu_rgb_u8 = ( _pseu_rgb.cpu().numpy() ** ( 1 / gamma ) * 255 ).astype(np.uint8)
            pseu_rgb_u8 = ( np.clip( _pseu_rgb.cpu().numpy() ** ( 1 / gamma ), 0, 1 ) * 255 ).astype(np.uint8)

            # for i in range(c2ws.shape[0]):
            #     imageio.imwrite(os.path.join(rgb_out_dir, str(i).zfill(4) + f'{s+1}.png'), rgb[i])
            #     imageio.imwrite(os.path.join(depth_out_dir, str(i).zfill(4) + f'{s+1}.png'), depths[i])

            # imageio.mimwrite(os.path.join(video_out_dir, f'img{s+1}.mp4'), rgb, fps=30, quality=9)
            # imageio.mimwrite(os.path.join(video_out_dir, f'depth{s+1}.mp4'), depths, fps=30, quality=9)

            imageio.mimwrite(os.path.join(video_out_dir, f'img{s+1}.gif'), rgb, fps=30, loop=0)
            imageio.mimwrite(os.path.join(video_out_dir, f'pseu_ml{s+1}.gif'), pseu_rgb_u8, fps=30, loop=0)
        

        ## Ligh-field Pseudo RGB bayer image
        pseu_bayer = ( spec_spiral @ pseu_bayer_filter ) / pseu_bayer_filter.shape[0]   # [N_view, H, W, C] x [C, 3] ->  [N_view, H, W, 3]
        pseu_bayer_8u = ( np.clip( pseu_bayer.cpu().numpy() ** ( 1 / gamma ), 0, 1 ) * 255 ).astype(np.uint8)

        depths = results['depths']
        depths = ( depths.cpu().numpy() * 255 ).astype( np.uint8 )  # far is 1.0 in NDC

        imageio.mimwrite(os.path.join(video_out_dir, f'pseu_bayer.gif'), pseu_bayer_8u, fps=30, loop=0)
        imageio.mimwrite(os.path.join(video_out_dir, f'depth.gif'), depths, fps=30, loop=0)

    '''Stare'''
    if args.stare:

        avg_psnr = []
        avg_ssim = []
        avg_sam = []
        avg_rmse = []

        for s in range( rgb_sens.shape[0] ):  # [9 views]
            '''Multispectral light-field image'''
            poses = torch.index_select( learned_poses, dim=0, index=torch.tensor([s]).to(device=my_devices) )
            print( f'view {s + 1}: , learned pose: { str( poses[0] ) }' )
            result = test_one_epoch( scene_train.H, scene_train.W, focal_net, poses, basis_net,
                                    scene_train.near, scene_train.far, model, my_devices, rgb_sens, args )
             
            spec = torch.einsum( "cb, hwb->hwc", bf_inv, result['spec'][0] ) # [C, 2xBF+1] x [H, W, 2xBF+1] -> [H, W, C]
            spec[ spec < 0 ], spec[ spec > 1 ] = 0, 1
            # spec = torch.matmul( bf_inv, result['spec'][0] )    # [C, 2xBF+1] x [H, W, 2xBF+1] -> [H, W, C]
            # for sc in range( bf_inv.shape[0] ):
            #     spec[:, :, sc] = torch.sum( bf_inv[sc, None, None] * result['spec'][0], dim=2 )   # [C, 2xBF+1] x [1, H, W, 2xBF+1] -> [H, W, C]
            
            ## Multispectral image
            depth = result['depths'][0] # [N, H, W]

            ## save spectral image
            subdir = os.path.join( spec_out_dir, f'v{ s + 1 }' )
            os.makedirs( subdir, exist_ok=True )
            spec_uint16 = ( spec.cpu().numpy() * 65535 ).astype( np.uint16 )   # [0, 1] to [0, 65535]
            for b in range( args.spec_chnls ):
                file_name = f'spec_v{ str(s+1).zfill(2)}_c{str(b+1).zfill(2) }.png'
                imageio.imwrite( os.path.join( subdir, file_name ), spec_uint16[:, :, b]  )

            ## Ligh-field Pseudo RGB image - spectrum * bayer 
            pseu_bayer = ( spec @ pseu_bayer_filter ) / pseu_bayer_filter.shape[0]   # [H, W, C] x [C, 3] -> [H, W, 3]
            pseu_bayer_8u = ( np.clip( pseu_bayer.cpu().numpy() ** ( 1 / gamma ), 0, 1 ) * 255 ).astype(np.uint8)
            imageio.imwrite( os.path.join( rgb_out_dir, f'pseu_rgb_v{str(s+1).zfill(2)}.png' ), pseu_bayer_8u )

            ## Multispectral Ligh-field Pseudo RGB image
            # spectrum * filters ( color + bayer )
            _pseu_ms_rgb = torch.einsum('hwc,nck->nhwk', spec, pseu_filter) # [H, W, C] x [N_Filter, C, 3] -> [N_Filter, H, W, 3]
            pseu_ms_rgb[ s * scene_train.H : (s + 1) * scene_train.H, :N_Filter * scene_train.W, : ] = \
                torch.cat( [ _pseu_ms_rgb[i] for i in range(N_Filter) ], dim=1 )

            ## Ligh-field RGB image
            for c in range( rgb_sens.shape[0] ):    # [9 color filters]
                rgb = result['rgb'][0, c, :, :, :]   # [N, V, H, W, 3]
                ## Write to folder
                # cv2.imwrite(os.path.join(mlf_depth_out_dir, f'rgb_v{str(s+1).zfill(2)}_c{str(c+1).zfill(2)}.exr'), rgb.cpu().numpy())
                rgb_8u = ( ( rgb.cpu().numpy() ** ( 1 / gamma ) ) * 255 ).astype(np.uint8)
                imageio.imwrite(os.path.join(mlf_depth_out_dir, f'rgb_v{str(s+1).zfill(2)}_c{str(c+1).zfill(2)}.png'), rgb_8u)

            ## Evaluation
            _GT = GT[s, ...]    # (H, W, 3)
            _rgb = result['rgb'][0, s, ...] # (H, W, 3)
            L2_loss = F.mse_loss( _rgb, _GT )
            train_psnr = mse2psnr( L2_loss.cpu().numpy() )
            # print( f"MSE of view { s + 1 }: { L2_loss }" )
            print( f"PSNR of view { s + 1 }: { train_psnr }" )
            avg_psnr.append( train_psnr )

            spec_psnr, spec_ssim, spec_sam, spec_rmse = evaluate_metrics( _GT, _rgb )
            avg_psnr.append(spec_psnr)
            avg_ssim.append(spec_ssim)
            avg_sam.append(spec_sam)
            avg_rmse.append(spec_rmse)

            depth_8u = ( depth.cpu().numpy() * 255 ).astype(np.uint8)  # far is 1.0 in NDC
            gt_gamma_8u = ( ( _GT.cpu().numpy() ** ( 1 / gamma ) ) * 255 ).astype( np.uint8 )

            imageio.imwrite(os.path.join(mlf_depth_out_dir, f'depth_v{str(s+1).zfill(2)}.png'), depth_8u)
            # cv2.imwrite(os.path.join(mlf_depth_out_dir, f'gt_v{str(s+1).zfill(2)}.exr'), _GT.cpu().numpy())
            imageio.imwrite(os.path.join(mlf_depth_out_dir, f'gt_v{str(s+1).zfill(2)}.png'), gt_gamma_8u)

        print( f"Tcomp mean PSNR: { np.mean(avg_psnr) }" )
        print( f"Tcomp mean SSIM: { np.mean(avg_ssim) }" )
        print( f"Tcomp mean SAM: { np.mean(avg_sam) }" )
        print( f"Tcomp mean RMSE: { np.mean(avg_rmse) }" )

        # save metric
        filename = os.path.join(test_dir, 'metrics.txt')
        with open( filename , "w") as f:
            f.write(f"file info : {filename}\n")
            f.write(f"mean PSNR : {np.mean(avg_psnr):.4f} dB\n")
            f.write(f"mean SSIM : {np.mean(avg_ssim):.4f} \n")
            f.write(f"mean SAM  : {np.mean(avg_sam):.4f} rad \n")
            f.write(f"mean RMSE : {np.mean(avg_rmse):.4f}\n")

        # save pseudo rgb image array
        pseu_ms_rgb_np = pseu_ms_rgb.cpu().numpy()
        pseu_ms_rgb_np =  np.clip( pseu_ms_rgb_np ** (1 / gamma), 0, 1 )
        pseu_ms_rgb_8u = ( pseu_ms_rgb_np * 255 ).astype( np.uint8 )
        imageio.imwrite( os.path.join( rgb_out_dir, f'pseu_rgb.png' ), pseu_ms_rgb_8u )

    

if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    with torch.no_grad():
        main(args)
