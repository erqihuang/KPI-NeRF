import sys
import os
import argparse
from pathlib import Path
import datetime
import shutil
import logging

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.any_folder_spec_klensPlus_RGB_gt import SpecDataLoaderAnyFolder
from utils.training_utils import set_randomness, mse2psnr, save_checkpoint
from utils.pos_enc import encode_position
from utils.volume_op_gt import volume_sampling_ndc, volume_spec_rendering_klen
from utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from utils.basis_func import log_basis
from models.mlf_nerf_models import MLF_Nerf
from models.intrinsics import LearnFocal
from models.poses import LearnPose, synthetic_poses
from models.basis import BasisFunction
from utils.laplacian import laplacian
from utils.block_sample import sample_blocks

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10000, type=int)
    parser.add_argument('--eval_interval', default=100, type=int, help='run eval every this epoch number')

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, type=eval, choices=[True, False])
    parser.add_argument('--base_dir', type=str, default='./data_dir/nerfmm_release_data')
    parser.add_argument('--scene_name', type=str, default='any_folder_demo/desk')

    parser.add_argument('--nerf_lr', default=1e-3, type=float)
    parser.add_argument('--nerf_milestones', default=list(range(0, 10000, 10)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--nerf_lr_gamma', type=float, default=0.9954, help="learning rate milestones gamma")

    parser.add_argument('--learn_focal', default=True, type=bool)
    parser.add_argument('--init_focal', default=None, type=float)

    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=False, type=eval, choices=[True, False])
    parser.add_argument('--focal_lr', default=1e-3, type=float)
    parser.add_argument('--focal_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+', help='learning rate schedule milestones')
    parser.add_argument('--focal_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--learn_R', default=True, type=eval, choices=[True, False])
    parser.add_argument('--learn_t', default=True, type=eval, choices=[True, False])
    parser.add_argument('--pose_lr', default=1e-3, type=float)
    parser.add_argument('--pose_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+', help='learning rate schedule milestones')
    parser.add_argument('--pose_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--basis_lr', default=1e-3, type=float)
    parser.add_argument('--basis_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+', help='learning rate schedule milestones')
    parser.add_argument('--basis_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")
    parser.add_argument('--ls_factor', type=float, default=1, help="light source power factor")
    parser.add_argument('--learn_basis', default=True, type=eval, choices=[True, False])
    parser.add_argument('--init_basis', type=str, default='DCT', help="basis function")
    parser.add_argument('--ls_name', type=str, default='xenon_c29', help="xenon_c29 / sun_c29")
    parser.add_argument('--cam_sen_name', type=str, default='camera_sensitivity_c29', help="camera_sensitivity_c29")

    parser.add_argument('--resize_ratio', type=int, default=1, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=10, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=128, help='network hidden unit dimensions')
    parser.add_argument('--spec_chnls', type=int, default=29, help='number of spectral channles')
    parser.add_argument('--train_rand_rows', type=int, default=32, help='rand sample these rows to train')
    parser.add_argument('--train_rand_cols', type=int, default=32, help='rand sample these cols to train')
    parser.add_argument('--num_sample', type=int, default=128, help='number samples along a ray')

    parser.add_argument('--bf_num', type=int, default=3, help='number of freqs for basis function')
    
    parser.add_argument('--pos_enc_levels', type=int, default=10, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for directional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--train_load_sorted', type=bool, default=True)
    parser.add_argument('--train_start', type=int, default=0, help='inclusive')
    parser.add_argument('--train_end', type=int, default=-1, help='exclusive, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')

    parser.add_argument('--alias', type=str, default='', help="experiments alias")
    parser.add_argument('--act', type=str, default='linear', help="activation function")

    
    return parser.parse_args()


def gen_detail_name(args):
    outstr = 'lr_' + str(args.nerf_lr) + \
             '_gpu' + str(args.gpu_id) + \
             '_seed_' + str(args.rand_seed) + \
             '_resize_' + str(args.resize_ratio) + \
             '_Nsam_' + str(args.num_sample) + \
             '_specChnls_' + str(args.spec_chnls) + \
             '_bfChnls_' + str(args.bf_num) + \
             '_lsfactor_' + str(args.ls_factor) + \
             '_' + str(args.alias) + \
             '_' + str(datetime.datetime.now().strftime('%y%m%d_%H%M'))
    return outstr


def model_render_image(c2w, rays_cam, t_vals, near, far, H, W, fxfy, 
                       model, perturb_t, sigma_noise_std, args, spec_act_fn, rgb_sens, bf):
    """Render an image or pixels.
    :param c2w:         (4, 4)                  pose to transform ray direction from cam to world.
    :param rays_cam:    (someH, someW, 3)       ray directions in camera coordinate, can be random selected
                                                rows and cols, or some full rows, or an ent ire image.
    :param t_vals:      (N_samples)             sample depth along a ray.
    :param fxfy:        a float or a (2, ) torch tensor for focal.
    :param perturb_t:   True/False              whether add noise to t.
    :param sigma_noise_std: a float             std dev when adding noise to raw density (sigma).
    :spec_act_fn:        sigmoid()               apply an activation fn to the raw rgb output to get actual rgb.
    :rgb_sens:       (9, 3)                  apply rgb_sens to the raw cube output to the rgb image.
    :return:            (someH, someW, 9)       volume rendered images for the input rays.
    """
    # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)
    sample_pos, _, ray_dir_world, t_vals_noisy = volume_sampling_ndc(c2w, rays_cam, t_vals, near, far,
                                                                     H, W, fxfy, perturb_t)

    # encode position: (H, W, N_sample, (2L+1)*C = 63)
    pos_enc = encode_position(sample_pos, levels=args.pos_enc_levels, inc_input=args.pos_enc_inc_in)

    # encode direction: (H, W, N_sample, (2L+1)*C = 27)
    if args.use_dir_enc:
        ray_dir_world = F.normalize(ray_dir_world, p=2, dim=2)  # (H, W, 3)
        dir_enc = encode_position(ray_dir_world, levels=args.dir_enc_levels, inc_input=args.dir_enc_inc_in)  # (H, W, 27)
        dir_enc = dir_enc.unsqueeze(2).expand(-1, -1, args.num_sample, -1)  # (H, W, N_sample, C)
    else:
        dir_enc = None

    # inference rgb and density using position and direction encoding.
    spec_density = model(pos_enc, dir_enc)  # (H, W, N_sample, 9)

    render_result = volume_spec_rendering_klen(spec_density, t_vals_noisy, sigma_noise_std, spec_act_fn, rgb_sens, bf)
    spec_rendered = render_result['spec']  # (H, W, C)
    rgb_rendered = render_result['rgb']  # (N, H, W, 3)
    depth_map = render_result['depth_map']  # (H, W)

    result = {
        'spec': spec_rendered,  # (H, W, C)
        'rgb': rgb_rendered,  # (N, H, W, 3)
        'sample_pos': sample_pos,  # (H, W, N_sample, 3)
        'depth_map': depth_map,  # (H, W)
        'rgb_density': spec_density,  # (H, W, N_sample, 4)
    }

    return result


def eval_one_epoch(eval_c2ws, scene_train, model, focal_net, pose_param_net,
                   basis_net, my_devices, args, epoch_i, writer, spec_act_fn):
    model.eval()
    focal_net.eval()
    pose_param_net.eval()
    basis_net.eval()

    fxfy = focal_net(0)
    bf = basis_net(0)

    ray_dir_cam = comp_ray_dir_cam_fxfy(scene_train.H, scene_train.W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img, H, W = eval_c2ws.shape[0], scene_train.H, scene_train.W

    rendered_img_list = []
    rendered_depth_list = []

    for i in range( N_img ):
        c2w = eval_c2ws[i].to(my_devices)  # (4, 4)
        rgb_sens = scene_train.rgb_sens.to(my_devices) # (N, 9, 3)

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(args.num_rows_eval_img, dim=0)
        rendered_img = []
        rendered_depth = []
        rendered_spec = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = model_render_image(c2w, rays_dir_rows, t_vals, scene_train.near, scene_train.far,
                                               scene_train.H, scene_train.W, fxfy,
                                               model, False, 0.0, args, spec_act_fn, rgb_sens, bf)
            mono_rendered_rows = render_result['rgb'][i]  # (num_rows_eval_img, W, 3)
            spec_rows = render_result['spec']   # (num_rows_eval_img, W, C)
            depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

            rendered_img.append(mono_rendered_rows)
            rendered_spec.append(spec_rows)
            rendered_depth.append(depth_map)

        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)
        rendered_spec = torch.cat(rendered_spec, dim=0) # (H, W, C)
        rendered_depth = torch.cat(rendered_depth, dim=0).unsqueeze(0)  # (1, H, W)

        # for vis
        rendered_img_list.append(rendered_img.cpu().numpy())
        rendered_depth_list.append(rendered_depth.cpu().numpy())

    # random display an eval image to tfboard
    rand_num = np.random.randint( low=0, high=N_img )
    disp_img = np.transpose( rendered_img_list[rand_num] ** ( 1 / 2.2 ), (2, 0, 1) )  # (3, H, W)
    disp_depth = rendered_depth_list[rand_num]  # (1, H, W)
    writer.add_image('eval_img', disp_img, global_step=epoch_i)
    writer.add_image('eval_depth', disp_depth, global_step=epoch_i)

    return


def train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, optimizer_basis, 
                    model, focal_net, pose_param_net, basis_net, my_devices, args, spec_act_fn, L):
    model.train()

    pose_param_net.train()
    # pose_param_net.eval()

    focal_net.train()

    # basis_net.train()
    basis_net.eval()
    
    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    L2_loss_epoch = []
    L2_raw_loss_epoch = []
    L2_spec_epoch = []
    L_depth_epoch = []

    rgb_sens = scene_train.rgb_sens.to(my_devices) # (N, C, 3)

    bf = scene_train.bf.to(my_devices)
    bf_inv = scene_train.bf_inv.to(my_devices)

    # lsm loss
    _sens_nrngnb = rgb_sens.permute( [2, 0, 1] ).reshape( N_img * 3, -1 )  # ( N, C, 3 ) -> ( 3 x N, C )   [r..., g..., b...]
    PHI = _sens_nrngnb @ bf.T # ( 3 x N, C ) * ( C, N_BF ) -> ( 3 x N, N_BF )
    Minv = torch.linalg.inv( PHI.T @ PHI ) @ PHI.T  # ( N_BF, 3 x N ) * ( 3 x N, N_BF ) * ( N_BF, 3 x N ) -> ( N_BF, 3 x N )

    # tonemapping
    tonemapping = lambda sg_x, x, : x / torch.sqrt( sg_x.detach() + 1e-3 )    # linear tonemapping
    # tonemapping = lambda sg_x, x, : x    # without tonemapping

    # training imgs
    for i in range(N_img):
        fxfy = focal_net(0)
        ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
        c2w = pose_param_net(i)  # (4, 4)
        # c2w = synthetic_poses(i).to(my_devices) # (4, 4)
        img = scene_train.imgs[i].to(my_devices)  # (H, W, 3)
        gt = scene_train.gt[i].to(my_devices)  # (H, W, C)
        # coeff = scene_train.coeff[i].to(my_devices)  # (H, W, N_BF)

        # sample pixel on an image and their rays for training.
        r_id = torch.randperm(H, device=my_devices)[:args.train_rand_rows]  # (N_select_rows)
        c_id = torch.randperm(W, device=my_devices)[:args.train_rand_cols]  # (N_select_cols)
        ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
        img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 1)
        gt_selected = gt[r_id][:, c_id]  # (N_select_rows, N_select_cols, C)
        # coeff_selected = coeff[r_id][:, c_id]  # (N_select_rows, N_select_cols, N_BF)

        # ray_selected_cam, img_selected = sample_blocks( ray_dir_cam, img, H, W, my_devices,
        #                                                   num_blocks=16, block_size=2 )

        # render an image using selected rays, pose, sample intervals, and the network
        render_result = model_render_image(c2w, ray_selected_cam, t_vals, scene_train.near, scene_train.far,
                                           scene_train.H, scene_train.W, fxfy,
                                           model, True, 0.0, args, spec_act_fn, rgb_sens, bf)  # (N_select_rows, N_select_cols, 1)
        rgb_rendered = render_result['rgb']  # (N, N_select_rows, N_select_cols, 3)
        comp_spec_rendered = render_result['spec']  # (H, W, N_BF)
        depth_rendered = render_result['depth_map']

        # lsm loss
        _img = rgb_rendered.clone()
        _img = _img.permute([1, 2, 3, 0]).reshape( args.train_rand_rows, args.train_rand_cols, 3 * N_img )   # ( N, H, W, 3 ) -> ( H, W, 3 x N )
        coeff_selected = _img @ Minv.T   # ( H, W, 3 x N ) x ( 3 x N, N_BF ) -> ( H, W, N_BF )
        _rgb_i = coeff_selected @ bf @ rgb_sens[i] # ( H, W, N_BF ) x ( N_BF, C ) x ( C, 3 ) -> ( H, W, 3 ) 
        L2_lsm = F.mse_loss( tonemapping( _rgb_i, _rgb_i ), tonemapping( _rgb_i, img_selected ) )
    
        # Depth loss (Deprecated)
        # depth_dx = depth_rendered[:, :-1:2] - depth_rendered[:, 1::2]
        # depth_dy = depth_rendered[:-1:2, :] - depth_rendered[1::2, :]
        # L_depth = torch.mean( ( torch.abs(depth_dx) ) + torch.mean( torch.abs(depth_dy) ) ** 2 )
        L_depth = torch.tensor(0).to( device = my_devices )
        
        # gt spec for referring
        spec = coeff_selected @ bf # (H, W, N_BF) x (N_BF, C) -> (H, W, C)
        spec = spec.clamp( 0, 1 )
        L2_lap = torch.mean( ( spec @ L ) ** 2 ) # (H, W, C) x (C, C) -> (H, W, C)
        L_depth = L2_lap

        L2_spec = F.mse_loss( spec, gt_selected ) 

        L2_raw_loss = F.mse_loss( rgb_rendered[i], img_selected )  # loss for one image
        L2_tm_loss = F.mse_loss( tonemapping(rgb_rendered[i], rgb_rendered[i]), tonemapping(rgb_rendered[i], img_selected) )  # loss for one image

        L2_loss = L2_lsm
        # L2_loss = L2_tm_loss
        # L2_loss += ( 1e-4 * L2_lap )
        # L2_loss += L2_lsm

        L2_loss.backward()
        optimizer_nerf.step()
        optimizer_focal.step()
        optimizer_pose.step()
        optimizer_basis.step()
        optimizer_nerf.zero_grad()
        optimizer_focal.zero_grad()
        optimizer_pose.zero_grad()
        optimizer_basis.zero_grad()

        L2_loss_epoch.append(L2_loss.item())
        L2_raw_loss_epoch.append(L2_raw_loss.item())
        L2_spec_epoch.append(L2_spec.item())
        L_depth_epoch.append(L_depth.item())

    L2_loss_epoch_mean = np.mean(L2_loss_epoch)  # loss for all images.
    L2_raw_loss_epoch_mean = np.mean(L2_raw_loss_epoch)  # loss for all images.
    L2_spec_epoch_mean = np.mean(L2_spec_epoch)  # loss for all images.
    L_depth_epoch_mean = np.mean(L_depth_epoch)  # loss for all images.

    mean_losses = {
        'L2': L2_loss_epoch_mean,
        'L2_raw': L2_raw_loss_epoch_mean,
        'L2_spec': L2_spec_epoch_mean,
        'L_depth': L_depth_epoch_mean,
    }
    return mean_losses


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    exp_root_dir = Path(os.path.join('./logs/any_folder_spec', args.scene_name))
    exp_root_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = Path(os.path.join(exp_root_dir, gen_detail_name(args)))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy('./models/mlf_nerf_models.py', experiment_dir)
    shutil.copy('./models/basis.py', experiment_dir)
    shutil.copy('./models/intrinsics.py', experiment_dir)
    shutil.copy('./models/poses.py', experiment_dir)
    shutil.copy('./tasks/any_folder_spec_klensPlus_RGB_coding/train_gt.py', experiment_dir)
    shutil.copy('./tasks/any_folder_spec_klensPlus_RGB_coding/spiral_gt.py', experiment_dir)
    shutil.copy('./dataloader/any_folder_spec_klensPlus_RGB_gt.py', experiment_dir)
    shutil.copy('./utils/volume_op_gt.py', experiment_dir)

    '''LOG'''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(args)

    '''Summary Writer'''
    writer = SummaryWriter(log_dir=str(experiment_dir))

    '''Data Loading'''
    scene_train = SpecDataLoaderAnyFolder(base_dir=args.base_dir,
                                          scene_name=args.scene_name,
                                          res_ratio=args.resize_ratio,
                                          load_sorted=args.train_load_sorted,
                                          load_img=True,
                                          channels=args.spec_chnls,
                                          bf_num=args.bf_num,
                                          ls_factor=args.ls_factor,
                                          ls_name=args.ls_name,
                                          cam_sen_name=args.cam_sen_name,
                                          )

    print('Train with {0:6d} images.'.format(scene_train.imgs.shape[0]))

    # We have no eval pose in this any_folder task. Eval with a 4x4 identity pose.
    # eval_c2ws = synthetic_poses(0).unsqueeze(0).float() # (1, 4, 4)
    eval_c2ws = torch.eye(4).unsqueeze(0).float()  # (1, 4, 4)

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

    # learn focal parameter
    focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order, init_focal=args.init_focal)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)

    # learn pose for each image
    pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)
    if args.multi_gpu:
        pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
    else:
        pose_param_net = pose_param_net.to(device=my_devices)

    # learn basis
    basis_net = BasisFunction(args.bf_num, args.spec_chnls, req_grad=args.learn_basis, init_basis=args.init_basis)
    if args.multi_gpu:
        basis_net = torch.nn.DataParallel(basis_net).to(device=my_devices)
    else:
        basis_net = basis_net.to(device=my_devices)

    '''Set Optimiser'''
    optimizer_nerf = torch.optim.Adam(model.parameters(), lr=args.nerf_lr)
    optimizer_focal = torch.optim.Adam(focal_net.parameters(), lr=args.focal_lr)
    optimizer_pose = torch.optim.Adam(pose_param_net.parameters(), lr=args.pose_lr)
    optimizer_basis = torch.optim.Adam(basis_net.parameters(), lr=args.basis_lr)

    scheduler_nerf = torch.optim.lr_scheduler.MultiStepLR(optimizer_nerf, milestones=args.nerf_milestones, gamma=args.nerf_lr_gamma)
    scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(optimizer_focal, milestones=args.focal_milestones, gamma=args.focal_lr_gamma)
    scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_pose, milestones=args.pose_milestones, gamma=args.pose_lr_gamma)
    scheduler_basis = torch.optim.lr_scheduler.MultiStepLR(optimizer_basis, milestones=args.basis_milestones, gamma=args.basis_lr_gamma)
    
    L = laplacian( args.spec_chnls ).to(device=my_devices)   # Laplacian smooth

    '''Training'''
    log_basis(writer, basis_net(0), 0)
    for epoch_i in tqdm(range(args.epoch), desc='epochs'):
        if args.act == 'linear':
            spec_act_fn = lambda x : x  # x=x
        elif args.act == 'sigmoid':
            spec_act_fn = torch.sigmoid # sigmoid
        elif args.act == 'exponential':
            spec_act_fn = torch.exp # exponential
        elif args.act == 'relu':    
            spec_act_fn = torch.relu    # relu
        elif args.act == 'softplus':
            spec_act_fn = torch.nn.Softplus()   # softplus

        train_epoch_losses = train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, optimizer_basis,
                                             model, focal_net, pose_param_net, basis_net, my_devices, args, spec_act_fn, L)
        train_L2_loss = train_epoch_losses['L2']
        train_L2_raw_loss = train_epoch_losses['L2_raw']
        train_L2_spec_loss = train_epoch_losses['L2_spec']
        train_L_depth_loss = train_epoch_losses['L_depth']
        scheduler_nerf.step()
        scheduler_focal.step()
        scheduler_pose.step()
        scheduler_basis.step()

        train_psnr = mse2psnr(train_L2_loss)
        train_raw_psnr = mse2psnr(train_L2_raw_loss)
        train_spec_psnr = mse2psnr(train_L2_spec_loss)
        train_depth_psnr = mse2psnr(train_L_depth_loss)
        writer.add_scalar('train/mse', train_L2_loss, epoch_i)
        writer.add_scalar('train/psnr', train_psnr, epoch_i)
        writer.add_scalar('train/lr', scheduler_nerf.get_lr()[0], epoch_i)
        
        logger.info('{0:6d} ep: Train: L2 raw loss: {1:.5f}, raw_PSNR: {2:.3f}, L2 loss: {3:.5f}, PSNR: {4:.3f}, depth loss: {5:.5f}, PSNR: {6:.3f}, spec loss: {7:.4f}, PSNR: {8:.3f}'
                    .format(epoch_i, train_L2_raw_loss, train_raw_psnr, train_L2_loss, train_psnr, train_L_depth_loss, train_depth_psnr, train_L2_spec_loss, train_spec_psnr))
        tqdm.write('{0:6d} ep: Train: L2 raw loss: {1:.5f}, raw_PSNR: {2:.3f}, L2 loss: {3:.5f}, PSNR: {4:.3f}, depth loss: {5:.5f}, PSNR: {6:.3f}, spec loss: {7:.4f}, PSNR: {8:.3f}'
                    .format(epoch_i, train_L2_raw_loss, train_raw_psnr, train_L2_loss, train_psnr, train_L_depth_loss, train_depth_psnr, train_L2_spec_loss, train_spec_psnr))
        

        if epoch_i % args.eval_interval == 0 and epoch_i > 0:
            with torch.no_grad():
                eval_one_epoch(eval_c2ws, scene_train, model, focal_net, pose_param_net, basis_net, my_devices, args, epoch_i, writer, spec_act_fn)

                fxfy = focal_net(0)
                tqdm.write('Est fx: {0:.2f}, fy {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))
                logger.info('Est fx: {0:.2f}, fy {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))

                # save the basis function image
                log_basis(writer, basis_net(0), epoch_i)

                # save the latest model
                save_checkpoint(epoch_i, model, optimizer_nerf, experiment_dir, ckpt_name='latest_nerf')
                save_checkpoint(epoch_i, focal_net, optimizer_focal, experiment_dir, ckpt_name='latest_focal')
                save_checkpoint(epoch_i, pose_param_net, optimizer_pose, experiment_dir, ckpt_name='latest_pose')
                save_checkpoint(epoch_i, basis_net, optimizer_basis, experiment_dir, ckpt_name='latest_basis')
    return


if __name__ == '__main__':
    args = parse_args() 
    set_randomness(args)
    main(args)
