import torch
import torch.nn as nn
from utils.lie_group_helper import make_c2w

import numpy as np


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.init_c2w = None
        if init_c2w is not None:
            self.init_c2w = nn.Parameter(init_c2w, requires_grad=False)

        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w[cam_id]

        return c2w
    

def synthetic_poses(viewpoint):
    Rt = torch.eye(4)
    if viewpoint == 0:
        Rt[0, -1], Rt[1, -1], Rt[2, -1] = -0.06, 0.06, 0
    elif viewpoint == 1:
        Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0, 0.06, 0
    elif viewpoint == 2:
        Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0.06, 0.06, 0
    elif viewpoint == 3:
        Rt[0, -1], Rt[1, -1], Rt[2, -1] = -0.06, 0, 0
    elif viewpoint == 4:
        Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0, 0, 0
    elif viewpoint == 5:
        Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0.06, 0, 0
    elif viewpoint == 6:
        Rt[0, -1], Rt[1, -1], Rt[2, -1] = -0.06, -0.06,0
    elif viewpoint == 7:
        Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0, -0.06, 0
    elif viewpoint == 8:
        Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0.06, -0.06, 0

    # Rt = torch.eye(4)
    # if viewpoint == 0:
    #     Rt[0, -1], Rt[1, -1], Rt[2, -1] = -0.06, 0.06, 2
    # elif viewpoint == 1:
    #     Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0, 0.06, 2
    # elif viewpoint == 2:
    #     Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0.06, 0.06, 2
    # elif viewpoint == 3:
    #     Rt[0, -1], Rt[1, -1], Rt[2, -1] = -0.06, 0, 2
    # elif viewpoint == 4:
    #     Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0, 0, 2
    # elif viewpoint == 5:
    #     Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0.06, 0, 2
    # elif viewpoint == 6:
    #     Rt[0, -1], Rt[1, -1], Rt[2, -1] = -0.06, -0.06, 2
    # elif viewpoint == 7:
    #     Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0, -0.06, 2
    # elif viewpoint == 8:
    #     Rt[0, -1], Rt[1, -1], Rt[2, -1] = 0.06, -0.06, 2

    return Rt   # [4, 4]