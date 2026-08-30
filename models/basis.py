import torch
import torch.nn as nn
import numpy as np

from utils.basis_func import DCT_bf, Fourier_bf, Legendre_bf, Chebyshev_bf
 
class BasisFunction(nn.Module):
    def __init__(self, bf_num, spec_CH, req_grad, init_basis=None):
        """
        :param bf_num:  True/False
        :param spec_CH:  True/False
        :param new_x:  True/False
        :param req_grad:  True/False
        :param init_basis: True/False
        """
        super(BasisFunction, self).__init__()

        self.init_basis = init_basis
        self.bf_num = bf_num
        self.spec_CH = spec_CH
        D = 128
        self.bf_enc_levels = bf_num // 2

        if init_basis == 'Fourier':
            bf = Fourier_bf(self.bf_enc_levels, self.spec_CH)
        elif init_basis == 'DCT':
            bf = DCT_bf(self.bf_num, self.spec_CH)
        elif init_basis == 'Legendre':
            bf = Legendre_bf(self.bf_num, self.spec_CH)
        elif init_basis == 'Chebyshev':
            bf = Chebyshev_bf(self.bf_num, self.spec_CH)
        elif init_basis == 'Identity':
            bf = np.eye( self.spec_CH, bf_num, dtype=np.float32 )   # ( N_BF, C )
        else:
            # self.basis_N = 2 * self.bf_enc_levels + 1
            wl = np.linspace(420, 700, spec_CH)
            wl_norm = (wl - wl[0]) / (wl[-1] - wl[0])
            bf = np.array( wl_norm, dtype = np.float32)

        bf = torch.from_numpy( bf )
        self.bf = nn.Parameter( bf, requires_grad = req_grad )

        self.layers0 = nn.Sequential(
            nn.Linear(spec_CH, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )
        self.fc_basis = nn.Linear(D, self.bf_num * spec_CH)
        self.fc_basis.bias.data = torch.zeros( self.bf_num * spec_CH, dtype=torch.float32 ) + 0.02


        # self.ef = nn.Parameter( torch.tensor( 1, dtype = torch.float32), requires_grad = req_grad )  # engery factor

    def forward(self, cam_id):

        # bf = self.layers0( self.bf )
        # bf = self.fc_basis( bf )
        # out = bf.view( self.bf_num, self.spec_CH ) # m<n
        
        out = self.bf

        # out, _ = torch.linalg.qr(out.transpose(1, 0))
        # out = out.transpose(1,0)

        return out
    