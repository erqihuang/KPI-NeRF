import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class MLF_Nerf(nn.Module):
    def __init__(self, pos_in_dims, dir_in_dims, D, spec_chnls):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        :param spec_channels: scalar, number of spectral channels
        """
        super(MLF_Nerf, self).__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims
        self.spec_chnls = spec_chnls

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.layers1 = nn.Sequential(
            nn.Linear(D + pos_in_dims, D), nn.ReLU(),  # shortcut
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.spectral_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D//2), nn.ReLU())
        self.fc_spectral = nn.Linear(D//2, self.spec_chnls)

        self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_spectral.bias.data = torch.zeros(self.spec_chnls, dtype=torch.float32) + 0.02

    def forward(self, pos_enc, dir_enc):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :return: rgb_density (H, W, N_sample, C=27)
        """
        x = self.layers0(pos_enc)  # (H, W, N_sample, D)
        x = torch.cat([x, pos_enc], dim=3)  # (H, W, N_sample, D+pos_in_dims)
        x = self.layers1(x)  # (H, W, N_sample, D)

        density = self.fc_density(x)  # (H, W, N_sample, 1)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        x = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        x = self.spectral_layers(x)  # (H, W, N_sample, D/2)
        spectral = self.fc_spectral(x)  # (H, W, N_sample, 27)

        spectral_den = torch.cat([spectral, density], dim=3)  # (H, W, N_sample, 27+1)
        
        return spectral_den
