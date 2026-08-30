import torch
import numpy as np

import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def log_basis(writer, bf, epoch_i):

    x_wl = np.linspace(0, 1, bf.shape[1])  # assumes shape [N, 29]

    # Create matplotlib figure and axes
    fig, ax = plt.subplots()

    for i in range(bf.shape[0]):
        ax.plot(x_wl, bf[i].detach().cpu().numpy())

    ax.set_title("Basis Functions")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Amplitude")

    # Save to buffer
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)  # close the figure properly
    buf.seek(0)

    # Convert to numpy
    image = Image.open(buf).convert('RGB')
    image_np = np.transpose( np.array(image), (2 ,0, 1) )

    # Log to TensorBoard
    writer.add_image('eval_basis', image_np, global_step=epoch_i)


def DCT_bf(bf_num, spec_CH):
    """Construct DCT basis function
    :param bf_num: (1, ) number of basis function
    :param spec_CH: (1, ) spectral channel number
    :return:  (2xN_BF+1, C)
    """

    bf = np.zeros( ( bf_num, spec_CH ), dtype=np.float32 )    # [ N_BF, C ]

    bf[0, :] = 1 / np.sqrt( spec_CH )
    
    norm_wl = np.linspace(0, spec_CH-1, num=spec_CH) / spec_CH

    for m in range( bf_num ):
        bf[ m, : ] = np.sqrt( 2 / spec_CH ) * np.cos( np.pi * ( norm_wl + 0.5 / spec_CH ) * m )

    bf[ 0, : ] /= np.sqrt( 2 )

    return bf


def Fourier_bf(bf_enc_levels, spec_CH):
    """Construct DCT basis function
    :param bf_enc_levels: (1, ) encoding level of basis function
    :param spec_CH: (1, ) spectral channel number
    :return:  (2xN_BF+1, C)
    """

    bf_num = 2 * bf_enc_levels + 1  # number of basis

    bf = np.zeros( ( bf_num, spec_CH ), dtype=np.float32 )    # [ 2*N_BF+1, C ]

    bf[0, :] = 1 / np.sqrt( spec_CH )

    # norm_wl = ( wl - wl[0] ) / ( wl[-1] - wl[0] )    # normalize the wavelength to [0,1]
    norm_wl = np.linspace(0, spec_CH-1, num=spec_CH) / spec_CH
    
    for m in range( bf_enc_levels ):
        bf[ 2 * m + 1, : ] = np.sqrt( 2 / spec_CH ) * np.cos( 2 * np.pi * ( m + 1 ) * norm_wl )
        bf[ 2 * m + 2, : ] = np.sqrt( 2 / spec_CH ) * np.sin( 2 * np.pi * ( m + 1 ) * norm_wl )

    return bf


def Legendre_bf(bf_enc_levels, spec_CH):
    """Construct Legendre polynomial basis functions
    :param bf_enc_levels: (1,) number of basis functions (highest degree = bf_enc_levels - 1)
    :param spec_CH: (1,) spectral channel number
    :return: (M, C)
    """
    M = 2 * bf_enc_levels + 1
    bf = np.zeros((M, spec_CH), dtype=np.float32)

    wl = np.arange(spec_CH, dtype=np.float64)
    t = (2 * wl - (wl[0] + wl[-1])) / (wl[-1] - wl[0])

    P_prev = np.ones(spec_CH, dtype=np.float64)
    bf[0, :] = P_prev * np.sqrt(1.0 / (wl[-1] - wl[0]))

    if M > 1:
        P_curr = t.copy()
        bf[1, :] = P_curr * np.sqrt(3.0 / (wl[-1] - wl[0]))
        for m in range(2, M):
            P_next = ((2*m - 1) * t * P_curr - (m - 1) * P_prev) / m
            norm_factor = np.sqrt((2*m + 1) / (wl[-1] - wl[0]))
            bf[m, :] = P_next * norm_factor
            P_prev, P_curr = P_curr, P_next

    return bf


def Chebyshev_bf():
    return