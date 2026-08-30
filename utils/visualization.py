import numpy as np
import matplotlib.pyplot as plt

def generate_filter(min_wl=420, max_wl=700, spec_resolution=10, fwhm=150):
    # Wavelength range (in nanometers)
    wl = np.linspace( min_wl, max_wl, int( ( max_wl - min_wl ) / spec_resolution ) + 1 )
    tar_wl = np.linspace( min_wl, max_wl, 9 )
    output = np.zeros( ( len( tar_wl ), len( wl ), 3 ), dtype=np.float32 )

    # Define response curves for L, M, and S cones
    L_cone_response = np.exp( -0.5 * ( (wl - 630) / (38) )**2 )  # Example curve for L cones
    M_cone_response = np.exp( -0.5 * ( (wl - 540) / (38) )**2 )  # Example curve for M cones
    S_cone_response = np.exp( -0.5 * ( (wl - 470) / (38) )**2 )  # Example curve for S cones

    # Normalize response curves
    L_cone_response /= np.max(L_cone_response)
    M_cone_response /= np.max(M_cone_response)
    S_cone_response /= np.max(S_cone_response)

    # Plot original response curves
    # plt.figure()
    # plt.plot(wl, L_cone_response, label='L Cones', color='red')
    # plt.plot(wl, M_cone_response, label='M Cones', color='green')
    # plt.plot(wl, S_cone_response, label='S Cones', color='blue')
    
    # plt.figure()
    # cmap = plt.get_cmap('jet')
    for i in range( len(tar_wl) ):
        # Define blue light curve (example)
        filter_curve = np.exp ( -0.5 * ( ( wl - tar_wl[i] ) / ( 10 ) ) **2 )  # Example curve for blue light
        filter_curve /= np.max(filter_curve)

        # Dot product of blue light curve with L, M, and S cone response curves
        dot_product_L_cone = filter_curve * L_cone_response
        dot_product_M_cone = filter_curve * M_cone_response
        dot_product_S_cone = filter_curve * S_cone_response

        output[i] = np.stack( ( dot_product_L_cone, dot_product_M_cone, dot_product_S_cone ), axis=1 )

        # color = cmap( i / len(tar_wl) )
        # plt.plot(wl, filter_curve, label='Filter curve', color=color)

    # # Plot dotted curves
    # plt.figure()
    # plt.plot(wl, dot_product_L_cone, label='Dot Product L Cones', linestyle='dashed', color='red')
    # plt.plot(wl, dot_product_M_cone, label='Dot Product M Cones', linestyle='dashed', color='green')
    # plt.plot(wl, dot_product_S_cone, label='Dot Product S Cones', linestyle='dashed', color='blue')

    # # Add labels and legend
    # plt.title('Response Curves and Dot Product')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Relative Sensitivity')
    # # plt.legend()
    # plt.grid(False)
    # plt.show()

    return output   # [ N_Filter, C, 3 ]


def generate_bayer_filter(min_wl=420, max_wl=700, spec_resolution=10):

    # Wavelength range (in nanometers)
    wl = np.linspace( min_wl, max_wl, int( ( max_wl - min_wl ) / spec_resolution ) + 1 )
    output = np.zeros( ( len( wl ), 3 ), dtype=np.float32 )

    # Define response curves for L, M, and S cones
    L_cone_response = np.exp( -0.5 * ( (wl - 630) / (38) )**2, dtype=np.float32)  # Example curve for L cones
    M_cone_response = np.exp( -0.5 * ( (wl - 540) / (38) )**2, dtype=np.float32 )  # Example curve for M cones
    S_cone_response = np.exp( -0.5 * ( (wl - 470) / (38) )**2, dtype=np.float32 )  # Example curve for S cones

    # Normalize response curves
    L_cone_response /= np.max(L_cone_response)
    M_cone_response /= np.max(M_cone_response)
    S_cone_response /= np.max(S_cone_response)

    # Plot original response curves
    # plt.figure()
    # plt.plot(wl, L_cone_response, label='L Cones', color='red')
    # plt.plot(wl, M_cone_response, label='M Cones', color='green')
    # plt.plot(wl, S_cone_response, label='S Cones', color='blue')
    

    output = np.stack( ( L_cone_response, M_cone_response, S_cone_response ), axis=1 )

    return output   # [ C, 3 ]
