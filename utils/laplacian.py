import torch

# [1, -2, 1]
def laplacian( n ):
    main_diag = -2 * torch.ones(n)
    off_diag = 1 * torch.ones(n - 1)

    L = torch.diag(main_diag)
    L += torch.diag( off_diag, diagonal=1 )
    L += torch.diag( off_diag, diagonal=-1 )

    # Neumann BC
    L[0, 1] = 2
    L[-1, -2] = 2

    return L