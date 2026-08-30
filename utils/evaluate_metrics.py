import torch
import torch.nn.functional as F
from math import pi

def evaluate_metrics(ref_img: torch.Tensor, test_img: torch.Tensor):
    """
    Compute PSNR, SSIM, SAM, and RMSE between two hyperspectral images.

    Args:
        ref_img  : Reference hyperspectral image (H, W, C), torch.Tensor (float32/64)
        test_img : Reconstructed hyperspectral image (H, W, C), torch.Tensor

    Returns:
        psnr_val (float)
        ssim_val (float)
        sam_val  (float, radians)
        rmse_val (float)
    """

    # Ensure same shape
    assert ref_img.shape == test_img.shape, "Images must have the same dimensions"
    H, W, C = ref_img.shape

    ref_img  = ref_img.double()
    test_img = test_img.double()

    # --- 1. PSNR ---
    mse = torch.mean( (ref_img - test_img) ** 2 )
    psnr_val = -10 * torch.log10( mse )
    # psnr_val = -10 * torch.log10( mse + torch.finfo(mse.dtype).eps )

    # --- 2. SSIM ---
    # SSIM per channel, averaged
    # We'll implement a simple version with a Gaussian filter
    def ssim(img1, img2, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
        # expects shape (1,1,H,W)
        device = img1.device
        coords = torch.arange(window_size, dtype=img1.dtype, device=device) - window_size // 2
        g = torch.exp(-(coords**2) / (2*sigma*sigma))
        g = g / g.sum()
        window = (g[:, None] @ g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)

        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)

        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=1) - mu2_sq
        sigma12   = F.conv2d(img1*img2, window, padding=window_size//2, groups=1) - mu1_mu2

        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    ssim_vals = []
    for c in range(C):
        ch1 = ref_img[:,:,c].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        ch2 = test_img[:,:,c].unsqueeze(0).unsqueeze(0)
        ssim_vals.append(ssim(ch1, ch2))
    ssim_val = torch.stack(ssim_vals).mean()

    # --- 3. SAM (Spectral Angle Mapper) ---
    ref_vec  = ref_img.reshape(-1, C)  # (N,C)
    test_vec = test_img.reshape(-1, C)

    dot_prod = torch.sum(ref_vec * test_vec, dim=1)
    norm_ref = torch.norm(ref_vec, dim=1)
    norm_test= torch.norm(test_vec, dim=1)

    cos_theta = dot_prod / (norm_ref * norm_test + torch.finfo(ref_vec.dtype).eps)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    sam_map   = torch.acos(cos_theta)
    sam_val   = torch.mean(sam_map[~torch.isnan(sam_map)])

    # --- 4. RMSE ---
    rmse_val = torch.sqrt(mse)

    print(f"PSNR : {psnr_val.item():.4f} dB")
    print(f"SSIM : {ssim_val.item():.4f} (mean over {C} channels)")
    print(f"SAM  : {sam_val.item():.4f} rad ({sam_val.item()*180/pi:.2f} deg)")
    print(f"RMSE : {rmse_val.item():.4f}")

    return psnr_val.item(), ssim_val.item(), sam_val.item(), rmse_val.item()
