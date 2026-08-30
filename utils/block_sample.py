import torch

# def sample_blocks_v0(ray_dir_cam, img, H, W, device, num_blocks=16, block_size=2):
#     total_blocks = num_blocks * num_blocks  # e.g., 16x16 = 256
#     ray_blocks = []
#     img_blocks = []

#     for _ in range(total_blocks):
#         # 2x2
#         r0 = torch.randint(0, H - block_size, (1,), device=device).item()
#         c0 = torch.randint(0, W - block_size, (1,), device=device).item()

#         r_idx = torch.arange(r0, r0 + block_size, device=device)
#         c_idx = torch.arange(c0, c0 + block_size, device=device)

#         ray_block = ray_dir_cam[r_idx][:, c_idx]  # (2, 2, 3)
#         img_block = img[r_idx][:, c_idx]          # (2, 2, 1)

#         ray_blocks.append(ray_block)
#         img_blocks.append(img_block)

#     #  (16, 16, 2, 2, C) -> (32, 32, C)
#     ray_blocks = torch.stack(ray_blocks, dim=0).view(num_blocks, num_blocks, 2, 2, -1)
#     img_blocks = torch.stack(img_blocks, dim=0).view(num_blocks, num_blocks, 2, 2, -1)

#     ray_selected_cam = ray_blocks.permute(0, 2, 1, 3, 4).reshape(2 * num_blocks, 2 * num_blocks, -1)  # (32, 32, 3)
#     img_selected = img_blocks.permute(0, 2, 1, 3, 4).reshape(2 * num_blocks, 2 * num_blocks, -1)      # (32, 32, 1)

#     return ray_selected_cam, img_selected


def sample_blocks(ray_dir_cam, img, H, W, device, num_blocks=16, block_size=2):
    total_blocks = num_blocks * num_blocks  # e.g., 16x16 = 256

    r0 = torch.randint( 0, H - block_size, (total_blocks,), device=device )
    c0 = torch.randint( 0, W - block_size, (total_blocks,), device=device )

    # 2x2
    dr = torch.tensor([0, 0, 1, 1], device=device)
    dc = torch.tensor([0, 1, 0, 1], device=device)

    r_idx = ( r0[:, None] + dr[None, :] ).reshape(-1)  # (256×4,)
    c_idx = ( c0[:, None] + dc[None, :] ).reshape(-1)  # (256×4,)

    # 256×2×2
    rays = ray_dir_cam[r_idx, c_idx].reshape(total_blocks, 2, 2, -1)
    imgs = img[r_idx, c_idx].reshape(total_blocks, 2, 2, -1)

    # (16, 16, 2, 2, C) -> (32, 32, C)
    rays = rays.view(num_blocks, num_blocks, 2, 2, -1).permute(0, 2, 1, 3, 4).reshape(2 * num_blocks, 2 * num_blocks, -1)
    imgs = imgs.view(num_blocks, num_blocks, 2, 2, -1).permute(0, 2, 1, 3, 4).reshape(2 * num_blocks, 2 * num_blocks, -1)

    return rays, imgs

