import numpy as np
import tomopy
from scipy.ndimage import gaussian_filter


def remove_ring_with_soft_mask(recon, rwidth=35, thresh=0.0005, radius=40, sigma=2):
    """
    Apply tomopy.remove_ring with a soft blending mask toward the periphery.
    """
    recon_filtered = recon.copy()
    num_slices, h, w = recon.shape

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    center_y, center_x = h // 2, w // 2
    dist = np.sqrt((yy - center_y)**2 + (xx - center_x)**2)

    mask = 1 - np.exp(-((dist - radius) ** 2) / (2 * sigma ** 2))
    mask[dist <= radius] = 0
    mask = mask.clip(0, 1)

    for i in range(num_slices):
        slice_orig = recon[i]
        slice_filtered = tomopy.remove_ring(
            slice_orig[None, :, :], rwidth=rwidth, thresh=thresh
        )[0]

        recon_filtered[i] = slice_orig * (1 - mask) + slice_filtered * mask

    return recon_filtered


def fuse_recon_with_mask(recon_fw, recon, radius_cut=100, sigma_blur=10, zmax=None):
    """
    Fuse standard reconstruction and fw-reconstruction using a circular mask.
    """
    if zmax is None:
        zmax = min(recon.shape[0], recon_fw.shape[0])

    h, w = recon_fw.shape[1:]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2

    mask = ((X - cx)**2 + (Y - cy)**2) <= radius_cut**2
    mask = gaussian_filter(mask.astype(np.float32), sigma=sigma_blur)

    recon_final = recon_fw[:zmax] * mask + recon[:zmax] * (1 - mask)
    return recon_final
