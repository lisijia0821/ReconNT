import numpy as np
from scipy.ndimage import median_filter, gaussian_filter


def remove_gamma_spikes(img, threshold=3):
    """
    Remove isolated gamma spikes using median filtering.
    """
    med = median_filter(img, size=3)
    diff = img - med
    std = np.std(diff)
    mask = diff > threshold * std
    img[mask] = med[mask]
    return img


def remove_stripes_projection_2d(img, size_vertical=3, size_horizontal=3, bad_pixel_sigma=6):
    """
    Remove vertical stripes, horizontal stripes, and hot pixels.
    """
    img = median_filter(img, size=(1, size_vertical))
    img = median_filter(img, size=(size_horizontal, 1))

    med = median_filter(img, size=3)
    diff = img - med
    threshold = bad_pixel_sigma * np.std(diff)
    mask = np.abs(diff) > threshold
    img[mask] = med[mask]

    return img


def normalize_projection(img, flat_img, dark_val=600):
    """
    Apply flat-field correction and clipping.
    """
    img = img.astype(np.float32)
    img -= dark_val
    img[img < 0] = 0

    # gamma spike removal
    img = remove_gamma_spikes(img, threshold=3)

    # flat normalization
    norm = np.divide(img, flat_img, where=(flat_img > 0), out=np.zeros_like(img))

    # stripe removal & smoothing
    norm = remove_stripes_projection_2d(norm)
    norm = np.clip(norm, 1e-3, 10)
    norm = median_filter(norm, size=3)
    norm = gaussian_filter(norm, sigma=0.5)

    return norm
