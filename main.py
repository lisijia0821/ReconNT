import os
import numpy as np
import tifffile
import tomopy

from preprocessing import normalize_projection
from ring_and_fusion import remove_ring_with_soft_mask, fuse_recon_with_mask


# ---- User-modifiable paths ----
proj_dir = r"path/to/projection_folder"
flat_path = r"path/to/flat_field.tif"
output_dir = r"path/to/output"

os.makedirs(output_dir, exist_ok=True)

dark_val = 600
# To reduce the amount of data processed, a subset is loaded. 
# You can also crop the data as needed, but remember to adjust (or find) your rotation center later.
crop_roi = (slice(322, 1398), slice(500, 1331))   

# ---- Read projection file list ----
files = sorted([f for f in os.listdir(proj_dir) if f.endswith(".tif")])

# ---- Define angle range ----
n_angles = 259
theta = np.deg2rad(np.linspace(-240, -59.4, n_angles))
# n_angles = len(files) // 2

# ---- Load and crop flat-field image ----
flat_img = tifffile.imread(flat_path).astype(np.float32)[crop_roi]
flat_img -= dark_val
flat_img[flat_img <= 0] = np.min(flat_img[flat_img > 0])

# ---- Initialize projection array ----
h, w = flat_img.shape
proj = np.zeros((n_angles, h, w), dtype=np.float32)

# ---- Pair-wise averaging of projections ----
for k in range(n_angles):
    idx1, idx2 = 2*k, 2*k + 1

    img1 = tifffile.imread(os.path.join(proj_dir, files[idx1]))[crop_roi]
    img2 = tifffile.imread(os.path.join(proj_dir, files[idx2]))[crop_roi]

    avg_img = np.minimum(img1, img2)
    norm = normalize_projection(avg_img, flat_img, dark_val=dark_val)
    proj[k] = norm

# ---- Logarithmic transformation ----
# Users may need to adjust these parameters according to their data
proj_fw = tomopy.remove_stripe_fw(proj, level=5, wname='db5', sigma=1)

proj = -np.log(np.clip(proj, 1e-3, 10))
proj_fw = -np.log(np.clip(proj_fw, 1e-3, 10))

# ---- Set rotation center ----
y1, c1 = 400, 458
y2, c2 = 1200, 453
a = (c2 - c1) / (y2 - y1)
b = c1 - a * y1
y_range = np.arange(322, 1398)
centers = a * y_range + b
center = tomopy.find_center()
print(a,b)
fixed_center = a*958+b

# ---- Reconstruction ----
recon = tomopy.recon(proj, theta, center=fixed_center, algorithm='gridrec')
recon_fw = tomopy.recon(proj_fw, theta, center=fixed_center, algorithm='gridrec')

# ---- Ring artifacts removal ----
recon_ring = remove_ring_with_soft_mask(recon, rwidth=15, thresh=0.0044, radius=40, sigma=10)

# ---- Fuse two datasets ----
recon_final = fuse_recon_with_mask(recon_fw, recon_ring, radius_cut=70, sigma_blur=20)

# ---- Save reconstructed slices ----
recon_final = np.clip(recon_final, 0, None)[::-1]

for i in range(recon_final.shape[0]):
    tifffile.imwrite(os.path.join(output_dir, f"slice_{i:04d}.tif"), recon_final[i])

print("Reconstruction saved to:", output_dir)
