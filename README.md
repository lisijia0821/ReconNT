# Neutron Tomography Reconstruction Workflow

This repository contains the full Python workflow used for the neutron tomography reconstruction of marine-corroded iron cultural heritage objects. The associated paper focuses on the scientific interpretation, rather than the implementation details. Therefore, the complete reconstruction pipeline is provided here to ensure reproducibility without occupying space in the manuscript.

## What you need to modify for your own data

To apply this reconstruction pipeline to your own neutron tomography dataset, please adjust the following parameters in `main.py`:

+  Projection image directory (`proj_path`)
+  Flat-field image directory (`flat_path`)
+  Output directory for reconstructed volumes(`output_dir`)
+  Dark-field value or file (`dark_value`)
+  Rotation angle list (`theta`)
+  Rotation center (`fixed_center`)
+  Image cropping range or downsampling factor
+  Ring- and stripe-artifact removal parameters, including, e.g.:
   `remove_stripe_fw()` → `level`, `wname`, `sigma`
   `remove_ring_with_soft_mask()` → `rwidth`, `thresh`, `radius`, `sigma`
+ Fusion / masking settings used in post-processing:
   `fuse_recon_with_mask()` → `radius_cut`, `sigma_blur`

## Project Structure 
```
├── main.py, Main pipeline: loading → preprocessing → reconstruction → post-processing

├── preprocessing.py, Ring, stripe, and spike artifact removal methods

├── ring_and_fusion.py, Soft ring removal and Imaging fusion

└── README.md
```
