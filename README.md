# ExSRRF Analysis Code

In this repository you find the code used to register, evaluate and analyse the data in our publication:


### register_time_stacks.py
Script that registers the time stacks saved as tiff files in a folder and calculated the SSIM and MSE to evaluate the registrations.

### compare_exm_exsrrf_sted.ipynb
Notebook that was used to compare the SSIM between ridges and original images for ExM, ExSRRF and STED images.

### ridges_slit_diaphragm.ipynb
Notebook to extract regions of interest and ridges (slit diaphragm) and calculate their densities

### ridges_er_stress.ipynb
Notebook to extract regions of interest and ridges (ER) and calculate their densities

### local_spacing.ipynb (GPLv3)
Notebook to compute the local spacing of ridge images. Local spacing calculation is based on the local_thickness plugin of Fiji (ImageJ) and was ported to python and adapted for our 2D use case.

## Gallery
Output of local_spacing.ipynb for given sythetic test image:
1. original
2. distance transformed image
3. ridge image
4. local spacing image
<p align="center">
  <img src="https://github.com/imsb-uke/exsrrf_analyses/blob/main/gallery/test_ridges_img_output.png" width="85%"></img>
</p>
