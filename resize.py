import os
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose, LoadImage, Resize
'''
path_img = "/lustre/groups/iterm/sebnem/runs/02.07_11:56/test_fold0/outputs"
path_img = "/Users/sebnemcam/Desktop"

# Load the NIfTI file from the desktop
nifti_file_path = "/Users/sebnemcam/Desktop"

img_list = os.listdir(path_img)
for i in range(len(img_list)):
    if "nii.gz" in img_list[i]:

        img = nib.load(os.path.join(nifti_file_path,img_list[i]))
        data = img.get_fdata()

        resize = Resize(spatial_size=(100,100,100))
        resized_data = resize(data[np.newaxis, ...])[0]
        new_img = nib.Nifti1Image(resized_data, img.affine, img.header)

     
        imgFile = os.path.join(path_img, img_list[i])
        loader = LoadImage(reader='nibabelreader', image_only=True)
        img = loader(imgFile)
        sizer = Resize(spatial_size=(100,100,100))
        img = sizer(img)
     
        nib.save(new_img, os.path.join(path_img, "/Users/sebnemcam/Desktop/new_img"))
'''

nifti_file_path = "/Users/sebnemcam/Desktop/microglia/input cxc31/raw_new/patchvolume_2433_1.nii.gz"
img = nib.load(nifti_file_path)
data = img.get_fdata()
resize = Resize(spatial_size=(128,128,128))
resized_data = resize(data[np.newaxis, ...])[0]  # Adding a channel dimension

# Create a new NIfTI image
new_img = nib.Nifti1Image(resized_data, img.affine, img.header)
new_nifti_file_path = "/Users/sebnemcam/Desktop/new_raw"
nib.save(new_img, new_nifti_file_path)
