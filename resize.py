import os
import nibabel as nib
import numpy as np
from monai.transforms import Resize

'''
Resized the raw data we already have to match the size of our output data
'''

nifti_file_path = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/raw_new/"

seg_list = os.listdir(nifti_file_path)

for file in seg_list:

    img = nib.load(os.path.join(nifti_file_path,file))
    data = img.get_fdata()
    resize = Resize(spatial_size=(128,128,128))
    resized_data = resize(data[np.newaxis, ...])[0]  # Adding a channel dimension

    # Create a new NIfTI image
    new_img = nib.Nifti1Image(resized_data, img.affine, img.header)
    new_nifti_file_path = os.path.join("/lustre/groups/iterm/sebnem/resized_raws/",file)
    nib.save(new_img, new_nifti_file_path)
