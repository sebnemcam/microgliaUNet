import cc3d
import nibabel as nib
import numpy as np
import os
from cc3d import connected_components
import  torch
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill

#get raws and segmentations - DONE
#connected component analysis
#jedes component --> raw * gt
#davon hellster voxel mit np.max
#seed point = np.where(x)[0]

#soma mit binary dialation wachsen lassen

path_raws = "/lustre/groups/iterm/sebnem/resized_raws/"
path_segmentations = "/lustre/groups/iterm/sebnem/runs/04.07_12:00/test_fold1/outputs/"

patches = os.listdir(path_segmentations)

images_raw = []
images_seg = []
component_list_raw = []
component_list_seg = []

for patch in patches:

    file_raw = os.path.join(path_raws, patch)
    file_seg = os.path.join(path_segmentations, patch)

    img_raw = nib.load(file_raw)
    img_seg = nib.load(file_seg)

    raw_a = img_raw.get_fdata()
    seg_a = img_seg.get_fdata()

    overlap = np.multiply(raw_a,seg_a)

    images_raw.append(raw_a)
    images_seg.append(seg_a)

    #components_raw = connected_components(raw_a)
    #components_seg = connected_components(seg_a)

    components = connected_components(overlap)

    for label, image in cc3d.each(components, binary=False, in_place=True):
        print(f"image: {image}")
        brightest_value = np.max(image)
        seed = np.where(raw_a == brightest_value)[0]



    #overlaping_areas = np.multiply(component_list_raw,component_list_seg)





