import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch

import monai
from monai.utils import first
from monai.data import ArrayDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity, LoadImage

#!!!!!!PATH NEEDS TO BE ADJUSTED FOR HPC !!!!!!!!!

path_seg = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/gt_new"
path_img = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/raw_new"

#path_seg = "/Users/sebnemcam/Desktop/microglia/input cxc31/gt_new/"
#path_img = "/Users/sebnemcam/Desktop/microglia/input cxc31/raw_new/"

seg_list = os.listdir(path_seg)
img_list = os.listdir(path_img)
segmentations = []
images = []

for i in range(len(seg_list)):
    if "nii.gz" in seg_list[i] and img_list[i] == seg_list[i]:

        segFile = os.path.join(path_seg, seg_list[i])
        #segmentations.append(nib.load(segFile))
        segmentations.append(segFile)
        #print(segmentations)

        imgFile = os.path.join(path_img, img_list[i])
        #images.append(nib.load(imgFile))
        images.append(imgFile)
        #print(images)

#which transforms should be applied????, croping & flipping seems countereffective
transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
    ]
)

ds = ArrayDataset(images, transforms, segmentations, transforms)
im, seg = ds[0]

'''num_slices = im.shape[1]
for slice_idx in range(0, num_slices, max(1, num_slices // 10)):  # sample up to 10 slices
    #time.sleep(1)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Image Slice {slice_idx}")
    plt.imshow(im.numpy()[0, slice_idx], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title(f"Segmentation Slice {slice_idx}")
    plt.imshow(seg.numpy()[0, slice_idx], cmap='gray')
    plt.show()'''

#build Dataloader
#0 = img, 1 = seg
loader = DataLoader(ds, batch_size=5)
batch = first(loader)
#print(batch[0].shape)

fig, ax = plt.subplots(2, 1, figsize=(8, 4))

print("Visualizing <3")

ax[0].imshow(np.hstack(batch[0][:, 0, 80]),cmap='gray')
ax[0].set_title("Batch of Images")

ax[1].imshow(np.hstack(batch[1][:, 0, 80]),cmap='gray')
ax[1].set_title("Batch of Segmentations")

plt.tight_layout()
plt.show()
fig.savefig("/lustre/groups/iterm/sebnem/slurm_outputs/batches.png")
