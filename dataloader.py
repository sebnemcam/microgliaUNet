import logging
import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch

import monai
from monai.utils import first
from monai.data import ArrayDataset, DataLoader, partition_dataset
from monai.networks.nets import Unet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity, LoadImage

#!!!!!!PATH NEEDS TO BE ADJUSTED FOR HPC !!!!!!!!!

#path_seg = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/gt_new"
#path_img = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/raw_new"

path_seg = "/Users/sebnemcam/Desktop/microglia/input cxc31/gt_new/"
path_img = "/Users/sebnemcam/Desktop/microglia/input cxc31/raw_new/"

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
#should transforms only be applied to train set?
transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
    ]
)
#zipped_data = list(zip(images,segmentations))
#print(f"Zip length: {len(zipped_data)}")


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

#split data into train, test and validation set
train_data, test_data = partition_dataset(ds, ratios=[0.8,0.2], shuffle=True)
print(f"train_data 1, length: {len(train_data)}")
train_data, val_data = partition_dataset(train_data,ratios=[0.8,0.2])

print(f"test_data, length: {len(test_data)}")
print(f"train_data, length: {len(train_data)}")
print(f"val_data, length: {len(val_data)}")

fig, ax = plt.subplots(2, 1, figsize=(8, 4))

print("Visualizing <3")

ax[0].imshow(np.hstack(batch[0][:, 0, 80]),cmap='gray')
ax[0].set_title("Batch of Images")

ax[1].imshow(np.hstack(batch[1][:, 0, 80]),cmap='gray')
ax[1].set_title("Batch of Segmentations")

plt.tight_layout()
plt.show()
#fig.savefig("/lustre/groups/iterm/sebnem/slurm_outputs/batches.png")


device = "cuda" if torch.cuda.is_available() else "cpu"

#initializing model

model = Unet(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2,2,2,2),
    norm=Norm.BATCH,
    #act=torch.nn.functional.sigmoid()
).to(device)

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric()

epoch_loss_values = []

max_epochs = 200
val_interval = max_epochs/4

for epoch in range(1, max_epochs):
    print("-" * 10)
    print(f"epoch {epoch}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in tqdm(loader):
        step += 1
        inputs, labels = (
            batch_data[0].to(device),
            batch_data[1].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
             f"{step}/{len(train_ds) // train_loader.batch_size}, "
             f"train_loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if epoch % val_interval == 0:
        model.eval()

