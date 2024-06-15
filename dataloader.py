import logging
import os
import time

import pandas as pd
from monai.inferers import sliding_window_inference
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torchmetrics

import monai
from monai.utils import first
from monai.data import ArrayDataset, DataLoader, partition_dataset, decollate_batch, Dataset, CacheDataset
from monai.networks.nets import Unet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity, LoadImage, LoadImaged, \
    Resized, ToTensord, RandFlipd, AsDiscrete

#!!!!!!PATH NEEDS TO BE ADJUSTED FOR HPC !!!!!!!!!

path_seg = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/gt_new"
path_img = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/raw_new"
directory= "/lustre/groups/iterm/sebnem/"

#path_seg = "/Users/sebnemcam/Desktop/microglia/input cxc31/gt_new/"
#path_img = "/Users/sebnemcam/Desktop/microglia/input cxc31/raw_new/"
#directory = "/Users/sebnemcam/Desktop/Helmholtz/"

seg_list = os.listdir(path_seg)
img_list = os.listdir(path_img)
segmentations = []
images = []

print("I am inside the python file")

for i in range(len(seg_list)):
    if "nii.gz" in seg_list[i] and img_list[i] == seg_list[i]:

        segFile = os.path.join(path_seg, seg_list[i])
        segmentations.append(segFile)

        imgFile = os.path.join(path_img, img_list[i])
        images.append(imgFile)


# zip data into one list of image/segmenttaion pairs
zipped_data = list(zip(images,segmentations))
#print(f"Zip length: {len(zipped_data)}")

#split into train, test & validation files
ratio_a = len(zipped_data)*0.8
train_files_a = [zipped_data[i] for i in range(int(ratio_a))]
test_files = [zipped_data[i] for i in range(int(ratio_a),len(zipped_data))]
ratio_b = len(train_files_a)*0.8
train_files = [train_files_a[i] for i in range(int(ratio_b))]
val_files = [train_files_a[i] for i in range(int(ratio_b),len(train_files_a))]

train_data = [{'image' : img, 'segmentation' : seg } for img,seg in train_files]
val_data = [{'image' : img, 'segmentation' : seg } for img,seg in val_files]
test_data = [{'image' : img, 'segmentation' : seg } for img,seg in test_files]

'''
print(f"Len Train Data: {len(train_data)}")
print(f"Len Val Data: {len(val_data)}")
print(f"Len Test Data: {len(test_data)}")
'''

keys = ['image','segmentation']

dic_transforms_train = Compose(
    [
        LoadImaged(keys,image_only=True,ensure_channel_first=True),
        Resized(keys,spatial_size=(128, 128, 128)),
        ToTensord(keys),
        RandFlipd(keys, spatial_axis=0, prob=0.5),
        RandFlipd(keys, spatial_axis=1, prob=0.5),
        RandFlipd(keys, spatial_axis=2, prob=0.5),

    ]
)

dic_transforms = Compose(
    [
        LoadImaged(keys, ensure_channel_first=True, image_only=True),
        Resized(keys,spatial_size=(128, 128, 128)),
        ToTensord(keys)
    ]
)

train_set = CacheDataset(train_data,dic_transforms_train)
test_set = CacheDataset(test_data,dic_transforms)
val_set = CacheDataset(val_data,dic_transforms_train)

#visualizing some slices from the training data to make sure everything is fine
sample = train_set[0]
im,seg = sample['image'], sample['segmentation']
num_slices = im.shape[1]
num_rows = (num_slices // 10)
fig, axes = plt.subplots(num_rows-1, 2, figsize=(12, 6 * num_rows))
for idx, slice_idx in enumerate(range(0, num_slices, max(1, num_slices // 10))):  # Sample up to 10 slices
    row = idx
    # Image slice subplot
    axes[row, 0].imshow(im.numpy()[0, slice_idx], cmap='gray')
    axes[row, 0].set_title(f"Image Slice {slice_idx}")
    axes[row, 0].axis('off')

    # Segmentation slice subplot
    axes[row, 1].imshow(seg.numpy()[0, slice_idx], cmap='gray')
    axes[row, 1].set_title(f"Segmentation Slice {slice_idx}")
    axes[row, 1].axis('off')
plt.tight_layout()
plt.show()
#fig.savefig("/lustre/groups/iterm/sebnem/slurm_outputs/slices.png")

'''
print(f"test_data, length: {len(test_data)}")
print(f"train_data, length: {len(train_data)}")
print(f"val_data, length: {len(val_data)}")
'''

#build Dataloader
train_loader = DataLoader(train_set,batch_size=5)
test_loader = DataLoader(test_set,batch_size=5)
val_loader = DataLoader(val_set,batch_size=5)

print("Checkpoint 1")

device = "cuda" if torch.cuda.is_available() else "cpu"

#initializing model

model = Unet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2,2,2,2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

print("Checkpoint 2")

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = torchmetrics.Dice(zero_division=1)
dice_metric = DiceMetric(ignore_empty=False)
metric_values = []

epoch_loss_values = []

max_epochs = 1000
val_interval = 1
best_metric = -1
post_pred = AsDiscrete(argmax=True)
post_label=AsDiscrete()

print("Checkpoint 3")
for epoch in range(0, max_epochs):
    print("-" * 10)
    print(f"epoch {epoch+1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in test_loader:
        step += 1
        img, seg = (
            batch_data['image'].to(device),
            batch_data['segmentation'].to(device),
        )
        #print(f"Input to model: {img.size()}")
        #print(f"Segmentations: {seg.size()}")
        optimizer.zero_grad()
        outputs = model(img)
        #print(f"Output from model: {outputs.shape}")
        loss = loss_function(outputs, seg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
             f"{step}/{len(train_data) // train_loader.batch_size}, "
             f"train loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if epoch % val_interval == 0:
        print("Validation")
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_img, val_seg = (
                    val_data['image'].to(device),
                    val_data['segmentation'].to(device)
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_img, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_seg = [post_label(i) for i in decollate_batch(val_seg)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_seg)
                #print(f"output: {val_outputs}/n label: {val_seg}")

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                print(f"Dice Value: {metric}")

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        directory, "best_metric_model.pth"))
                    print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                        f"\nbest mean dice: {best_metric:.4f} "
                        f"at epoch: {best_metric_epoch}"
                    )

'''print(f"losses: {len(epoch_loss_values)}")
print(f"dice values: {len(metric_values)}")
print(f"best epoch: {len(best_metric_epoch)}")
print(f"best dice: {len(best_metric)}")'''


'''checkpoint =pd.DataFrame( {

     'train_loss': epoch_loss_values,
     'val_dice': metric_values,
     'best_metric_epoch': best_metric_epoch,
     'best_metric':best_metric
})



val_interval = 2
plt.figure("train", (15, 5))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Dice Loss")
x = [i + 1 for i in range(len(checkpoint["train_loss"]))]
y = checkpoint["train_loss"]
plt.xlabel("#Epochs")
plt.ylabel("Dice Loss")
plt.plot(x, y)
plt.plot(checkpoint["best_metric_epoch"],
checkpoint["train_loss"][checkpoint["best_metric_epoch"]], 'r*', markersize=8)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice Score")
x = [val_interval * (i + 1) for i in range(len(checkpoint["val_dice"]))]
y = checkpoint["val_dice"]
plt.xlabel("#Epochs")
plt.plot(x, y)
plt.plot(checkpoint["best_metric_epoch"],
checkpoint["val_dice"][checkpoint["best_metric_epoch"]//2], 'r*', markersize=10)
plt.annotate("Best Score[470, 0.9516]", xy=(checkpoint["best_metric_epoch"],
checkpoint["val_dice"][checkpoint["best_metric_epoch"]//2]))'''

# Plotting the loss and dice scores
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values)
plt.title("Epoch Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(range(1, len(metric_values) + 1), metric_values)
plt.title("Dice Metric")
plt.xlabel("Epochs")
plt.ylabel("Dice Score")

plt.show()
plt.savefig("/lustre/groups/iterm/sebnem/LearningCurves_MPL.png")
#plt.savefig("/Users/sebnemcam/Desktop/LearningCurves.png")


'''
STILL TO DO
    - track metrics
    - 5-fold cross Validation
'''
