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

import monai
from monai.utils import first
from monai.data import ArrayDataset, DataLoader, partition_dataset, decollate_batch
from monai.networks.nets import Unet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity, LoadImage

#!!!!!!PATH NEEDS TO BE ADJUSTED FOR HPC !!!!!!!!!

path_seg = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/gt_new"
path_img = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/raw_new"

#path_seg = "/Users/sebnemcam/Desktop/microglia/input cxc31/gt_new/"
#path_img = "/Users/sebnemcam/Desktop/microglia/input cxc31/raw_new/"
#directory = "/Users/sebnemcam/Desktop/Helmholtz/"
directory= "/lustre/groups/iterm/sebnem/"

seg_list = os.listdir(path_seg)
img_list = os.listdir(path_img)
segmentations = []
images = []

for i in range(len(seg_list)):
    if "nii.gz" in seg_list[i] and img_list[i] == seg_list[i]:

        segFile = os.path.join(path_seg, seg_list[i])
        segmentations.append(segFile)
        #print(segmentations)

        imgFile = os.path.join(path_img, img_list[i])
        images.append(imgFile)
        #print(images)

#which transforms should be applied????, croping & flipping seems countereffective
#should transforms only be applied to train set?
img_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((128,128,128))
    ]
)

basic_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((128,128,128))
    ]
)

# zip data into one list of image/segmenttaion pairs
# (not exactly necessary but I get paranoid about accidentally messing up the sequences)
zipped_data = list(zip(images,segmentations))
#print(f"Zip length: {len(zipped_data)}")

#split into train, test & validation files
ratio_a = len(zipped_data)*0.8
train_files_a = [zipped_data[i] for i in range(int(ratio_a))]
test_images = [zipped_data[i][0] for i in range(int(ratio_a),len(zipped_data))]
test_segmentations = [zipped_data[i][1] for i in range(int(ratio_a),len(zipped_data))]

ratio_b = len(train_files_a)*0.8
train_images = [train_files_a[i][0] for i in range(int(ratio_b))]
train_segmentations = [train_files_a[i][1] for i in range(int(ratio_b))]
val_images = [train_files_a[i][0] for i in range(int(ratio_b),len(train_files_a))]
val_segmentations = [train_files_a[i][1] for i in range(int(ratio_b),len(train_files_a))]

'''
print(f"Train length: {len(train_images)}")
print(f"Test length: {len(test_images)}")
print(f"Val length: {len(val_images)}")
print(f"Train Files: {train_images}")
print(f"Test Files: {test_images}")
print(f"Val images: {val_images}")
print(f"Val segs: {val_segmentations}")
print(f"images: {images}")
print(f"segs: {segmentations}")
'''
#create datasets from the files with respective transforms
train_data = ArrayDataset(train_images, img_transforms,train_segmentations,basic_transforms)
test_data = ArrayDataset(test_images,basic_transforms,test_segmentations,basic_transforms)
val_data = ArrayDataset(val_images,basic_transforms,val_segmentations,basic_transforms)


#visualizing some slices from the training data to make sure everything is fine
im, seg = train_data[0]
num_slices = im.shape[1]
print(im.shape)

num_slices = im.shape[1]
num_rows = (num_slices // 10)
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))

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
fig.savefig("/lustre/groups/iterm/sebnem/slurm_outputs/slices.png")

'''
print(f"test_data, length: {len(test_data)}")
print(f"train_data, length: {len(train_data)}")
print(f"val_data, length: {len(val_data)}")
'''

#build Dataloader
train_loader = DataLoader(train_data, batch_size=1)
train_batch = first(train_loader)
print(train_batch[0].shape)
#print(torch.reshape(train_batch[0],(1,1,128,128,128)))

test_loader = DataLoader(test_data)
val_loader = DataLoader(val_data)
val_batch = first(val_loader)
#print(f"val_batch: {val_batch[1].shape}")

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
    act=torch.nn.functional.leaky_relu()
).to(device)

print("Checkpoint 2")

loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric()
metric_values = []

epoch_loss_values = []

max_epochs = 200
val_interval = 2
best_metric = -1

print("Checkpoint 3")
for epoch in range(1, max_epochs):
    print("-" * 10)
    print(f"epoch {epoch}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in tqdm(train_loader):
        step += 1
        img, seg = (
            batch_data[0].to(device),
            batch_data[1].to(device),
        )
        print(f"Input to model: {img.size()}")
        print(f"Segmentations: {seg.size()}")
        optimizer.zero_grad()
        outputs = model(img)
        print(f"Output from model: {outputs.shape}")
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
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_img, val_seg = (
                    val_data[0].to(device),
                    val_data[1].to(device)
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_img, roi_size, sw_batch_size, model)
                val_outputs = [i for i in decollate_batch(val_outputs)]
                val_seg = [i for i in decollate_batch(val_seg)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_seg)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
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

checkpoint = pd.DataFrame(
    {'train loss': epoch_loss_values,
     'dice values': metric_values,
     'best metric epoch': best_metric_epoch,
     'best metric': best_metric,
     'model_state_dict': model.state_dict(),
     'optimizer_state_dict': optimizer.state_dict(),
    }
)

val_interval = 2
plt.figure("train", (15, 5))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Dice Loss")
x = [i + 1 for i in range(len(checkpoint["train loss"]))]
y = checkpoint["train loss"]
plt.xlabel("#Epochs")
plt.ylabel("Dice Loss")
plt.plot(x, y)
plt.plot(checkpoint["best metric epoch"],
checkpoint["train loss"][checkpoint["best metric epoch"]], 'r*', markersize=8)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice Score")
x = [val_interval * (i + 1) for i in range(len(checkpoint["dice values"]))]
y = checkpoint["dice values"]
plt.xlabel("#Epochs")
plt.plot(x, y)
plt.plot(checkpoint["best metric epoch"],
checkpoint["dice values"][checkpoint["best metric epoch"]//2], 'r*', markersize=10)
plt.annotate("Best Score[470, 0.9516]", xy=(checkpoint["best metric epoch"],
checkpoint["dice values"][checkpoint["best metric epoch"]//2]))
plt.savefig("LearningCurves.png")
plt.show()


'''
STILL TO DO
    - Figure out which transforms to use
    - track metrics
    - 5-fold cross Validation
'''