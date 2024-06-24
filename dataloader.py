import logging
import os
import time

import pandas as pd
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from zipfile import ZipFile

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
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity, LoadImage, LoadImaged, \
    Resized, ToTensord, RandFlipd, AsDiscrete, Activations, Activationsd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#!!!!!!PATH NEEDS TO BE ADJUSTED FOR HPC !!!!!!!!!

path_seg = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/gt_new"
path_img = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/raw_new"
directory= "/lustre/groups/iterm/sebnem/"
'''
path_seg = "/Users/sebnemcam/Desktop/microglia/input cxc31/gt_new/"
path_img = "/Users/sebnemcam/Desktop/microglia/input cxc31/raw_new/"
directory = "/Users/sebnemcam/Desktop/Helmholtz/"
'''
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
#fig.savefig("/lustre/groups/iterm/sebnem/slices.png")

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
    up_kernel_size=3,
    strides=(2,2,2,2)
).to(device)

print("Checkpoint 2")

lr = 0.1
loss_function = DiceLoss(sigmoid=True)
dice_metric = torchmetrics.Dice(zero_division=1).to(device)
metric_values = []

epoch_loss_values = []
lr_values = []

max_epochs = 3000
val_interval = 1
best_metric = -1

print("Checkpoint 3")

fig, axs = plt.subplots(3, 1, figsize=(10, 15))


optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,threshold=0.00001, patience=50)
for epoch in range(0, max_epochs):
    print(f"TRAINING WITH LEARNING RATE {lr}")
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in test_loader:
        step += 1
        img, seg = (
            batch_data['image'].to(device),
            batch_data['segmentation'].to(device),
        )
        # print(f"Input to model: {img.size()}")
        # print(f"Segmentations: {seg.size()}")
        optimizer.zero_grad()
        outputs = model(img)
        # print(f"Output from model: {outputs.shape}")
        loss = loss_function(outputs, seg)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        epoch_loss += loss.item()
        lr_values.append(scheduler.get_last_lr())
        print(
            f"{step}/{len(train_data) // train_loader.batch_size}, "
            f"train loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if epoch % val_interval == 0:
        print("Validation")
        model.eval()
        batch_idx = 0
        with torch.no_grad():
            zip_filename = os.path.join(directory, f"val_outputs_epoch{epoch + 1}.zip")
            with ZipFile(zip_filename, 'w') as zipf:
                for val_data in val_loader:
                    batch_idx += 1
                    val_img, val_seg = (
                        val_data['image'].to(device),
                        val_data['segmentation'].to(device)
                    )
                    val_seg = val_seg.type(torch.short)
                    val_outputs = model(val_img)
                    val_outputs[val_outputs < 0.5] = 0
                    val_outputs[val_outputs >= 0.5] = 1
                    dice_metric(preds=val_outputs, target=val_seg)
                    val_outputs_np = val_outputs.cpu().numpy()
                    val_seg_np = val_seg.cpu().numpy()
                    val_img_np = val_img.cpu().numpy()

                    for i in range(val_outputs_np.shape[0]):
                        # Extract the ith sample, first channel, all depth, height, and width slices
                        output_image = val_outputs_np[i, 0, :, :, :]
                        seg_image = val_seg_np[i, 0, :, :, :]
                        raw_image = val_img_np[i, 0, :, :, :]

                        # Create NIfTI images
                        output_nifti = nib.Nifti1Image(output_image, np.eye(4))
                        seg_nifti = nib.Nifti1Image(seg_image, np.eye(4))
                        raw_nifti = nib.Nifti1Image(raw_image, np.eye(4))

                        #define file names
                        output_filename = f"zip_files/zip_files/output_epoch{epoch + 1}_batch{batch_idx}_image{i}.nii.gz"
                        seg_filename = f"zip_files/zip_files/seg_epoch{epoch + 1}_batch{batch_idx}_image{i}.nii.gz"
                        raw_filename = f"zip_files/zip_files/raw_epoch{epoch + 1}_batch{batch_idx}_image{i}.nii.gz"

                        #save NIFTI images to temporary files
                        output_filepath = os.path.join(directory, output_filename)
                        seg_filepath = os.path.join(directory, seg_filename)
                        raw_filepath = os.path.join(directory, raw_filename)
                        nib.save(output_nifti, output_filepath)
                        nib.save(seg_nifti, seg_filepath)
                        nib.save(raw_nifti, raw_filepath)

                        #add files to zip file
                        zipf.write(output_filepath, arcname=output_filename)
                        zipf.write(seg_filepath, arcname=seg_filename)
                        zipf.write(raw_filepath, arcname=raw_filename)

                        # Remove temporary files
                        os.remove(output_filepath)
                        os.remove(seg_filepath)
                        os.remove(raw_filepath)

                        print(f"Saved pred, seg & raw")

            # aggregate the final mean dice result
            metric = dice_metric.compute().item()
            # reset the status for next validation round
            dice_metric.reset()
            metric_values.append(metric)
            print(f"Dice Value: {metric}")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                #torch.save(model.state_dict(), os.path.join(
                    #directory, "best_metric_model.pth"))
                #print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    # Plotting the loss and dice scores
axs[0].plot(range(1, max_epochs + 1), epoch_loss_values)
axs[1].plot(range(1, max_epochs + 1), metric_values)
axs[2].plot(range(1, len(lr_values) + 1), lr_values)


axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Dice Loss for Different Learning Rates')
axs[0].set_ylim(0,1)

axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Dice Score')
axs[1].set_title('Dice Score for Different Learning Rates')
axs[1].set_ylim(0,1)

axs[2].set_xlabel('Epochs')
axs[2].set_ylabel('Learning Rate')
axs[2].set_title('Learning Rate Schedule')
axs[2].set_ylim(0,lr+0.005)

plt.show()
#plt.savefig("/lustre/groups/iterm/sebnem/LearningCurves_no_sig.png")
#plt.savefig("/Users/sebnemcam/Desktop/LearningCurves.png")




'''
STILL TO DO
    - 5-fold cross Validation
'''
