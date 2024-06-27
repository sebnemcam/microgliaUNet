import logging
import os
import time

import pandas as pd
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from zipfile import ZipFile

import numpy as np
from sklearn.model_selection import KFold
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

'''
ratio_b = len(train_files_a)*0.8
train_files = [train_files_a[i] for i in range(int(ratio_b))]
val_files = [train_files_a[i] for i in range(int(ratio_b),len(train_files_a))]

train_data = [{'image' : img, 'segmentation' : seg } for img,seg in train_files]
val_data = [{'image' : img, 'segmentation' : seg } for img,seg in val_files]
'''
kfold = KFold(5,True,1)
test_data = [{'image' : img, 'segmentation' : seg } for img,seg in test_files]
data = [{'image' : img, 'segmentation' : seg } for img,seg in train_files_a]

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

test_set = CacheDataset(test_files,dic_transforms)

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

lr = 0.1
loss_function = DiceLoss(sigmoid=True)
dice_metric = torchmetrics.Dice(zero_division=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,threshold=0.00001, patience=50)

metric_values = []
epoch_loss_values = []
lr_values = []
best_metric = []
max_epochs = 3000
val_interval = 1
fold = -1

fig, axs = plt.subplots(5, 3, figsize=(15, 10))

for train_data, val_data in kfold.split(data['image'],data['segmentation']):

    train_set = CacheDataset(train_data,dic_transforms_train)
    val_set = CacheDataset(val_data,dic_transforms_train)

    train_loader = DataLoader(train_set,batch_size=5)
    test_loader = DataLoader(test_set,batch_size=5)
    val_loader = DataLoader(val_set,batch_size=5)

    fold += 1

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
            lr_values[fold].append(scheduler.get_last_lr())
            print(
                f"{step}/{len(train_data) // train_loader.batch_size}, "
                f"train loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values[fold].append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if epoch % val_interval == 0:
            print("Validation")
            model.eval()
            batch_idx = 0
            with torch.no_grad():
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
                    #val_outputs_np = val_outputs.cpu().numpy()
                    #val_seg_np = val_seg.cpu().numpy()
                    #val_img_np = val_img.cpu().numpy()

                    '''
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
                        output_filename = f"outputs/output_epoch{epoch + 1}_batch{batch_idx}_image{i}.nii.gz"
                        seg_filename = f"gts/seg_epoch{epoch + 1}_batch{batch_idx}_image{i}.nii.gz"
                        raw_filename = f"raws/raw_epoch{epoch + 1}_batch{batch_idx}_image{i}.nii.gz"

                        #save NIFTI images to temporary files
                        output_filepath = os.path.join(directory, output_filename)
                        seg_filepath = os.path.join(directory, seg_filename)
                        raw_filepath = os.path.join(directory, raw_filename)
                        nib.save(output_nifti, output_filepath)
                        nib.save(seg_nifti, seg_filepath)
                        nib.save(raw_nifti, raw_filepath)
                       # print(f"Saved pred, seg & raw")
                       '''

                # aggregate the final mean dice result
                metric = dice_metric.compute().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values[fold].append(metric)
                print(f"Dice Value: {metric}")

                if metric > best_metric[fold]:
                    best_metric[fold] = metric


    # Plotting the loss and dice scores
    colors = plt.cm.viridis(np.linspace(0, 1, 5))

    axs[0].plot(range(1, max_epochs + 1), epoch_loss_values[fold], label=f'Fold {fold+1}', color=colors[fold])
    axs[1].plot(range(1, max_epochs + 1), metric_values[fold], label=f'Fold {fold+1}', color=colors[fold])
    axs[2].plot(range(1, len(lr_values) + 1), lr_values[fold], label=f'Fold {fold+1}', color=colors[fold])

    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Dice Loss for Different Folds')
    axs[0].set_ylim(0,1)

    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Dice Score')
    axs[1].set_title('Dice Score for Different Folds')
    axs[1].set_ylim(0,1)

    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Learning Rate')
    axs[2].set_title('Learning Rate Schedule for Different Folds')
    axs[2].set_ylim(0,lr+0.005)

    for ax in axs:
        ax.legend()


plt.show()
plt.savefig("/lustre/groups/iterm/sebnem/LearningCurves.png")

df = {'Fold' : [0,1,2,3,4], 'Best Dice': best_metric}
#df.to_csv(os.path.join(directory,'folds'))
#plt.savefig("/Users/sebnemcam/Desktop/LearningCurves.png")
