import logging
import os
import time

import pandas as pd

import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torchmetrics

from monai.data import ArrayDataset, DataLoader, CacheDataset
from monai.networks.nets import Unet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, LoadImaged, Resized, ToTensord, RandFlipd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#!!!!!!PATH NEEDS TO BE ADJUSTED FOR HPC !!!!!!!!!

path_seg = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/gt_new"
path_img = "/lustre/groups/iterm/Annotated_Datasets/Annotated Datasets/Microglia - Microglia LSM and Confocal/input cxc31/raw_new"
directory= "/lustre/groups/iterm/sebnem/runs/shuffled/09.07_22:33/"
'''
path_seg = "/Users/sebnemcam/Desktop/microglia/input cxc31/gt_new/"
path_img = "/Users/sebnemcam/Desktop/microglia/input cxc31/raw_new/"
directory = "/Users/sebnemcam/Desktop/Helmholtz/"
'''

seg_list = os.listdir(path_seg)
img_list = os.listdir(path_img)
segmentations = []
images = []

data = []

print("I am inside the python file")

for i in range(len(seg_list)):
    if "nii.gz" in seg_list[i] and img_list[i] == seg_list[i]:

        segFile = os.path.join(path_seg, seg_list[i])
        segmentations.append(segFile)

        imgFile = os.path.join(path_img, img_list[i])
        images.append(imgFile)
        data.append([{'image' : imgFile, 'segmentation' : segFile, 'name': img_list[i]}])

# zip data into one list of image/segmenttaion pairs
#zipped_data = list(zip(images,segmentations))
#print(f"Zip length: {len(zipped_data)}")

#split into train, test & validation files
#ratio_a = len(zipped_data)*0.8
#train_files_a = [zipped_data[i] for i in range(int(ratio_a))]
#test_files = [zipped_data[i] for i in range(int(ratio_a),len(zipped_data))]

'''
ratio_b = len(train_files_a)*0.8
train_files = [train_files_a[i] for i in range(int(ratio_b))]
val_files = [train_files_a[i] for i in range(int(ratio_b),len(train_files_a))]

train_data = [{'image' : img, 'segmentation' : seg } for img,seg in train_files]
val_data = [{'image' : img, 'segmentation' : seg } for img,seg in val_files]
'''
kfold = KFold(n_splits=5,shuffle=True,random_state=1)
#data = [{'image' : img, 'segmentation' : seg} for img,seg in zipped_data]
#print(f"images: {data[0]}")

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
        LoadImaged(keys=keys, ensure_channel_first=True, image_only=True),
        Resized(keys=keys,spatial_size=(128, 128, 128)),
        ToTensord(keys)
    ]
)

#test_set = CacheDataset(test_files,dic_transforms)

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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,mode='max')

max_epochs = 1500
val_interval = 1
test_fold = 0

fig, axs = plt.subplots(5, 3, figsize=(15, 10))

for i, (train_val_idx, test_idx) in enumerate(kfold.split(data)):

    test_fold += 1
    print(f"TEST FOLD {test_fold}")

    train_val_data = [data[i] for i in train_val_idx]

    test_data = [data[i] for i in test_idx]
    print(len(test_data))
    test_set = CacheDataset(test_data, dic_transforms)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    all_lr_values = []
    all_metric_values = []
    all_epoch_loss_values = []
    fold = 0
    test_dice_values = []
    test_names = []

    for i, (train_idx, val_idx) in enumerate(kfold.split(train_val_data)):

        fold += 1
        print(f"TRAIN FOLD {fold}")

        train_data = [train_val_data[i] for i in train_idx]
        val_data = [train_val_data[i] for i in val_idx]

        train_set = CacheDataset(train_data,dic_transforms_train)
        val_set = CacheDataset(val_data,dic_transforms_train)

        train_loader = DataLoader(train_set,batch_size=5)
        test_loader = DataLoader(test_set,batch_size=5)
        val_loader = DataLoader(val_set,batch_size=5)

        best_metric_epoch = -1

        lr_values = []
        metric_values = []
        epoch_loss_values = []
        best_metric = -1

        for epoch in range(0, max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs} in train fold {fold}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                img, seg, name = (
                    batch_data['image'].to(device),
                    batch_data['segmentation'].to(device),
                    batch_data['name']
                )
                # print(f"Input to model: {img.size()}")
                # print(f"Segmentations: {seg.size()}")
                optimizer.zero_grad()
                outputs = model(img)
                # print(f"Output from model: {outputs.shape}")
                loss = loss_function(outputs, seg)
                loss.backward()
                optimizer.step()
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
                    for val_data in val_loader:
                        batch_idx += 1
                        val_img, val_seg, name = (
                            val_data['image'].to(device),
                            val_data['segmentation'].to(device),
                            val_data['name']
                        )
                        val_seg = val_seg.type(torch.short)
                        val_outputs = model(val_img)
                        val_outputs[val_outputs < 0.5] = 0
                        val_outputs[val_outputs >= 0.5] = 1
                        dice_metric(preds=val_outputs, target=val_seg)
                        #val_outputs_np = val_outputs.cpu().numpy()

                        '''
                        for i in range(val_outputs_np.shape[0]):
                            # Extract the ith sample, first channel, all depth, height, and width slices
                            output_image = val_outputs_np[i, 0, :, :, :]
    
                            # Create NIfTI images
                            output_nifti = nib.Nifti1Image(output_image, np.eye(4))
    
                            #define file names
                            output_filename = f"outputs/{name}_epoch{epoch + 1}_"
    
                            #save NIFTI images to temporary files
                            output_filepath = os.path.join(directory, output_filename)
                            nib.save(output_nifti, output_filepath)
                           # print(f"Saved")
                           '''

                    # aggregate the final mean dice result
                    metric = dice_metric.compute().item()
                    scheduler.step(metric)
                    # reset the status for next validation round
                    dice_metric.reset()
                    metric_values.append(metric)
                    #metric_values[test_fold][fold][epoch]=metric
                    print(f"Dice Value: {metric}")

                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch= epoch +1
                        print(f"Train Fold {fold} \nBest Dice {best_metric} \nat epoch {best_metric_epoch}")
                        model_path = os.path.join(directory, f"test_fold{test_fold}/best_metric_model.pth")
                        torch.save(model.state_dict(),model_path)

        all_lr_values.append(lr_values)
        all_metric_values.append(metric_values)
        all_epoch_loss_values.append(epoch_loss_values)

        # Plotting the loss and dice scores
    colors = plt.cm.viridis(np.linspace(0, 1, 5))

    for fold in range(5):
        axs[test_fold-1,0].plot(range(1, max_epochs + 1), all_epoch_loss_values[fold], label=f'Fold {fold}', color=colors[fold])
        axs[test_fold-1,1].plot(range(1, max_epochs + 1), all_metric_values[fold], label=f'Fold {fold}', color=colors[fold])
        axs[test_fold-1,2].plot(range(1, len(lr_values) + 1), all_lr_values[fold], label=f'Fold {fold}', color=colors[fold])

    axs[test_fold-1,0].set_xlabel('Epochs')
    axs[test_fold-1,0].set_ylabel('Loss')
    axs[test_fold-1,0].set_title('Dice Loss for Different Folds')
    axs[test_fold-1,0].set_ylim(0,1)
    axs[test_fold-1, 0].legend()

    axs[test_fold-1,1].set_xlabel('Epochs')
    axs[test_fold-1,1].set_ylabel('Dice Score')
    axs[test_fold-1,1].set_title('Dice Score for Different Folds')
    axs[test_fold-1,1].set_ylim(0,1)
    axs[test_fold-1, 1].legend()

    axs[test_fold-1,2].set_xlabel('Epochs')
    axs[test_fold-1,2].set_ylabel('Learning Rate')
    axs[test_fold-1,2].set_title('Learning Rate Schedule for Different Folds')
    axs[test_fold-1,2].set_ylim(0,lr+0.005)
    axs[test_fold-1, 2].legend()

    plt.suptitle(f"Test Fold {test_fold}")


    model = Unet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    up_kernel_size=3,
    strides=(2,2,2,2)
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            img,seg,names = (
                test_data['image'].to(device),
                test_data['segmentation'].to(device),
                test_data['name']
            )
            test_names.append(names)
            seg = seg.type(torch.short)
            outputs = model(img)
            outputs[outputs < 0.5] = 0
            outputs[outputs >= 0.5] = 1
            print(f"outputs {names} has shape {outputs.shape}")
            print(f"target has shape {seg.shape}")
            dice_metric(preds=outputs, target=seg)
            metric = dice_metric.compute().item()
            dice_metric.reset()
            test_dice_values.append([metric,metric,metric,metric,metric])
            print(f"Test Dice Value: {metric}")

            outputs_np = outputs.cpu().numpy()

            for i in range(outputs_np.shape[0]):
                # Extract the ith sample, first channel, all depth, height, and width slices
                output_image = outputs_np[i, 0, :, :, :]

                # Create NIfTI images
                output_nifti = nib.Nifti1Image(output_image, np.eye(4))

                # define file names
                output_filename = f"test_fold{test_fold}/outputs/{names[i]}"

                # save NIFTI images to temporary files
                output_filepath = os.path.join(directory, output_filename)
                nib.save(output_nifti, output_filepath)
                print(f"Saved {names[i]}")

    plt.savefig(f"/lustre/groups/iterm/sebnem/runs/shuffled/09.07_22:33/test_fold{test_fold}/LearningCurvesTestFold{test_fold}.png")
    df = {'filename': test_names,
          'dice score': test_dice_values}
    df.to_csv(os.path.join(directory,f'test_fold{test_fold}/batch_dice_scores.csv'))
plt.savefig(f"/lustre/groups/iterm/sebnem/runs/shuffled/09.07_22:33/LearningCurves.png")
