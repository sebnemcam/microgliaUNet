# microgliaUNet
# 3D Segmentation of Microglia Cells

This project utilizes the 3D-UNet for segmentation of microglia somata in mouse-brain scans.
The further objective is to succesfully segment the processes of the cells.

## model_test_train.py
- Loads the train, test and validation sets
- Trains the model
- Performs 5-Fold cross validation
- Savesg outputs of each fold in specified directory

## present_scores.py 
- Prerequisite: run blob_comparison on the outputs of all 5 folds and their GT
- Calculates summary statistics of the calculated dice scores
- Saves a summary table
- Visualizes dice scores in box plots

## resize.py
- resized the files we already have to match the model output

# Outlook
Objective of process segmentation could possibly be achieved by performing flood filling on images with binary dilated somata, followed by erosion and recsontruction of the unfilled parts of the processes
