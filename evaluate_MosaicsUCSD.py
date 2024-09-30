import glob
import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to calculate PA, mPA, and mIoU
def calculate_segmentation_metrics(pred, label, num_classes):
    """
    Calculate Pixel Accuracy (PA), Mean Pixel Accuracy (mPA), and Mean IoU (mIoU)
    Args:
        pred: numpy array of predicted segmentation mask
        label: numpy array of ground truth mask
        num_classes: number of classes in the segmentation
    Returns:
        PA, mPA, mIoU
    """
    hist = np.zeros((num_classes, num_classes))
    for p, l in zip(pred.flatten(), label.flatten()):
        if l < num_classes and p < num_classes:  # Only count valid classes
            hist[l, p] += 1

    # Pixel Accuracy (PA)
    PA = np.diag(hist).sum() / hist.sum()

    # Mean Pixel Accuracy (mPA)
    class_acc = np.diag(hist) / hist.sum(axis=1)
    mPA = np.nanmean(class_acc)

    # Mean Intersection over Union (mIoU)
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)

    return PA, mPA, mIoU

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--images_pth", help="path of expanded images", required=True)
parser.add_argument("-gt", "--ground_truth_pth", help="path of ground truth images", required=True)
args = parser.parse_args()

image_pth = args.images_pth
gt_pth = args.ground_truth_pth

if not os.path.exists(image_pth):
    print("Image path does not exist")
    exit()

if not os.path.exists(gt_pth):
    print("Ground truth path does not exist")
    exit()

# Define the number of classes (including background)
NUM_CLASSES = 34  # Adjust if needed, assuming 34 classes + background (34)

def process_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    gt_base_name = base_name + ".png"
    gt_filename = os.path.join(gt_pth, gt_base_name)
    gt = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)

    if gt is None:
        print(f"Ground truth not found for image {filename}")
        return None

    if img.shape != gt.shape:
        print(f"Shape mismatch: {img.shape} != {gt.shape}")
        return None

    # Calculate the metrics for this image
    pa, mpa, miou = calculate_segmentation_metrics(img, gt, NUM_CLASSES)
    
    del img
    del gt
    return pa, mpa, miou

# Processing the images and aggregating the metrics
image_files = glob.glob(image_pth + '/*.*')

total_pa = 0
total_mpa = 0
total_miou = 0
total_images = 0

with tqdm(total=len(image_files), desc="Evaluating images") as pbar:
    for filename in image_files:
        result = process_image(filename)
        if result is not None:
            pa, mpa, miou = result
            total_pa += pa
            total_mpa += mpa
            total_miou += miou
            total_images += 1

        pbar.update(1)

# Calculate averages across all images
mean_pa = total_pa / total_images if total_images > 0 else 0
mean_mpa = total_mpa / total_images if total_images > 0 else 0
mean_miou = total_miou / total_images if total_images > 0 else 0

print(f"\nMean Pixel Accuracy (PA): {mean_pa}")
print(f"Mean Pixel Accuracy per class (MPA): {mean_mpa}")
print(f"Mean Intersection over Union (MIoU): {mean_miou}")
