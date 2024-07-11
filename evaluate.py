import glob
import cv2
import argparse
import random
import os
import math
import numpy as np
import pandas as pd

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

# Calculate MIoU and MPA in all images
miou_total = 0
mpa_total = 0  # Total for Mean Pixel Accuracy
total_images = 0

for filename in glob.glob(image_pth + '/*.*'):
    img = cv2.imread(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    gt_base_name = base_name.replace("_labels_rgb", "") + ".bmp"
    gt_filename = os.path.join(gt_pth, gt_base_name)
    gt = cv2.imread(gt_filename)

    # Ensure images are the same dimensions
    assert img.shape == gt.shape, "Image and GT dimensions do not match"

    # Calculate intersection: pixels where color matches exactly
    intersection = np.sum(np.all(gt == img, axis=-1))

    # Calculate the union correctly
    # Pixels that are positive in either the ground truth or the prediction
    gt_positive = np.any(gt != [0, 0, 0], axis=-1)  # Assuming black [0, 0, 0] is the background
    img_positive = np.any(img != [0, 0, 0], axis=-1)
    combined_positive = gt_positive | img_positive  # Combined foreground
    union = np.sum(combined_positive)

    # Calculate IoU for the current image
    iou = intersection / union if union != 0 else 0

    # Calculate Pixel Accuracy for the current image
    total_pixels = img.shape[0] * img.shape[1]
    pa = intersection / total_pixels

    # Accumulate MIoU, MPA, and count
    miou_total += iou
    mpa_total += pa
    total_images += 1

# Calculate mean MIoU and MPA across all images
mean_miou = miou_total / total_images if total_images > 0 else 0
mean_mpa = mpa_total / total_images if total_images > 0 else 0

# print(f"Mean MIoU: {mean_miou}")
print(f"Mean Pixel Accuracy: {mean_mpa}")
