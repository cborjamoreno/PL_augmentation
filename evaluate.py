import glob
import argparse
import os
import numpy as np
import torch
import torchmetrics
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--images_pth", help="path of expanded images or a single image", required=True)
parser.add_argument("-gt", "--ground_truth_pth", help="path of ground truth images", required=True)
parser.add_argument("-bg", "--background_class", help="background class label", default="34")
parser.add_argument("-n", "--num_classes", help="number of classes", type=int, default=35)
parser.add_argument("-m", "--metric_background", help="include background class in metrics", action="store_true")
args = parser.parse_args()

image_pth = args.images_pth
gt_pth = args.ground_truth_pth

# Ensure paths exist
if not os.path.exists(image_pth) or not os.path.exists(gt_pth):
    raise ValueError("One or more paths do not exist.")

NUM_CLASSES = args.num_classes
background_class = int(args.background_class)

if not args.metric_background:
    # Metric setup
    pa_metric = torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES, ignore_index=background_class)
    mpa_metric = torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES, ignore_index=background_class, average='macro')
    mpa_metric_per_class = torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES, ignore_index=background_class, average='none')
    iou_metric = torchmetrics.JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, ignore_index=background_class)
    iou_metric_per_class = torchmetrics.JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, ignore_index=background_class, average='none')
else:
    pa_metric = torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES)
    mpa_metric = torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES, average='macro')
    mpa_metric_per_class = torchmetrics.Accuracy(task='multiclass', num_classes=NUM_CLASSES, average='none')
    iou_metric = torchmetrics.JaccardIndex(task='multiclass', num_classes=NUM_CLASSES)
    iou_metric_per_class = torchmetrics.JaccardIndex(task='multiclass', num_classes=NUM_CLASSES, average='none')

# List of image files
if os.path.isdir(image_pth):
    image_files = glob.glob(os.path.join(image_pth, '*.*'))
else:
    unique_labels = np.unique(Image.open(os.path.join(gt_pth, os.path.splitext(os.path.basename(image_pth))[0] + ".png")).convert("L"))
    unique_labels = [int(i) for i in unique_labels]
    image_files = [image_pth]

# Define a function for processing an individual image
def process_image(filename):
    # Load expanded image and ground truth as grayscale
    pil_img = Image.open(filename).convert("L")
    gt_pil_img = Image.open(os.path.join(gt_pth, os.path.splitext(os.path.basename(filename))[0] + ".png")).convert("L")

    #count number of pixels of class 25 in ground truth and image
    gt_np = np.array(gt_pil_img)
    img_np = np.array(pil_img)

    # Convert ground truth to class indices (directly as grayscale values)
    gt_gray = gt_np.astype(int)

    # Convert numpy arrays to tensors
    img_torch = torch.tensor(img_np, dtype=torch.int)
    labels_torch = torch.tensor(gt_gray, dtype=torch.int)

    # Update metrics
    pa_metric.update(img_torch, labels_torch)
    mpa_metric.update(img_torch, labels_torch)
    mpa_metric_per_class.update(img_torch, labels_torch)
    iou_metric.update(img_torch, labels_torch)
    iou_metric_per_class.update(img_torch, labels_torch)

# Process all images sequentially
for filename in tqdm(image_files, desc="Evaluating images"):
    process_image(filename)

# Final computation of metrics
acc = pa_metric.compute()
m_acc = mpa_metric.compute()
m_acc_per_class = mpa_metric_per_class.compute()
miou = iou_metric.compute()
miou_per_class = iou_metric_per_class.compute()

# Print mIoU per class excluding background class and calculate sums
iou_sum = 0
acc_sum = 0
valid_classes = 0
    
# List of image files
if os.path.isdir(image_pth):
    for cls in range(NUM_CLASSES):
        if cls == background_class:
            continue
        class_iou = miou_per_class[cls].item()
        class_acc = m_acc_per_class[cls].item()
        print(f"Class {cls} mPA: {class_acc * 100:.2f}, mIoU: {class_iou * 100:.2f}")
        iou_sum += class_iou
        acc_sum += class_acc
        valid_classes += 1
else:
    for cls in range(NUM_CLASSES):
        if cls == background_class:
            continue
        if cls not in unique_labels:
            continue
        class_iou = miou_per_class[cls].item()
        class_acc = m_acc_per_class[cls].item()
        print(f"Class {cls} mPA: {class_acc * 100:.2f}, mIoU: {class_iou * 100:.2f}")
        iou_sum += class_iou
        acc_sum += class_acc
        valid_classes += 1

# Calculate and print averages
iou_avg = iou_sum / valid_classes
acc_avg = acc_sum / valid_classes

print("PA:", acc.item(), ", mPA:", acc_avg, ", mIOU:", iou_avg)
