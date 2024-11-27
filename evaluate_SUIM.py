import glob
import cv2
from PIL import Image
import argparse
import os
import numpy as np
import torch
import torchmetrics
from tqdm import tqdm

# Define a mapping from RGB values to class labels
class_mapping = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,
    (0, 255, 0): 2,
    (0, 255, 255): 3,
    (255, 0, 0): 4,
    (255, 0, 255): 5,
    (255, 255, 0): 6,
    (255, 255, 255): 7
}

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--images_pth", help="path of expanded images", required=True)
parser.add_argument("-gt", "--ground_truth_pth", help="path of ground truth images", required=True)
parser.add_argument("-m", "--metric_background", help="include background class in metrics", action="store_true")

args = parser.parse_args()

image_pth = args.images_pth
gt_pth = args.ground_truth_pth

if not os.path.exists(image_pth):
    print("Image path does not exist")
    exit()

if not os.path.exists(gt_pth):
    print("Ground truth path does not exist")
    exit()

# Metric setup
NUM_CLASSES = len(class_mapping)  # Total number of classes, including background
background_class = 0  # The background class

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
image_files = glob.glob(image_pth + '/*.*')

def calculate_metrics(img, gt):


    # Map RGB values to class labels
    gt_labels = np.zeros(gt.shape[:2], dtype=np.uint8)
    img_labels = np.zeros(img.shape[:2], dtype=np.uint8)

    
    for rgb, label in class_mapping.items():
        gt_labels[np.all(gt == rgb, axis=-1)] = label
        img_labels[np.all(img == rgb, axis=-1)] = label

    # Convert numpy arrays to tensors for metric calculation
    img_labels_torch = torch.tensor(img_labels, dtype=torch.int)
    gt_labels_torch = torch.tensor(gt_labels, dtype=torch.int)

    # Update metrics
    pa_metric.update(img_labels_torch, gt_labels_torch)
    mpa_metric.update(img_labels_torch, gt_labels_torch)
    iou_metric.update(img_labels_torch, gt_labels_torch)
    iou_metric_per_class.update(img_labels_torch, gt_labels_torch)
    mpa_metric_per_class.update(img_labels_torch, gt_labels_torch)

def process_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    gt_base_name = base_name + ".png"
    gt_filename = os.path.join(gt_pth, gt_base_name)
    gt = cv2.imread(gt_filename)
    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

    if gt is None:
        print(f"Ground truth not found for image {filename}")
        return None

    if img.shape != gt.shape:
        print(img.shape, gt.shape)
        return None

    calculate_metrics(img, gt)

# List of image files
if os.path.isdir(image_pth):
    image_files = glob.glob(os.path.join(image_pth, '*.*'))
else:
    unique_labels = np.unique(np.array(Image.open(os.path.join(gt_pth, os.path.splitext(os.path.basename(image_pth))[0] + ".png")).convert("RGB")).reshape(-1, 3), axis=0)
    unique_labels = [class_mapping[tuple(label)] for label in unique_labels if isinstance(label, np.ndarray) and tuple(label) in class_mapping]
    unique_labels = list(unique_labels)
    image_files = [image_pth]

# Process all images sequentially
for filename in tqdm(image_files, desc="Evaluating images"):
    process_image(filename)

# Final computation of metrics
acc = pa_metric.compute()
m_acc = mpa_metric.compute()
m_acc_per_class = mpa_metric_per_class.compute()
miou = iou_metric.compute()
miou_per_class = iou_metric_per_class.compute()

# print(f"Per-class mPA (len {len(m_acc_per_class)}):", m_acc_per_class)

# print(f"Per-class mIoU (len {len(miou_per_class)}):", miou_per_class)

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