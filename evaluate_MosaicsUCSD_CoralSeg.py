import glob
import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--images_pth", help="path of expanded images", required=True)
parser.add_argument("-gt", "--ground_truth_pth", help="path of ground truth images", required=True)
parser.add_argument('--classes', type=int, required=True, help="Number of classes")
args = parser.parse_args()

image_pth = args.images_pth
gt_pth = args.ground_truth_pth
N_CLASSES = args.classes

if not os.path.exists(image_pth):
    print("Image path does not exist")
    exit()

if not os.path.exists(gt_pth):
    print("Ground truth path does not exist")
    exit()

def calculate_metrics(img, gt):
    # Initialize counters
    correct_pixels = np.zeros(N_CLASSES)
    total_gt_pixels = np.zeros(N_CLASSES)
    total_pred_pixels = np.zeros(N_CLASSES)
    intersection = np.zeros(N_CLASSES)
    union = np.zeros(N_CLASSES)

    for cls in range(N_CLASSES):
        correct_pixels[cls] = np.sum((gt == cls) & (img == cls))
        total_gt_pixels[cls] = np.sum(gt == cls)
        total_pred_pixels[cls] = np.sum(img == cls)
        intersection[cls] = np.sum((gt == cls) & (img == cls))
        union[cls] = np.sum((gt == cls) | (img == cls))

    # Calculate overall PA for the image
    pa = np.sum(correct_pixels) / np.sum(total_gt_pixels)

    # Calculate per-class PA
    class_pa = {cls: correct_pixels[cls] / total_gt_pixels[cls] if total_gt_pixels[cls] > 0 else 0 for cls in range(N_CLASSES)}

    # Calculate per-class IoU
    class_iou = {cls: intersection[cls] / union[cls] if union[cls] > 0 else 0 for cls in range(N_CLASSES)}

    return pa, class_pa, class_iou

def process_image(filename, pixel_count_per_class):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    gt_base_name = base_name + ".png"
    gt_filename = os.path.join(gt_pth, gt_base_name)
    gt = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)

    if gt is None:
        print(f"Ground truth not found for image {filename}")
        return None

    if img.shape != gt.shape:
        print(img.shape, gt.shape)
        return None

    # Count pixels per class for the current image
    unique, counts = np.unique(gt, return_counts=True)
    for cls, count in zip(unique, counts):
        if cls not in pixel_count_per_class:
            pixel_count_per_class[cls] = 0
        pixel_count_per_class[cls] += count

    metrics = calculate_metrics(img, gt)
    del img
    del gt
    return metrics

image_files = glob.glob(image_pth + '/*.*')

total_pa = 0
total_images = 0
class_iou_aggregate = {}
class_pa_aggregate = {}

# Dictionary to store the number of pixels per class
pixel_count_per_class = {}

with tqdm(total=len(image_files), desc="Evaluating images") as pbar:
    for filename in image_files:
        result = process_image(filename, pixel_count_per_class)
        if result is not None:
            pa, class_pa, class_iou = result
            total_pa += pa
            total_images += 1

            for cls, pa_cls in class_pa.items():
                if cls not in class_pa_aggregate:
                    class_pa_aggregate[cls] = []
                class_pa_aggregate[cls].append(pa_cls)

            for cls, iou in class_iou.items():
                if cls not in class_iou_aggregate:
                    class_iou_aggregate[cls] = []
                class_iou_aggregate[cls].append(iou)

        pbar.update(1)

# Calculate mean PA across all images
mean_pa = total_pa / total_images if total_images > 0 else 0

# Calculate mean PA per class
mean_class_pa = {cls: np.mean(pa_list) for cls, pa_list in class_pa_aggregate.items() if pa_list}

# Calculate overall mean PA (MPA)
mean_mpa = np.mean(list(mean_class_pa.values())) if mean_class_pa else 0

# Calculate mean IoU per class
mean_class_iou = {cls: np.mean(iou_list) for cls, iou_list in class_iou_aggregate.items() if iou_list}

# Calculate overall mean IoU
mean_miou = np.mean(list(mean_class_iou.values())) if mean_class_iou else 0

print(f"\nPixel Accuracy (PA): {mean_pa}")
print(f"Mean Pixel Accuracy (MPA): {mean_mpa}")
print(f"Mean Intersection over Union (MIoU): {mean_miou}")

# Sort the items by cls
sorted_mean_class_iou = sorted(mean_class_iou.items())

# Print the sorted IoU per class
print("\nMean IoU per class:")
for cls, miou in sorted_mean_class_iou:
    print(f"Class {cls}: {miou:.3f}")