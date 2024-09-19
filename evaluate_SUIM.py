import glob
import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm

# Define a mapping from RGB values to class labels
class_mapping = {
    (64, 0, 64): 0,  # Background
    (0, 0, 0): 1,
    (0, 0, 255): 2,
    (0, 255, 0): 3,
    (0, 255, 255): 4,
    (255, 0, 0): 5,
    (255, 0, 255): 6,
    (255, 255, 0): 7,
    (255, 255, 255): 8
}

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

def calculate_metrics(img, gt):
    # Exclude background color (64, 0, 64) from calculations
    background_color = [64, 0, 64]
    valid_pixels_mask = ~(np.all(gt == background_color, axis=-1))

    # Apply mask to exclude background
    img_valid = img[valid_pixels_mask]
    gt_valid = gt[valid_pixels_mask]

    # Map RGB values to class labels
    gt_labels = np.zeros(gt_valid.shape[:2], dtype=np.uint8)
    img_labels = np.zeros(img_valid.shape[:2], dtype=np.uint8)
    for rgb, label in class_mapping.items():
        gt_labels[np.all(gt_valid == rgb, axis=-1)] = label
        img_labels[np.all(img_valid == rgb, axis=-1)] = label

    # Calculate per-class IoU and PA
    unique_classes = np.unique(gt_labels)
    if 0 in unique_classes:  # Skip background class (label 0)
        unique_classes = unique_classes[unique_classes != 0]

    class_iou = {}
    class_pa = []

    for cls in unique_classes:
        gt_cls = gt_labels == cls
        img_cls = img_labels == cls

        pa_cls = np.sum(gt_cls & img_cls) / np.sum(gt_cls) if np.sum(gt_cls) != 0 else 0
        class_pa.append(pa_cls)

        intersection = np.sum(gt_cls & img_cls)
        union = np.sum(gt_cls | img_cls)

        iou = intersection / union if union != 0 else 0
        class_iou[cls] = iou

    mean_pa = np.mean(class_pa) if class_pa else 0
    return mean_pa, class_iou, class_pa

def process_image(filename):
    img = cv2.imread(filename)
    base_name = os.path.splitext(os.path.basename(filename))[0]
    gt_base_name = base_name + ".bmp"
    gt_filename = os.path.join(gt_pth, gt_base_name)
    gt = cv2.imread(gt_filename)

    if gt is None:
        print(f"Ground truth not found for image {filename}")
        return None

    if img.shape != gt.shape:
        print(img.shape, gt.shape)
        return None

    metrics = calculate_metrics(img, gt)
    del img
    del gt
    return metrics

image_files = glob.glob(image_pth + '/*.*')

total_pa = 0
total_images = 0
class_iou_aggregate = {label: [] for label in class_mapping.values() if label != 0}
class_pa_aggregate = {label: [] for label in class_mapping.values() if label != 0}

with tqdm(total=len(image_files), desc="Evaluating images") as pbar:
    for filename in image_files:
        result = process_image(filename)
        if result is not None:
            pa, class_iou, class_pa = result
            total_pa += pa
            total_images += 1

            for cls, iou in class_iou.items():
                class_iou_aggregate[cls].append(iou)

            for cls, pa_cls in zip(class_iou.keys(), class_pa):
                class_pa_aggregate[cls].append(pa_cls)

        pbar.update(1)

mean_pa = total_pa / total_images if total_images > 0 else 0

# Calculate mean IoU per class
mean_class_iou = {cls: np.mean(iou_list) for cls, iou_list in class_iou_aggregate.items() if iou_list}

# Calculate overall mean IoU
mean_miou = np.mean(list(mean_class_iou.values())) if mean_class_iou else 0

# Calculate mean PA per class
mean_class_pa = {cls: np.mean(pa_list) for cls, pa_list in class_pa_aggregate.items() if pa_list}

# Calculate overall mean PA (MPA)
mean_mpa = np.mean(list(mean_class_pa.values())) if mean_class_pa else 0

print(f"Mean Pixel Accuracy (PA): {mean_pa}")
print(f"Mean Pixel Accuracy per class (MPA): {mean_mpa}")
print(f"Mean Intersection over Union (MIoU): {mean_miou}")

# Print mean IoU and PA for each class
for cls, miou in mean_class_iou.items():
    print(f"{cls}: {miou:.3f}")