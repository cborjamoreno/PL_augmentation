import glob
import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm

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
    # Exclude background class (assuming background class is 0)
    valid_pixels_mask = gt != 0

    # Apply mask to exclude background
    img_valid = img[valid_pixels_mask]
    gt_valid = gt[valid_pixels_mask]

    # Calculate per-class IoU and PA
    unique_classes = np.unique(gt_valid)

    class_iou = {}
    class_pa = []

    for cls in unique_classes:
        gt_cls = gt_valid == cls
        img_cls = img_valid == cls

        pa_cls = np.sum(gt_cls & img_cls) / np.sum(gt_cls) if np.sum(gt_cls) != 0 else 0
        class_pa.append(pa_cls)

        intersection = np.sum(gt_cls & img_cls)
        union = np.sum(gt_cls | img_cls)

        iou = intersection / union if union != 0 else 0
        class_iou[cls] = iou

    mean_pa = np.mean(class_pa) if class_pa else 0
    return mean_pa, class_iou, class_pa

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
        print(img.shape, gt.shape)
        return None

    metrics = calculate_metrics(img, gt)
    del img
    del gt
    return metrics

image_files = glob.glob(image_pth + '/*.*')

total_pa = 0
total_images = 0
class_iou_aggregate = {}
class_pa_aggregate = {}

with tqdm(total=len(image_files), desc="Evaluating images") as pbar:
    for filename in image_files:
        result = process_image(filename)
        if result is not None:
            pa, class_iou, class_pa = result
            total_pa += pa
            total_images += 1

            for cls, iou in class_iou.items():
                if cls not in class_iou_aggregate:
                    class_iou_aggregate[cls] = []
                class_iou_aggregate[cls].append(iou)

            for cls, pa_cls in zip(class_iou.keys(), class_pa):
                if cls not in class_pa_aggregate:
                    class_pa_aggregate[cls] = []
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

# Sort the items by cls
sorted_mean_class_iou = sorted(mean_class_iou.items())

# Print the sorted items
for cls, miou in sorted_mean_class_iou:
    print(f"{miou:.3f}")