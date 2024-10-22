import glob
import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--images_pth", help="path of expanded images", required=True)
parser.add_argument("-gt", "--ground_truth_pth", help="path of ground truth images", required=True)
parser.add_argument("-bg", "--background_class", help="background class label", type=int, default=0)
args = parser.parse_args()

image_pth = args.images_pth
gt_pth = args.ground_truth_pth
background_class = int(args.background_class)

if not os.path.exists(image_pth):
    print("Image path does not exist")
    exit()

if not os.path.exists(gt_pth):
    print("Ground truth path does not exist")
    exit()

def calculate_metrics(img, gt, background_class):
    # Calculate PA (Pixel Accuracy)
    correct_pixels = np.sum(img == gt)
    total_pixels = gt.size
    pa = correct_pixels / total_pixels

    # Exclude background class for MPA and IoU calculation
    valid_pixels_mask = gt != background_class

    # Apply mask to exclude background
    img_valid = img[valid_pixels_mask]
    gt_valid = gt[valid_pixels_mask]

    # Calculate per-class IoU and MPA
    unique_classes = np.unique(gt_valid)

    class_iou = {}
    class_mpa = []

    for cls in unique_classes:
        gt_cls = gt_valid == cls
        img_cls = img_valid == cls

        mpa_cls = np.sum(gt_cls & img_cls) / np.sum(gt_cls) if np.sum(gt_cls) != 0 else 0
        class_mpa.append(mpa_cls)

        intersection = np.sum(gt_cls & img_cls)
        union = np.sum(gt_cls | img_cls)

        iou = intersection / union if union != 0 else 0
        class_iou[cls] = iou

    mean_mpa = np.mean(class_mpa) if class_mpa else 0
    mean_iou = np.mean(list(class_iou.values())) if class_iou else 0
    return pa, mean_mpa, mean_iou, class_iou, class_mpa

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

    metrics = calculate_metrics(img, gt, background_class)
    del img
    del gt
    return metrics

image_files = glob.glob(image_pth + '/*.*')

all_pa = []
all_mpa = []
all_miou = []
class_iou_aggregate = {}

with tqdm(total=len(image_files), desc="Evaluating images") as pbar:
    for filename in image_files:
        result = process_image(filename)
        if result is not None:
            pa, mean_mpa, mean_iou, class_iou, class_mpa = result
            all_pa.append(pa)
            all_mpa.append(mean_mpa)
            all_miou.append(mean_iou)

            for cls, iou in class_iou.items():
                if cls not in class_iou_aggregate:
                    class_iou_aggregate[cls] = []
                class_iou_aggregate[cls].append(iou)

        pbar.update(1)

# Calculate final measurements
final_pa = np.mean(all_pa)
final_mpa = np.mean(all_mpa)
final_miou = np.mean(all_miou)

print(f"Final PA: {final_pa}")
print(f"Final MPA: {final_mpa}")
print(f"Final MIoU: {final_miou}")

# Sort the items by cls
sorted_mean_class_iou = sorted(class_iou_aggregate.items())

# Print the sorted items
for cls, iou_list in sorted_mean_class_iou:
    mean_iou = np.mean(iou_list)
    print(f"{mean_iou:.3f}")