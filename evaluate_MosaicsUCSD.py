import glob
import cv2
import argparse
import os
import numpy as np

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
    # Exclude background color (0) from calculations
    valid_pixels_mask = gt != 0

    # Apply mask to exclude background
    img_valid = img[valid_pixels_mask]
    gt_valid = gt[valid_pixels_mask]

    # Calculate PA and MPA excluding background pixels
    total_valid_pixels = img_valid.size
    pixel_accuracy = np.sum(gt_valid == img_valid) / total_valid_pixels if total_valid_pixels > 0 else 0

    unique_classes = np.unique(gt_valid)
    class_pa = []

    for cls in unique_classes:
        gt_cls = gt_valid == cls
        img_cls = img_valid == cls

        pa_cls = np.sum(gt_cls & img_cls) / np.sum(gt_cls) if np.sum(gt_cls) != 0 else 0
        class_pa.append(pa_cls)

    mean_pa = np.mean(class_pa) if class_pa else 0

    # MIoU calculation already excludes background, so no changes needed here
    class_iou = []

    for cls in unique_classes:
        gt_cls = gt_valid == cls
        img_cls = img_valid == cls

        intersection = np.sum(gt_cls & img_cls)
        union = np.sum(gt_cls | img_cls)

        iou = intersection / union if union != 0 else 0
        class_iou.append(iou)
        # print('iou of class ', cls, ' is ', iou)

    mean_iou = np.mean(class_iou) if class_iou else 0

    return pixel_accuracy, mean_pa, mean_iou

total_pa = 0
total_mpa = 0
total_miou = 0
total_images = 0

for filename in glob.glob(image_pth + '/*.*'):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    base_name = os.path.splitext(os.path.basename(filename))[0]
    gt_base_name = base_name+".png"

    gt_filename = os.path.join(gt_pth, gt_base_name)
    gt = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)

    if gt is None:
        print(f"Ground truth not found for image {filename}")
        continue

    if img.shape != gt.shape:
        print(img.shape, gt.shape)
        continue
    pa, mpa, miou = calculate_metrics(img, gt)

    # print('pa: ', pa)
    # print('mpa: ', mpa)
    # print('miou: ', miou)

    total_pa += pa
    total_mpa += mpa
    total_miou += miou
    total_images += 1

mean_pa = total_pa / total_images if total_images > 0 else 0
mean_mpa = total_mpa / total_images if total_images > 0 else 0
mean_miou = total_miou / total_images if total_images > 0 else 0

print(f"Mean Pixel Accuracy (PA): {mean_pa}")
print(f"Mean Pixel Accuracy (MPA): {mean_mpa}")
print(f"Mean Intersection over Union (MIoU): {mean_miou}")