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
    assert img.shape == gt.shape, "Image and GT dimensions do not match"

    # Calculate PA and MPA with all pixels
    total_pixels = img.reshape(-1, img.shape[-1]).shape[0]
    pixel_accuracy = np.sum(np.all(gt == img, axis=-1)) / total_pixels

    unique_classes = np.unique(gt.reshape(-1, gt.shape[-1]), axis=0)
    class_pa = []

    for cls in unique_classes:
        gt_cls = np.all(gt == cls, axis=-1)
        img_cls = np.all(img == cls, axis=-1)

        pa_cls = np.sum(gt_cls & img_cls) / np.sum(gt_cls) if np.sum(gt_cls) != 0 else 0
        class_pa.append(pa_cls)

    mean_pa = np.mean(class_pa) if class_pa else 0

    # Exclude black (unlabeled) pixels from both images for MIoU calculation
    valid_pixels = ~(np.all(gt == [0, 0, 0], axis=-1))
    img_valid = img[valid_pixels]
    gt_valid = gt[valid_pixels]

    if img_valid.size == 0 or gt_valid.size == 0:
        return pixel_accuracy, mean_pa, 0  # No valid pixels to process for MIoU

    class_iou = []

    for cls in unique_classes:
        if np.all(cls == [0, 0, 0]):  # Skip background for mIoU calculation
            continue
        gt_cls = np.all(gt_valid == cls, axis=-1)
        img_cls = np.all(img_valid == cls, axis=-1)

        intersection = np.sum(gt_cls & img_cls)
        union = np.sum(gt_cls | img_cls)

        iou = intersection / union if union != 0 else 0
        class_iou.append(iou)

    mean_iou = np.mean(class_iou) if class_iou else 0

    return pixel_accuracy, mean_pa, mean_iou

total_pa = 0
total_mpa = 0
total_miou = 0
total_images = 0

for filename in glob.glob(image_pth + '/*.*'):
    img = cv2.imread(filename)

    base_name = os.path.splitext(os.path.basename(filename))[0]
    gt_base_name = base_name.replace("_labels_rgb", "") + ".bmp"
    # gt_base_name = base_name + ".bmp"

    # print(f"Evaluating image {base_name}")
    # print(f"Ground truth filename: {gt_base_name}")
    # print(f'Image name: {filename}')

    gt_filename = os.path.join(gt_pth, gt_base_name)
    gt = cv2.imread(gt_filename)

    if gt is None:
        print(f"Ground truth not found for image {filename}")
        continue

    if img.shape != gt.shape:
        print(img.shape, gt.shape)
        continue
    pa, mpa, miou = calculate_metrics(img, gt)

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