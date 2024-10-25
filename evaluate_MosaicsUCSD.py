import glob
import cv2
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torchmetrics
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--images_pth", help="path of expanded images", required=True)
parser.add_argument("-gt", "--ground_truth_pth", help="path of ground truth images", required=True)
parser.add_argument("-bg", "--background_class", help="background class label", type=int, default=34)
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

# Number of classes set to 35, excluding the background (class 34)
NUM_CLASSES = 35

pa_metric = torchmetrics.Accuracy(task='multiclass', num_classes = NUM_CLASSES, ignore_index=background_class)
mpa_metric = torchmetrics.Accuracy(task='multiclass', num_classes = NUM_CLASSES, ignore_index=background_class, average='macro')
iou_metric = torchmetrics.JaccardIndex(task='multiclass', num_classes = NUM_CLASSES, ignore_index=background_class)

image_files = glob.glob(image_pth + '/*.*')

with tqdm(total=len(image_files), desc="Evaluating images") as pbar:
    for filename in image_files:
        pil_img = Image.open(filename)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        gt_base_name = base_name + ".png"
        gt_filename = os.path.join(gt_pth, gt_base_name)
        GT_pil_img = Image.open(gt_filename)

        # Convert PIL images to numpy arrays
        img_np = np.array(pil_img)
        gt_np = np.array(GT_pil_img)

        img_torch = torch.from_numpy(img_np).int()
        labels_torch = torch.from_numpy(gt_np).int()

        inactive_index = labels_torch == background_class
        img_torch[inactive_index] = background_class

        acc = pa_metric(img_torch, labels_torch)
        m_acc = mpa_metric(img_torch, labels_torch)
        m_iou = iou_metric(img_torch, labels_torch)

        pbar.update(1)

acc = pa_metric.compute()
m_acc = mpa_metric.compute()
miou = iou_metric.compute()

print("per class mean intersection over union:", miou)

class_ious_torch=miou[miou != 0]
mean_iou_torch = torch.nanmean(class_ious_torch)

print("PA:", acc.item()*100, ", mPA:", m_acc.item()*100, ", mIOU per class:", mean_iou_torch.item()*100)
