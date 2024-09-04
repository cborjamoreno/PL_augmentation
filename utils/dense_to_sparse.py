import glob
import cv2
import argparse
import random
import os
import math
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--images_pth", help="path of images ", required=True)
parser.add_argument("-o", "--output_file", help="output csv file name", required=True)
parser.add_argument("-n", "--num_labels", help="number of labels", required=True)
args = parser.parse_args()

num_labels = int(args.num_labels)

image_pth = args.images_pth
output_file = args.output_file

if not os.path.exists(image_pth):
    print("Image path does not exist")
    exit()

data = []

for filename in glob.glob(image_pth + '/*.*'):
    img = cv2.imread(filename, 0)

    i_size, j_size = img.shape
    aspect_ratio = i_size / j_size
    sqrt_labels_adjusted = math.sqrt(num_labels / aspect_ratio)
    n_i_points = round(sqrt_labels_adjusted * aspect_ratio)  # Adjusted number of labels per column
    n_j_points = round(sqrt_labels_adjusted)  # Adjusted number of labels per row

    # Ensure we do not exceed NUM_LABELS
    while n_i_points * n_j_points > num_labels:
        if n_i_points >= n_j_points:
            n_i_points -= 1
        else:
            n_j_points -= 1

    space_betw_i = i_size // n_i_points  # space between every label
    space_betw_j = j_size // n_j_points
    start_i = (i_size - space_betw_i * (n_i_points - 1)) // 2  # pixel to start labeling
    start_j = (j_size - space_betw_j * (n_j_points - 1)) // 2

    for i in range(n_i_points):
        for j in range(n_j_points):
            pixel_i = start_i + i * space_betw_i
            pixel_j = start_j + j * space_betw_j
            if img[pixel_i, pixel_j] != 255 and img[pixel_i, pixel_j] != 0:
                data.append([os.path.basename(filename), pixel_i, pixel_j, img[pixel_i, pixel_j]])

output_df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label'])
output_df.to_csv(output_file, index=False)


