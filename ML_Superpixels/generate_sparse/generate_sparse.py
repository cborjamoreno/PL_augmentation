import glob
import cv2
import argparse
import random
import os
import math
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train", default='../Datasets/camvid')
parser.add_argument("--n_labels", help="Number of pixel labels to have", type=int, default=250)
parser.add_argument("--gridlike", help="Whether to have a grid-like structured ground-truth", type=int, default=1)
parser.add_argument("--image_format", help="Labeled image format (jpg, jpeg, png...)", default='png')
parser.add_argument("--default_value", help="Value of non-labeled pixels", type=int, default=255)
args = parser.parse_args()

path_names = args.dataset.split('/')
if path_names[-1] == '':
    path_names = path_names[:-1]
dataset_name = path_names[-1]

sparse_folder = os.path.join(dataset_name, 'sparse_GT')
labels_folder = os.path.join(args.dataset, 'labels')

NUM_LABELS = args.n_labels
grid = bool(args.gridlike)

folders = ['test', 'train']  # folders of the dataset

if not os.path.exists(sparse_folder):
    os.makedirs(sparse_folder)

for folder in folders:
    folder_to_write = os.path.join(sparse_folder, folder)
    if not os.path.exists(folder_to_write):
        os.makedirs(folder_to_write)

    for filename in glob.glob(os.path.join(labels_folder, folder) + '/*.' + args.image_format):
        image_name = os.path.basename(filename)
        new_filename = os.path.join(folder_to_write, image_name)

        img = cv2.imread(filename, 0)
        sparse = cv2.imread(filename, 0)
        sparse[:, :] = args.default_value

        i_size, j_size = img.shape
        if grid:  # Perform grid-like sparse ground-truth
            rate = i_size * 1. / j_size  # rate between height and width
            sqrt = math.sqrt(NUM_LABELS)
            n_i_points = int(rate * sqrt) + 1  # number of labels per column
            n_j_points = int(NUM_LABELS / n_i_points) + 1  # number of labels per row
            space_betw_i = int(i_size / n_i_points)  # space between every label
            space_betw_j = int(j_size / n_j_points)
            start_i = int((i_size - space_betw_i * (n_i_points - 1)) / 2)  # pixel to start labeling
            start_j = int((j_size - space_betw_j * (n_j_points - 1)) / 2)

            for i in range(start_i, n_i_points * space_betw_i, space_betw_i):
                for j in range(start_j, n_j_points * space_betw_j, space_betw_j):
                    sparse[i, j] = img[i, j]  # assign a label

        else:  # Perform random sparse ground-truth
            for _ in range(NUM_LABELS):
                i_point = random.randint(0, i_size - 1)
                j_point = random.randint(0, j_size - 1)
                sparse[i_point, j_point] = img[i_point, j_point]

        cv2.imwrite(new_filename, sparse)
        unique_values = np.unique(sparse)
        # print(f"Unique values in sparse GT for {image_name}: {unique_values}")

# print('GENERATION COMPLETED')