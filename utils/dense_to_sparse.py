import glob
import cv2
import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--images_pth", help="path of images", required=True)
    parser.add_argument("-o", "--output_file", help="output csv file name", required=True)
    parser.add_argument("-n", "--num_labels", help="number of labels to generate", default=300, type=int)
    return parser.parse_args()

def generate_grid_points(image, num_labels=300):
    height, width = image.shape
    aspect_ratio = height / width
    sqrt_labels_adjusted = np.sqrt(num_labels / aspect_ratio)
    n_i_points = round(sqrt_labels_adjusted * aspect_ratio)  # Adjusted number of labels per column
    n_j_points = round(sqrt_labels_adjusted)  # Adjusted number of labels per row

    # Adjust the number of points to be as close as possible to num_labels
    while n_i_points * n_j_points > num_labels:
        if n_i_points >= n_j_points:
            n_i_points -= 1
        else:
            n_j_points -= 1

    space_betw_i = height // n_i_points  # space between every label
    space_betw_j = width // n_j_points
    start_i = (height - space_betw_i * (n_i_points - 1)) // 2  # pixel to start labeling
    start_j = (width - space_betw_j * (n_j_points - 1)) // 2

    points = []
    for i in range(n_i_points):
        for j in range(n_j_points):
            pos_i = start_i + i * space_betw_i
            pos_j = start_j + j * space_betw_j
            points.append((pos_i, pos_j))

    # Add extra points if needed
    while len(points) < num_labels:
        extra_i = random.randint(0, height - 1)
        extra_j = random.randint(0, width - 1)
        if (extra_i, extra_j) not in points:
            points.append((extra_i, extra_j))

    # Remove excess points if needed
    if len(points) > num_labels:
        points = random.sample(points, num_labels)

    return points

def process_image(args):
    filename, num_labels = args
    data = []
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    points = generate_grid_points(img, num_labels)

    for pos_i, pos_j in points:
        color = img[pos_i, pos_j]
        if color != 255 and color != 0:  # Exclude white and black pixels
            image_name = os.path.basename(filename)
            image_name = os.path.splitext(image_name)[0] + ".png"
            data.append([image_name, pos_i, pos_j, color])

    return data

def process_images(image_pth, output_file, num_labels=300):
    image_files = glob.glob(image_pth + '/*.*')
    data = []

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_image, [(f, num_labels) for f in image_files]), total=len(image_files), desc="Processing images"))

    for result in results:
        data.extend(result)

    output_df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label'])
    output_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    args = parse_arguments()
    process_images(args.images_pth, args.output_file, args.num_labels)