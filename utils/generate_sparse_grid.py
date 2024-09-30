import glob
import os
import pandas as pd
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import random
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--images_pth", help="path of images", required=True)
    parser.add_argument("-gt", "--ground_truth_pth", help="path of ground truth images", required=True)
    parser.add_argument("-o", "--output_file", help="name of the output csv file without extension", required=True)
    parser.add_argument("-n", "--num_labels", help="number of labels to generate", default=300, type=int)
    parser.add_argument("-t", "--image_type", help="type of ground truth images (grayscale or color)", choices=['grayscale', 'color'], default='grayscale')
    parser.add_argument("-c", "--color_dict", help="path to color dictionary CSV file", required=False)
    return parser.parse_args()

def load_color_dict(color_dict_path):
    color_dict = {}
    df = pd.read_csv(color_dict_path)
    class_names = df.columns.tolist()
    for class_name in class_names:
        rgb_values = df[class_name].tolist()
        color_dict[tuple(rgb_values)] = class_name
    return color_dict

def show_anns(anns, img):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)
    for ann in sorted_anns:
        print(f"Area: {ann['area']}")
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor('white')  # Set the figure background color to white
        ax.set_facecolor('white')  # Set the axes background color to white
        m = ann['segmentation']
        # Set the color to yellow (R=1, G=1, B=0, A=1)
        color_mask = np.array([1, 1, 0, 1])
        # Create a black background only on the size of the image
        img_mask = np.zeros((img.shape[0], img.shape[1], 4))
        img_mask[m] = color_mask
        # Create a white canvas larger than the image
        canvas = np.ones((img.shape[0], img.shape[1], 4))
        canvas[:, :, :3] = 0  # Set RGB to white
        canvas[:, :, 3] = 1  # Set alpha to 1 for opacity
        # Overlay the black background and yellow mask on the white canvas
        canvas[img_mask[:, :, 3] > 0] = img_mask[img_mask[:, :, 3] > 0]
        ax.imshow(np.ones_like(img) * 255)  # Display the original image as the background
        ax.imshow(canvas)  # Overlay the mask with some transparency
        ax.axis('off')
        plt.show()

def calculate_centroid(mask):
    indices = np.argwhere(mask)
    centroid = np.mean(indices, axis=0)
    return int(centroid[0]), int(centroid[1])

def generate_grid_points(image, num_labels=300):
    height, width = image.shape[:2]
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
    gt_filename, img_filename, num_labels, image_type, color_dict = args
    data = []

    # Ensure num_labels is an integer
    num_labels = int(num_labels)

    # Read the ground truth image
    img = cv2.imread(img_filename, cv2.COLOR_BGR2RGB)

    print('image_type', image_type)

    if image_type == 'grayscale':
        gt_img = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)
    else:
        gt_img = cv2.imread(gt_filename, cv2.IMREAD_COLOR)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    
    # Print the unique colors in gt_img
    if len(gt_img.shape) == 2:  # Grayscale image
        unique_colors = np.unique(gt_img)
    else:  # RGB image
        unique_colors = np.unique(gt_img.reshape(-1, gt_img.shape[2]), axis=0)

    # print('Unique colors in gt_img:', unique_colors)

    # print('color_dict', color_dict)
    
    points = generate_grid_points(img, num_labels)

    # Get the extension from the img_filename
    _, ext = os.path.splitext(img_filename)

    for pos_i, pos_j in points:
        if image_type == 'grayscale':
            color = gt_img[pos_i, pos_j]
            image_name = os.path.basename(img_filename)
            image_name = os.path.splitext(image_name)[0] + ext
            data.append([image_name, pos_i, pos_j, color])
        else:
            color = tuple(gt_img[pos_i, pos_j])
            if color_dict and color not in color_dict:
                print(f"Color not found in dictionary: {color}")
                continue
            image_name = os.path.basename(img_filename)
            image_name = os.path.splitext(image_name)[0] + ext
            label = color_dict.get(color, color)
            data.append([image_name, pos_i, pos_j, label])

    return data

def process_images(images_pth, ground_truth_pth, output_file, num_labels=300, image_type='grayscale', color_dict_path=None):
    data = []

    color_dict = None
    if color_dict_path:
        color_dict = load_color_dict(color_dict_path)

    # Get the list of ground truth images
    if os.path.isfile(ground_truth_pth):
        gt_image_files = [ground_truth_pth]
    else:
        gt_image_files = glob.glob(ground_truth_pth + '/*.*')

    # Get the list of images from images_pth
    if os.path.isfile(images_pth):
        image_files = [images_pth]
    else:
        image_files = glob.glob(images_pth + '/*.*')

    # Create a mapping of GT images to corresponding images
    gt_image_map = {os.path.splitext(os.path.basename(gt))[0]: gt for gt in gt_image_files}

    # Create a list of tuples (gt_image, corresponding_image)
    image_pairs = []
    for image_file in image_files:
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        if image_name in gt_image_map:
            gt_image = gt_image_map[image_name]
            image_pairs.append((gt_image, image_file))

    # Ensure num_labels is an integer
    num_labels = int(num_labels)

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_image, [(gt, img, num_labels, image_type, color_dict) for gt, img in image_pairs]), total=len(image_pairs), desc="Processing images"))

    for result in results:
        data.extend(result)

    # Modify the output filename to include the number of labels and add .csv extension
    modified_output_file = f"{output_file}_{num_labels}.csv"

    output_df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label'])
    output_df.to_csv(modified_output_file, index=False)

if __name__ == "__main__":
    args = parse_arguments()
    process_images(args.images_pth, args.ground_truth_pth, args.output_file, args.num_labels, args.image_type, args.color_dict)