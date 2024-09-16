import glob
import cv2
import argparse
import os
import numpy as np
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--images_pth", help="path of gt images of SUIM", required=True)
    parser.add_argument("-o", "--output_file", help="output csv file name", required=True)
    parser.add_argument("--color_dict", help="color dictionary for labels", default="SUIM_color_dict.csv")
    return parser.parse_args()

def load_color_dict(color_dict_path):
    color_dict = pd.read_csv(color_dict_path).to_dict()
    return {key: np.array(list(value.values())) for key, value in color_dict.items()}

def find_closest_color(color, color_dict):
    min_distance = float('inf')
    closest_color = None
    for key, value in color_dict.items():
        distance = np.linalg.norm(color - value)
        if distance < min_distance:
            min_distance = distance
            closest_color = key
    return closest_color

def generate_grid_points(image):
    height, width, _ = image.shape
    n_i_points = 10
    n_j_points = 10

    space_betw_i = height / n_i_points
    space_betw_j = width / n_j_points

    points = []
    for i in range(n_i_points):
        for j in range(n_j_points):
            pos_i = int((i + 0.5) * space_betw_i)
            pos_j = int((j + 0.5) * space_betw_j)
            points.append((pos_i, pos_j))

    return points

def process_images(image_pth, output_file, color_dict):
    data = []
    images_processed = 0

    for filename in glob.glob(image_pth + '/*.*'):
        img = cv2.imread(filename)
        points = generate_grid_points(img)

        for pos_i, pos_j in points:
            color = img[pos_i, pos_j]
            closest_color = find_closest_color(color, color_dict)
            if closest_color is not None:
                image_name = os.path.basename(filename)
                image_name = os.path.splitext(image_name)[0] + ".jpg"
                data.append([image_name, pos_i, pos_j, closest_color])

        images_processed += 1

    print(f"Processed {images_processed} images.")
    output_df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label'])
    output_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    args = parse_arguments()
    NUM_LABELS = 100
    color_dict = load_color_dict(args.color_dict)
    process_images(args.images_pth, args.output_file, color_dict)