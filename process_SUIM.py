import numpy as np
import cv2
import glob
import argparse
import pandas as pd
import os

def find_closest_color(color, color_dict):
    closest_color = None
    min_distance = float('inf')
    for dict_color in color_dict.values():
        distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color, dict_color)))
        if distance < min_distance:
            min_distance = distance
            closest_color = dict_color
    return closest_color

def all_colors_match(image, color_dict):
    unique_colors = set(tuple(color) for row in image for color in row)
    dict_colors = set(color_dict.values())
    return unique_colors.issubset(dict_colors)

def load_color_dict(csv_path):
    df = pd.read_csv(csv_path, header=None)
    color_dict = {df[i][0]: (int(df[i][3]), int(df[i][2]), int(df[i][1])) for i in df.columns}
    return color_dict

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--images_pth", help="path of gt images of SUIM", required=True)
parser.add_argument("-o", "--output_pth", help="path to save processed images", required=True)
parser.add_argument("--color_dict", help="color dictionary for labels", default="SUIM_color_dict.csv")
args = parser.parse_args()

color_dict = load_color_dict(args.color_dict)

for filename in glob.glob(args.images_pth + '/*.*'):
    img = cv2.imread(filename)

    if not all_colors_match(img, color_dict):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                original_color = tuple(img[i, j])
                closest_color = find_closest_color(original_color, color_dict)
                img[i, j] = closest_color
        
        output_pth = args.output_pth
        if not os.path.exists(output_pth):
            os.makedirs(output_pth)
        cv2.imwrite(args.output_pth + '/' + filename.split('/')[-1], img)