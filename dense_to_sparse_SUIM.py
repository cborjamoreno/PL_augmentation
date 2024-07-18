import glob
import cv2
import argparse
import random
import os
import math
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--images_pth", help="path of gt images of SUIM", required=True)
parser.add_argument("-o", "--output_file", help="output csv file name", required=True)
parser.add_argument("--color_dict", help="color dictionary for labels", default="SUIM_color_dict.csv")
args = parser.parse_args()

NUM_LABELS = 300

WIDTH = 640
HEIGHT = 480

image_pth = args.images_pth
output_file = args.output_file
color_dict = pd.read_csv(args.color_dict).to_dict()

if not os.path.exists(image_pth):
    print("Image path does not exist")
    exit()

data = []
images_processed = 0
images_with_less_labels = 0

for filename in glob.glob(image_pth + '/*.*'):
    img = cv2.imread(filename)
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    i_size, j_size, _ = img.shape

    sqrt = math.sqrt((i_size * j_size) / NUM_LABELS)
    n_i_points = max(int(i_size / sqrt), 1)
    n_j_points = max(int(j_size / sqrt), 1)

    while n_i_points * n_j_points > NUM_LABELS:
        if n_i_points > n_j_points:
            n_i_points -= 1
        else:
            n_j_points -= 1

    space_betw_i = i_size / n_i_points
    space_betw_j = j_size / n_j_points

    generated_labels = 0

    # print('numero de pontos:', n_i_points * n_j_points)

    tolerance = 10  # Define a tolerance level for color matching

    for i in range(n_i_points):
        for j in range(n_j_points):
            pos_i = int((i + 0.5) * space_betw_i)
            pos_j = int((j + 0.5) * space_betw_j)

            # Extract the BGR color at the point (pos_i, pos_j)
            color = img[pos_i, pos_j]

            # Compare the extracted color with the colors in color_dict_processed
            for key, value in color_dict.items():
                value_list = list(value.values())
                value_array = np.array(value_list)
                # Calculate Euclidean distance between the color and value_array
                distance = np.linalg.norm(color - value_array)
                # Check if the distance is within the tolerance
                if distance <= tolerance:
                    # if distance > 0:
                        # print(f"Matched color {key} at ({pos_i}, {pos_j}) with distance {distance}")
                    image_name = os.path.basename(filename)
                    # Change the extension of the image name to jpg
                    image_name = os.path.splitext(image_name)[0]
                    data.append([image_name, pos_i, pos_j, key])
                    generated_labels += 1
                    break  # Assuming each point can only match one color

    if generated_labels != NUM_LABELS:
        images_with_less_labels += 1
    images_processed += 1

print(f"Processed {images_processed} images.")
print(f"Images with less labels: {images_with_less_labels}")

output_df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label'])
output_df.to_csv(output_file, index=False)