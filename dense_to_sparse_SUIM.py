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

NUM_LABELS = 100

image_pth = args.images_pth
output_file = args.output_file
color_dict = pd.read_csv(args.color_dict).to_dict()

if not os.path.exists(image_pth):
    print("Image path does not exist")
    exit()


data = []

for filename in glob.glob(image_pth + '/*.*'):
    img = cv2.imread(filename)

    i_size, j_size, _ = img.shape
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
            # search the color of img[i, j] in color_dict
            color = img[i, j]
            for key, value in color_dict.items():
                value_list = list(value.values())
                value_array = np.array(value_list)
                # print(color, value_array)
                if np.array_equal(color, value_array):
                    image_name = os.path.basename(filename)
                    # change the extension of the image name to jpg
                    image_name = os.path.splitext(image_name)[0]
                    data.append([image_name, i, j, key])

# cv2.destroyAllWindows() 

output_df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label'])
output_df.to_csv(output_file, index=False)


