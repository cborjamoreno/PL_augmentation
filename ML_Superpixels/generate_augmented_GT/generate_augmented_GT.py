import glob
import time
import cv2
import numpy as np
import csv
from collections import Counter
import argparse
import os
import math
import multiprocessing

class Superpixel:
    def __init__(self, maxlength=10000): #10000000
        self.lista_x = np.zeros([maxlength], dtype=np.int16)
        self.lista_y = np.zeros([maxlength], dtype=np.int16)
        self.lista_x[:]=-1
        self.lista_y[:]=-1
        self.index = 0

    def add(self, value_x, value_y):
        if self.index >= len(self.lista_x):
            #make it bigger
            new_size = len(self.lista_x)*10
            lista_x_new = np.zeros([new_size], dtype=np.int16)
            lista_y_new = np.zeros([new_size], dtype=np.int16)
            lista_x_new[:] = -1
            lista_y_new[:] = -1

            lista_x_new[:len(self.lista_x)] = self.lista_x
            lista_y_new[:len(self.lista_y)] = self.lista_y
            self.lista_x = lista_x_new
            self.lista_y = lista_y_new

        self.lista_x[self.index]=value_x
        self.lista_y[self.index]=value_y
        self.index += 1
        # print(len(lista_y_new))

    def clean(self):
        self.lista_x = self.lista_x[self.lista_x>=0]
        self.lista_y = self.lista_y[self.lista_y>=0]



# Given a superpixel and a GT image, returns the label value of the superpixel
def label_mayoria_x_y(superpixel, gt, DEFAULT_VALUE):
    # pixel label values of the superpixels
    pixel_values = gt[superpixel.lista_x.astype(int), superpixel.lista_y.astype(int)]
    # pixel label values of the superpixels excluding the default value
    values_labels = pixel_values[pixel_values < DEFAULT_VALUE]
    # Returns the value which appears the most
    if len(values_labels) == 0:
        return DEFAULT_VALUE
    else:
        count = Counter(values_labels)
        return count.most_common()[0][0]

# Given a csv file with segmentations (csv_name) and a sparse GT image (gt_name), returns Superpixel-augmented GT image
def image_superpixels_gt(csv_name, gt_name, n_superpixels, DEFAULT_VALUE):
    # print(gt_name)
    gt = cv2.imread(gt_name, 0)
    blank_image = np.zeros((gt.shape[0], gt.shape[1], 1), np.uint8)

    superpixels = {}

    for i in range(n_superpixels*2):
        superpixels[str(i)] = Superpixel()

    start_csv = time.time()
    # For each csv segmentation file, creates the Superpixel class
    with open(csv_name, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        i = 0

        for row in spamreader:
            fila = row[0].split(',')
            fila_count = len(fila)
            for j in range(fila_count):
                superpixel_index = fila[j]
                # Validate superpixel_index before accessing the dictionary
                if superpixel_index and superpixel_index in superpixels:
                    # The pixel here is (i, j). (superpixel_index) is the segmentation which the pixel belongs to
                    # Add the pixel to the Superpixel instance
                    superpixels[superpixel_index].add(i, j)

            i = i + 1
    # print("CSV time: " + str(time.time() - start_csv))

    start_getting_labels = time.time()
    # For each superpixel, gets its label value and writes it into the image to return
    for index in range(len(superpixels)):
        superpixels[str(index)].clean()
        label_superpixel = label_mayoria_x_y(superpixels[str(index)], gt, DEFAULT_VALUE)
        blank_image[superpixels[str(index)].lista_x.astype(int), superpixels[str(index)].lista_y.astype(int)] = int(
            label_superpixel)
    # print("Getting labels time: " + str(time.time() - start_getting_labels))
    # print("--------------------")   

    return blank_image

def process_file(filename, out_folder, superpixels_folder, csv_sizes, DEFAULT_VALUE):
    print('filename',filename)
    gt_name = filename.split('/')[-1]
    print('gt_name',gt_name)
    print('out_folder',out_folder)
    gt_filename = os.path.join(out_folder, gt_name)
    image_format = gt_name.split('.')[-1]

    # For each different segmentation generated
    for index in range(len(csv_sizes)):
        csv_name = os.path.join(superpixels_folder, 'superpixels_' + str(csv_sizes[index]),
                                gt_name.replace('.' + image_format, '') + '.csv')

        if index == 0:
            # creates the first one (it has to be the more detailed one, the segmentation with more segments)
            image_gt_new = image_superpixels_gt(csv_name, filename, csv_sizes[index], DEFAULT_VALUE)
        else:
            # Mask it with the less detailed segmentations in order to fill the areas with no valid labels
            image_gt_new_low = image_superpixels_gt(csv_name, filename, csv_sizes[index], DEFAULT_VALUE)
            image_gt_new[image_gt_new == DEFAULT_VALUE] = image_gt_new_low[image_gt_new == DEFAULT_VALUE]

    cv2.imwrite(gt_filename, image_gt_new)

def generate_augmented_GT(filename, dataset, default_value, number_levels, start_n_superpixels, last_n_superpixels):

    DEFAULT_VALUE = int(default_value)
    NL = int(number_levels)
    start_superpixels = int(start_n_superpixels)
    last_superpixels = int(last_n_superpixels)
    csv_sizes = []
    reduction_factor = math.pow(float(last_superpixels) / start_superpixels, 1. / (NL - 1))
    for level in range(NL):
        csv_sizes = csv_sizes + [int(round(start_superpixels * math.pow(reduction_factor, level)))]

    path_names = dataset.split('/')
    if path_names[-1] == '':
        path_names = path_names[:-1]
    directorio = path_names[-1]

    print('directorio:', directorio)
    sparse_dir = os.path.join('ML_Superpixels/Datasets',directorio, 'sparse_GT')
    out_dir = os.path.join('ML_Superpixels/Datasets',directorio, 'augmented_GT')
    superpixels_dir = os.path.join('ML_Superpixels/Datasets',directorio, 'superpixels')

    folder = 'train'

    # Execute superpixel genration
    size_sup_string = " "
    for size in csv_sizes:
        size_sup_string = size_sup_string + str(size) + " "
    
    # print('csv_sizes:', csv_sizes)

    start = time.time()
    # Generate superpixels
    os.system("sh ML_Superpixels/generate_superpixels/generate_superpixels.sh " + dataset + size_sup_string)
    # print("Superpixels generation time: " + str(time.time() - start))

    start_processing = time.time()
    in_folder = os.path.join(sparse_dir, folder)
    out_folder = os.path.join(out_dir, folder)
    superpixels_folder = os.path.join(superpixels_dir, folder)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    filename = os.path.join(in_folder, filename)
    process_file(filename, out_folder, superpixels_folder, csv_sizes, DEFAULT_VALUE)
    print("Processing time: " + str(time.time() - start_processing))