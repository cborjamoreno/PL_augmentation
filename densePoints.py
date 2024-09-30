
import argparse
import random
import shutil
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys, os
import pandas as pd
import colorsys
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod
from sklearn.neighbors import KernelDensity
import time
import subprocess
import multiprocessing as mp
from sklearn.cluster import KMeans
from shapely.geometry import Polygon
from shapely.validation import make_valid
import torch
import torchvision

from ML_Superpixels.generate_augmented_GT.generate_augmented_GT import generate_augmented_GT

BORDER_SIZE = 0
MAX_DISTANCE = 100

def get_key(val, dict):
   
    for key, value in dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=100, marker_color='blue', edge_color='white'):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color=marker_color, marker='*', s=marker_size, edgecolor=edge_color, linewidth=1.25) 
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def blend_translucent_color(img, row, column, color, alpha):
    # Get the existing color
    existing_color = img[row, column]
    
    # Blend the new color with the existing color
    blended_color = existing_color * (1 - alpha) + color * alpha

    # print('blended_color:', existing_color, color, blended_color, alpha)

    # Assign the blended color to the pixel
    img[row, column] = blended_color

def estimate_density(points, bandwidth=1.0, n_jobs=4):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(points)
    return kde

def evaluate_density(kde, points):
    densities = np.exp(kde.score_samples(points))
    return densities

def adjust_priorities_with_density(image_df, densities, density_factor=0.5, size_factor=0.5):
    # Calculate initial priorities based on label frequency
    priorities = image_df['Label'].value_counts() / len(image_df)
    priorities = 1 - priorities

    label_densities = []
    label_sizes = []
    for label in priorities.index:
        label_indices = image_df[image_df['Label'] == label].index
        label_density = densities[label_indices].mean()
        label_densities.append(label_density)
        
        # Calculate the size of the segment
        label_size = len(label_indices)
        label_sizes.append(label_size)
    
    # Normalize densities so the maximum density is 1
    max_density = max(label_densities)
    label_densities = [density / max_density for density in label_densities]
    
    # Normalize sizes so the maximum size is 1
    max_size = max(label_sizes)
    label_sizes = [size / max_size for size in label_sizes]
    
    # Adjust priorities with density and size
    label_priorities = {}
    for label, priority, density, size in zip(priorities.index, priorities, label_densities, label_sizes):
        adjusted_priority = priority * (1 - density_factor - size_factor) + density_factor * density + size_factor * (1 - size)
        print(density)
        print(priority)
        print(adjusted_priority)
        exit()
        label_priorities[label] = adjusted_priority
    
    return label_priorities
    
def calculate_iou_boxes(segment_data, other_segment_data):
    # Calculate bounding boxes
    bbox1 = [segment_data['Column'].min(), segment_data['Row'].min(), segment_data['Column'].max(), segment_data['Row'].max()]
    bbox2 = [other_segment_data['Column'].min(), other_segment_data['Row'].min(), other_segment_data['Column'].max(), other_segment_data['Row'].max()]

    # Calculate intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0, None

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    intersection_box = [x_left, y_top, x_right, y_bottom]

    # Calculate union
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0, None

    # Calculate and return IoU
    return intersection_area / union_area, intersection_box

def calculate_segment_priorities(image_df, densities, density_factor=1.0):
    segment_densities = image_df.groupby('Segment').apply(lambda x: densities[x.index].mean())

    # Normalize densities
    max_density = segment_densities.max()
    normalized_densities = segment_densities / max_density

    # Calculate priorities based on density alone
    segment_priorities = normalized_densities * density_factor
    print(segment_priorities)
    return segment_priorities

def is_point_in_segment(point, segment_data):
    """Check if the point exists in segment_data."""
    return any((segment_data['Row'] == point[1]) & (segment_data['Column'] == point[0]))
    
def merge_labels(image_df, gt_points, gt_labels):
    merged_df = pd.DataFrame(columns=['Name', 'Row', 'Column', 'Label'])
    segments = image_df.groupby('Segment')
    iou_threshold = 0.6
    chosen_segments_by_iou = set()  # Set to keep track of segments chosen by IoU
    points_to_add = []

    intersection_records = []

    for i, (segment_id, segment_data) in enumerate(segments):
        for other_segment_id, other_segment_data in list(segments)[i+1:]:
            if segment_data['Label'].iloc[0] != other_segment_data['Label'].iloc[0]:
                iou, ibox = calculate_iou_boxes(segment_data, other_segment_data)

                if iou > iou_threshold:

                    # print(f"Segments {segment_id} of label {segment_data['Label'].iloc[0]} and {other_segment_id} of label {other_segment_data['Label'].iloc[0]} have IoU {iou}")

                    # Initialize counters
                    segment_count_right = 0
                    segment_count_wrong = 0
                    other_segment_count_right = 0
                    other_segment_count_wrong = 0

                    label_segment = segment_data['Label'].iloc[0]
                    label_other_segment = other_segment_data['Label'].iloc[0]

                    # Get gt points that belong to each class and save their labels
                    gt_points_in_segment = [(point, gt_labels[idx]) for idx, point in enumerate(gt_points) if is_point_in_segment(point, segment_data)]
                    gt_points_in_other_segment = [(point, gt_labels[idx]) for idx, point in enumerate(gt_points) if is_point_in_segment(point, other_segment_data)]

                    # plot segment_data points and other_segment_data points
                    # plt.scatter(other_segment_data['Column'], -other_segment_data['Row'], color='red')
                    # plt.scatter(segment_data['Column'], -segment_data['Row'], color='blue')
                    # plt.show()

                    # Count the number of gt points inide each segment
                    for _, label in gt_points_in_segment:
                        if label == label_segment:
                            segment_count_right += 1
                        else:
                            segment_count_wrong += 1
                    
                    for _, label in gt_points_in_other_segment:
                        if label == label_other_segment:
                            other_segment_count_right += 1
                        else:
                            other_segment_count_wrong += 1
                    
                    difference_segement = segment_count_right - segment_count_wrong
                    # print(f"Segment {segment_id} with label {segment_data['Label'].iloc[0]} has {segment_count_right} right points and {segment_count_wrong} wrong points")
                    # print(f"Difference segment: {difference_segement}")
                    difference_other_segment = other_segment_count_right - other_segment_count_wrong
                    # print(f"Other Segment {other_segment_id} with label {other_segment_data['Label'].iloc[0]} has {other_segment_count_right} right points and {other_segment_count_wrong} wrong points")
                    # print(f"Difference other segment: {difference_other_segment}")

                    chosen_segment_id = segment_id if difference_segement > difference_other_segment else other_segment_id
                    chosen_segment_data = segment_data if difference_segement > difference_other_segment else other_segment_data
                    # print(f"Chose segment {chosen_segment_id} with label {chosen_segment_data['Label'].iloc[0]}, difference_segment: {difference_segement}, difference_other_segment: {difference_other_segment}")
                    
                    # concat merged_df with segment data
                    merged_df = pd.concat([merged_df, chosen_segment_data])
                    chosen_segments_by_iou.add(segment_id)
                    chosen_segments_by_iou.add(other_segment_id)
     
    unique_labels = np.unique(gt_labels)
    kde_dict = {label: estimate_density(gt_points[gt_labels == label], bandwidth=1.0) for label in unique_labels}
    assigned_densities = np.zeros(len(image_df))

    # Calculate densities for each segment
    for segment_id, segment_data in segments:
        label = segment_data['Label'].iloc[0]
        expanded_label_points = segment_data[['Row', 'Column']].values
        if expanded_label_points.size == 0:
            continue
        densities_label = evaluate_density(kde_dict[label], expanded_label_points)
        assigned_densities[segment_data.index] = densities_label

    image_density = len(gt_points) / (image_df['Row'].max() * image_df['Column'].max())
    normalized_densities = assigned_densities / image_density

    # Calculate segment priorities
    segment_priorities = calculate_segment_priorities(image_df, normalized_densities, density_factor=0.5)

    # Process remaining segments not in chosen_segments_by_iou
    remaining_segments = [seg for seg in segments if seg[0] not in chosen_segments_by_iou]
    remaining_segments.sort(key=lambda seg: segment_priorities[seg[0]], reverse=True)

    for segment_id, segment_data in remaining_segments:
        if segment_id in chosen_segments_by_iou:
            continue
        merged_df = pd.concat([merged_df, segment_data])
        chosen_segments_by_iou.add(segment_id)

    point_labels = {}
    label_counts = {}
    for index, row in image_df.iterrows():
        if row['Segment'] in chosen_segments_by_iou:
            continue
        adjusted_point = (row['Row'], row['Column'])
        label = row['Label']
        name = row['Name']
        label_count = label_counts.get(adjusted_point, 0) + 1
        label_counts[adjusted_point] = label_count
        other_label = point_labels.get(adjusted_point, None)
        if label_count > 1 and segment_priorities[row['Segment']] > segment_priorities[other_label[1]]:
            point_labels[adjusted_point] = (name, label)
        elif label_count == 1:
            point_labels[adjusted_point] = (name, label)

    for point, (name, label) in point_labels.items():
        row, column = point
        points_to_add.append({'Name': name, 'Row': row, 'Column': column, 'Label': label})

    merged_df = pd.concat([merged_df, pd.DataFrame(points_to_add)], ignore_index=True)
    return merged_df

def generate_image(image_df, image, gt_points, color_dict, image_name, output_dir, label_str=None):
    if label_str is None:
        label_str = 'ALL'

    # Create a black image
    black = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    black = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    black = black.astype(float) / 255
    black = np.dstack((black, np.ones((black.shape[0], black.shape[1])))) # RGBA

    height, width, _ = black.shape
    dpi = 100  # Change this to adjust the quality of the output image
    figsize = width / dpi, height / dpi

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(black)

    show_points(gt_points, np.ones(len(gt_points), dtype=int), plt.gca(), marker_color='black', edge_color='yellow')
    plt.axis('off')

    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + image_name + '_' + label_str + '_sparse.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    start = time.time()
    # Color the points in the image
    for _, row in image_df.iterrows():
        if row['Column'] < black.shape[1] and row['Row'] < black.shape[0]:
            point = (row['Row'] + BORDER_SIZE, row['Column'] + BORDER_SIZE)

            # Get the color and add an alpha channel
            color = np.array([color_dict[str(row['Label'])][0] / 255.0, color_dict[str(row['Label'])][1] / 255.0, color_dict[str(row['Label'])][2] / 255.0, 1])
            blend_translucent_color(black, point[0], point[1], color, color[3])
    # print(f"Time taken by color the points in the image: {time.time() - start} seconds")

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(black)
    show_points(gt_points, np.ones(len(gt_points), dtype=int), plt.gca(), marker_color='black', edge_color='yellow')
    plt.axis('off')

    plt.savefig(output_dir + image_name + '_' + label_str + '_expanded.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    print('Image saved in', output_dir + image_name + '_' + label_str + '_expanded.png')

def generate_image_per_class(image_df, image, points, labels, color_dict, image_name, output_dir, label_str):

    # if label_str is not a string convert it to a string
    if not isinstance(label_str, str):
        label_str = str(label_str)
    
    # Create a black image
    black = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    black = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    black = black.astype(float) / 255
    black = np.dstack((black, np.ones((black.shape[0], black.shape[1])))) # RGBA

    height, width, _ = black.shape
    dpi = 100  # Change this to adjust the quality of the output image
    figsize = width / dpi, height / dpi

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(black)
    show_points(points, labels, plt.gca(), marker_color='black', edge_color='yellow')
    plt.axis('off')

    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + image_name + '_' + label_str + '_sparse.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    print('color_dict:', color_dict)

    # Color the points in the image
    for _, row in image_df.iterrows():
        if row['Column'] < black.shape[1] and row['Row'] < black.shape[0]:
            point = (row['Row'] + BORDER_SIZE, row['Column'] + BORDER_SIZE)

            # Get the color and add an alpha channel
            color = np.array([color_dict[str(row['Label'])][0] / 255.0, color_dict[str(row['Label'])][1] / 255.0, color_dict[str(row['Label'])][2] / 255.0, 1])
            blend_translucent_color(black, point[0], point[1], color, color[3])

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(black)
    show_points(points, labels, plt.gca(), marker_color='black', edge_color='yellow')
    plt.axis('off')

    plt.savefig(output_dir + image_name + '_' + label_str + '_expanded.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_image_per_segment(image_df, segment, image, points, labels, color_dict, image_name, output_dir, label_str):
    
    # Create a black image
    black = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    black = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    black = black.astype(float) / 255
    black = np.dstack((black, np.ones((black.shape[0], black.shape[1])))) # RGBA

    height, width, _ = black.shape
    dpi = 100  # Change this to adjust the quality of the output image
    figsize = width / dpi, height / dpi

    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Color the points in the image
    for _, row in image_df.iterrows():
        if row['Column'] < black.shape[1] and row['Row'] < black.shape[0]:
            point = (row['Row'] + BORDER_SIZE, row['Column'] + BORDER_SIZE)

            # Get the color and add an alpha channel
            color = np.array([color_dict[row['Label']][0] / 255.0, color_dict[row['Label']][1] / 255.0, color_dict[row['Label']][2] / 255.0, 1])
            blend_translucent_color(black, point[0], point[1], color, color[3])

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(black)
    show_points(points, labels, plt.gca(), marker_color='black', edge_color='yellow')
    plt.axis('off')

    plt.savefig(f"{output_dir}{image_name}_segment_{segment}.png", dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_distinct_colors(n, threshold=50):

    # Generate a large number of random colors
    colors = np.random.randint(0, 256, size=(n*100, 3))

    # Use the k-means algorithm to partition the colors into n clusters
    kmeans = KMeans(n_clusters=n, random_state=0).fit(colors)

    # Use the centroid of each cluster as a distinct color
    distinct_colors = kmeans.cluster_centers_.astype(int)

    return distinct_colors

def adjust_colors(colors, n_clusters=15):
    # Fit the k-means algorithm to the colors
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(colors)

    # Replace each color with the centroid of its cluster
    adjusted_colors = kmeans.cluster_centers_[kmeans.labels_]

    return adjusted_colors.astype(int)

def export_colors(color_dict, output_dir):
    # Export the color_dict to a file
    color_dict_df = pd.DataFrame(color_dict)
    color_dict_df.to_csv(output_dir + 'generated_color_dict.csv', index=False, header=True)

def create_color_dict(labels, output_dir, n_clusters=15):
    # Generate a set of distinct colors
    distinct_colors = generate_distinct_colors(len(labels))
    distinct_colors = adjust_colors(distinct_colors, n_clusters=n_clusters)
    
    # Prepare the color dictionary for export
    color_dict = {}
    for label, color in zip(labels, distinct_colors):
        color_dict[label] = color
    
    # Export the color dictionary
    export_colors(color_dict, output_dir)

    return color_dict

def plot_mosaic(image, image_labels, results_path):
    num_images = len(os.listdir(results_path))

    # Get the files in the results directory
    results = os.listdir(results_path)
    aux = []

    # Short results keeping the same order of the labels
    for label in image_labels:
        for result in results:
            if label in result:
                # Move the result to the front
                aux.append(result)
                results.remove(result)
                break

    results = aux
    
    # Specify the number of rows and columns
    num_rows = 3
    num_cols = (num_images + 1) // num_rows + ((num_images + 1) % num_rows > 0)

    # Calculate the aspect ratio of the images
    aspect_ratio = image.shape[1] / image.shape[0]

    # Calculate the size of the subplots
    fig_width = 25
    subplot_width = fig_width / num_cols
    subplot_height = subplot_width / aspect_ratio

    # Create a figure with a grid of subplots with the correct aspect ratio
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, subplot_height * num_rows))

    # Plot the original image
    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0][0].imshow(image_RGB, aspect='auto')
    axes[0][0].axis('off')

    # Plot the images with the labels
    for i in range(1, num_images):
        cv2_image = cv2.imread(results_path + results[i-1])
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        row = i // num_cols
        col = i % num_cols
        axes[row, col].imshow(cv2_image, aspect='auto')
        
        # Remove the axis
        axes[row, col].axis('off')

    # Remove the empty subplots
    for i in range(num_images, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])
    
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0.05, top=1)

    # Create a list of patches for the legend
    patches = []

    for label in image_labels:
        if label in color_dict:
            color = np.array([color_dict[label][0]/255.0, color_dict[label][1]/255.0, color_dict[label][2]/255.0])
            print('label color:', label, color)
            patch = mpatches.Patch(color=color, label=label)
            patches.append(patch)

    # Calculate the width of the legend items
    legend_item_width = max(len(label) for label in image_labels) * 0.01  # Approximate width of a legend item in inches

    # Calculate the maximum number of columns that can fit in the figure
    ncols = int(fig_width / legend_item_width)

    # Add the legend at the bottom of the mosaic
    legend = plt.legend(handles=patches, bbox_to_anchor=(0.6, 0), loc='upper center', ncol=ncols, borderaxespad=0.)

    # Adjust the font size of the legend to fit the figure
    plt.setp(legend.get_texts(), fontsize='small')

    plt.savefig(results_path + 'mosaic.png', bbox_inches='tight', pad_inches=0)
    plt.show()

def checkMissmatchInFiles(image_names_csv, image_names_dir):
    # Print images in .csv that are not in the image directory
    for image_name in image_names_csv:
        if image_name + '.jpg' not in image_names_dir:
            print(f"Image {image_name} in .csv but not in image directory")

    # Print images in the image directory that are not in the .csv
    for image_name in image_names_dir:
        if image_name[:-4] not in image_names_csv:
            print(f"Image {image_name} in image directory but not in .csv")
    
    print("Done checking missmatch in files!\n\n")

class LabelExpander(ABC):
    def __init__(self, color_dict, input_df, labels, output_df): 
        self.color_dict = color_dict
        self.input_df = input_df
        self.unique_labels_str = labels
        self.image_names_csv = input_df['Name'].unique()
        self.output_df = output_df
        self.gt_points = np.array([])
        self.gt_labels = np.array([])
        self.remove_far_points = remove_far_points
        self.generate_eval_images = generate_eval_images
        self.generate_csv = generate_csv

    def generate_csv(self):
        # TODO: 
        pass

    def expand_image(self, unique_labels_str_i, image, eval_images_dir_i):      
        expanded_df = self.expand_labels(points, labels, unique_labels_str_i, image, image_name, eval_images_dir_i)
        
        if self.generate_eval_images:
            start_generate_image = time.time() 
            generate_image(expanded_df, image, self.gt_points, self.color_dict, image_name, eval_images_dir_i)
            print(f"Time taken by generate_image: {time.time() - start_generate_image} seconds")

        if self.generate_csv:
            pass
            # start_generate_csv = time.time()
            # # TODO: Perform Kmeans clustering separately for each class
            # if not image_df.empty:
            #     num_clusters = len(unique_labels_str_i)
            #     kmeans = KMeans(n_clusters=num_clusters)
            #     image_df['Cluster'] = kmeans.fit_predict(image_df[['Row', 'Column']])

            #     # Count the number of points in each cluster
            #     cluster_counts = image_df['Cluster'].value_counts()

            #     # Calculate the sampling rate for each class (inverse of class count)
            #     sampling_rates = 1 / cluster_counts

            #     # Normalize the sampling rates so that they sum up to 1
            #     sampling_rates = sampling_rates / sampling_rates.sum()

            #     # Calculate the number of points to sample from each class
            #     points_per_cluster = (sampling_rates * 950).round().astype(int)

            #     # Create an empty DataFrame to store the sampled points
            #     sampled_df = pd.DataFrame(columns=['Name', 'Row', 'Column', 'Label'])

            #     # Sample points from each cluster
            #     for cluster in image_df['Cluster'].unique():
            #         cluster_points = image_df[image_df['Cluster'] == cluster]
            #         num_points = points_per_cluster[cluster]
            #         sampled_points = cluster_points.sample(min(len(cluster_points), num_points))
            #         sampled_df = pd.concat([sampled_df, sampled_points])

            #     # If there are still points left to sample, sample randomly from the entire DataFrame
            #     if len(sampled_df) < 950:
            #         remaining_points = 950 - len(sampled_df)
            #         additional_points = image_df.sample(remaining_points)
            #         sampled_df = pd.concat([sampled_df, additional_points])
                    
            #     sparse_df = pd.concat([sparse_df, sampled_df])
            # else:
            #     print("Warning: image_df is empty. Skipping KMeans clustering and sampling.")
            
            # print(f"Time taken by generate_image: {time.time() - start_generate_csv} seconds")

        return expanded_df
        
    @abstractmethod
    def expand_labels(self, points, labels, unique_labels_str, image, image_name, eval_image_dir=None):
        pass

class SAMLabelExpander(LabelExpander):
    def __init__(self, color_dict, input_df, labels, output_df, predictor, generate_eval_images=False):
        super().__init__(color_dict, input_df, labels, output_df)
        self.predictor = predictor
        self.generate_eval_images = generate_eval_images
        
    def expand_labels(self, points, labels, unique_labels_str, image, image_name, eval_image_dir=None):
        expanded_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label", "Segment"])

        # crop the image BORDER_SIZE pixels from each side
        cropped_image = image[BORDER_SIZE:image.shape[0]-BORDER_SIZE, BORDER_SIZE:image.shape[1]-BORDER_SIZE]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(cropped_image)

        # Initialize the segment counter
        segment_counter = 0

        self.gt_points = np.array([])
        self.gt_labels = np.array([])

        # use SAM to expand the points into masks. Artificial new GT points.
        for i in range(len(unique_labels_str)):
            
            # Filter points and labels by unique_labels_str[i]
            filtered_points = points[labels==unique_labels_str[i]]
            filtered_labels = np.array(labels)[labels==unique_labels_str[i]]

            # Create a mask for points within the image borders
            inbound = (filtered_points[:, 0] > BORDER_SIZE) & (filtered_points[:, 0] < image.shape[0] - BORDER_SIZE) & (filtered_points[:, 1] > BORDER_SIZE) & (filtered_points[:, 1] < image.shape[1] - BORDER_SIZE)

            # Apply the mask to points and labels
            _points = filtered_points[inbound]
            _labels = filtered_labels[inbound]

            if len(_points) == 0:
                print(f"No points for label {unique_labels_str[i]}")
                continue

            # Transform the points after cropping
            # Change x and y coordinates
            _points = np.flip(_points, axis=1)

            # Store _points in gt_points
            if len(self.gt_points) == 0:
                self.gt_points = _points
            else:
                self.gt_points = np.concatenate((self.gt_points, _points), axis=0)

            _labels_ones = np.ones(len(_points), dtype=int)

            # Store _labels in gt_labels
            if len(self.gt_labels) == 0:
                self.gt_labels = _labels
            else:
                self.gt_labels = np.concatenate((self.gt_labels, _labels), axis=0)

            _points_pred = _points.copy()
            _points_pred[:, 0] -= BORDER_SIZE
            _points_pred[:, 1] -= BORDER_SIZE

            best_mask = np.zeros((cropped_image.shape[0], cropped_image.shape[1]), dtype=bool)
            new_points = np.argwhere(best_mask)
            data = []

            for p, l in zip(_points_pred, _labels_ones):

                # print(f"Expanding point {p} with label {l}")

                # Convert point and label to a format that can be compared
                point_row, point_column = p[1], p[0]
                label_str = unique_labels_str[i]

                start = time.time()
                # Check if point with the same coordinates and label already exists in data
                point_exists = any(d["Row"] == point_row and d["Column"] == point_column and d["Label"] == label_str for d in data)
                # print(f"Time taken by point_exists: {time.time() - start} seconds")

                if point_exists:
                    # print(f"Point {p} with label {l} already exists in data")
                    continue

                start = time.time()
                _, _, logits = self.predictor.predict(
                    point_coords=np.array([p]),
                    point_labels=np.array([l]),
                    multimask_output=True,
                )
                
                mask_input = logits[0, :, :] # Choose the model's best mask

                mask, _, _ = self.predictor.predict(
                    point_coords=np.array([p]),
                    point_labels=np.array([l]),
                    mask_input=mask_input[None, :, :],
                    multimask_output=True,
                )

                segment_counter += 1

                # Determine new points for the current mask
                new_points = np.argwhere(mask[0])

                # Step 1: Create a set for existing points for the specific label
                existing_points = {(d["Row"], d["Column"]) for d in data if d["Label"] == label_str}

                for point in new_points:
                    # Step 3: Check if the point exists in the set
                    if (point[0], point[1]) not in existing_points:
                        # The point does not exist, so add it to both `data` and the set
                        data.append({
                            "Name": image_name,
                            "Row": point[0],
                            "Column": point[1],
                            "Label": label_str,
                            "Segment": segment_counter
                        })
                        # Step 4: Add the new point to the set
                        existing_points.add((point[0], point[1]))

            _points_pred[:, 0] += BORDER_SIZE
            _points_pred[:, 1] += BORDER_SIZE

            new_data_df = pd.DataFrame(data)
            
            if self.generate_eval_images:
                generate_image_per_class(new_data_df, image, _points_pred, _labels_ones, color_dict, image_name, eval_image_dir, unique_labels_str[i])
            
            expanded_df = pd.concat([expanded_df, new_data_df], ignore_index=True)            
            print(f'{len(_points_pred)} points of class \'{unique_labels_str[i]}\' expanded to {len(new_points)} points')

        # Merge the dense labels
        gt_points = self.gt_points - BORDER_SIZE
        merged_df = merge_labels(expanded_df, gt_points, self.gt_labels)

        return merged_df

class SuperpixelLabelExpander(LabelExpander):
    def __init__(self, dataset, color_dict, input_df, labels, output_df):
        super().__init__(color_dict, input_df, labels, output_df)
        self.dataset = dataset

    def createSparseImage(self, points, labels, image_shape=(1000, 1000)):
        # Create a white image with the specified shape
        sparse_image = np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8) * 255
        
        for point, label in zip(points, labels):

            grayscale_value = int(label)
            # Ensure the grayscale value is within the valid range [0, 255]
            grayscale_value = np.clip(grayscale_value, 0, 255)
            # Set the pixel at the point's location to the grayscale value
            sparse_image[int(point[1]), int(point[0])] = [grayscale_value, grayscale_value, grayscale_value]

        return sparse_image

    def expand_labels(self, points, labels, unique_labels_str, image, image_name, eval_image_dir=None):

        # crop the image BORDER_SIZE pixels from each side
        cropped_image = image[BORDER_SIZE:image.shape[0]-BORDER_SIZE, BORDER_SIZE:image.shape[1]-BORDER_SIZE]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        numeric_labels = np.zeros(len(labels), dtype=int)
        for i, label in enumerate(unique_labels_str, start=1):
            numeric_labels[labels == label] = i

        inbound = (points[:, 0] > BORDER_SIZE) & (points[:, 0] < image.shape[0] - BORDER_SIZE) & (points[:, 1] > BORDER_SIZE) & (points[:, 1] < image.shape[1] - BORDER_SIZE)
        _points = points[inbound]
        _labels = numeric_labels[inbound]

        _points = np.flip(_points, axis=1)

        if len(self.gt_points) == 0:
            self.gt_points = _points
        else:
            self.gt_points = np.concatenate((self.gt_points, _points), axis=0)

        if len(self.gt_labels) == 0:
            self.gt_labels = _labels
        else:
            self.gt_labels = np.concatenate((self.gt_labels, _labels), axis=0)

        _points[:, 0] -= BORDER_SIZE
        _points[:, 1] -= BORDER_SIZE

        filename = image_name.split('.')[0]
        filename = filename + '.png'
        
        sparseImage = self.createSparseImage(_points, _labels, cropped_image.shape)

        # Delete existing dataset with the same name in ML_Superpixels/Datasets
        if os.path.exists("ML_Superpixels/Datasets/"+self.dataset):
            shutil.rmtree("ML_Superpixels/Datasets/"+self.dataset)

        new_filename_path = "ML_Superpixels/Datasets/"+self.dataset + "/sparse_GT/train/"
        if not os.path.exists(new_filename_path):
            os.makedirs(new_filename_path)
        new_filename = new_filename_path + filename

        # Create a new dataset for the images used
        os.makedirs("ML_Superpixels/Datasets/"+self.dataset+"/images/train")
        cv2.imwrite("ML_Superpixels/Datasets/"+self.dataset+"/images/train/"+filename, cropped_image)

        # Save the image
        cv2.imwrite(new_filename, sparseImage)

        generate_augmented_GT(filename,"ML_Superpixels/Datasets/"+self.dataset, default_value=255, number_levels=15, start_n_superpixels=3000, last_n_superpixels=30)

        print("Superpixel expansion done")

        # Load the expanded image in grayscale
        expanded_image = cv2.imread("ML_Superpixels/Datasets/"+self.dataset+ "/augmented_GT/train/" + filename, cv2.IMREAD_GRAYSCALE)
        expanded_image[expanded_image == 255] = 0

        _points[:, 0] += BORDER_SIZE
        _points[:, 1] += BORDER_SIZE

        # Convert the image to dataframe
        expanded_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label"])
        for i in range(1, len(unique_labels_str)+1):
            expanded_points = np.argwhere(expanded_image == i)
            data = []
            for point in expanded_points:
                data.append({
                    "Name": image_name,
                    "Row": point[0],
                    "Column": point[1],
                    "Label": unique_labels_str[i-1]
                })
            new_data_df = pd.DataFrame(data)
            expanded_df = pd.concat([expanded_df, new_data_df], ignore_index=True)

        return expanded_df


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="Directory containing images", required=True)
parser.add_argument("--output_dir", help="Directory to save the output images", required=True)
parser.add_argument("-gt","--ground_truth", help="CSV file containing the points and labels", required=True)
parser.add_argument("--model", help="Model to use for prediction. If superpixel, the ML_Superpixels folder must be at the same path than this script.", default="mixed", choices=["sam", "superpixel", "mixed"])
parser.add_argument("--dataset", help="Dataset to use for superpixel expansion", required=False)
parser.add_argument("--max_distance", help="Maximum distance between expanded points and the seed", type=int)
parser.add_argument("--generate_eval_images", help="Generate evaluation images for the expansion (sparse and expanded images for all the classes)", required=False, action='store_true')
parser.add_argument("--color_dict", help="CSV file containing the color dictionary", required=False)
parser.add_argument("--generate_csv", help="Generate a sparse csv file", required=False, action='store_true')
parser.add_argument("--frame", help="Frame size to crop the images", required=False, type=int, default=0)
parser.add_argument("--gt_images", help="Directory containing the ground truth images. Just for visual comparison", required=False)
parser.add_argument("-b", "--background_class", help="background class value (for grayscale, provide an integer; for color, provide a tuple)", required=True, default=0)
args = parser.parse_args()

remove_far_points = False
generate_eval_images = False
generate_csv = False
color_dict = {}

# Get input points and labels from csv file
input_df = pd.read_csv(args.ground_truth)
output_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label"])

if args.generate_csv:
    generate_csv = True

if args.max_distance:
    MAX_DISTANCE = args.max_distance
    remove_far_points = True

image_path = args.input_dir
print("Image directory:", image_path)
if not os.path.exists(image_path):
    parser.error(f"The directory {image_path} does not exist")

if args.frame:
    BORDER_SIZE = args.frame

if args.gt_images:
    gt_images_path = args.gt_images
    if not os.path.exists(gt_images_path):
        parser.error(f"The directory {gt_images_path} does not exist")

# Get all the images names in the input directory
# image_names = os.listdir(image_dir)
# image_names = [image_name[:-4] for image_name in image_names if image_name[-4:] == '.jpg']

# # Remove all the points of input_df that are not in the images
# input_df = input_df[input_df['Name'].isin(image_names)]

output_dir = args.output_dir
if output_dir[-1] != '/':
    output_dir += '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

unique_labels = input_df['Label'].unique()

if args.generate_eval_images:
    generate_eval_images = True

if args.dataset is None and (args.model == "superpixel" or args.model == "mixed"):
    print("Dataset not provided for selected model. Exiting...")
    sys.exit()

if args.color_dict:
    print("color dictionary provided. Loading color_dict...")
    color_df = pd.read_csv(args.color_dict, header=None)
    keys = color_df.iloc[0].tolist()
    values = color_df.iloc[1:].values.tolist()
    
    # Create the dictionary
    color_dict = {str(keys[i]): [row[i] for row in values] for i in range(len(keys))}
    
    # Get the labels that are in self.color_dict.keys() but not in labels
    extra_labels = set(color_dict.keys()) - set(map(str, unique_labels))

    # Remove the extra labels from color_dict
    for label in extra_labels:
        del color_dict[label]

    assert set(map(str, unique_labels)) == set(color_dict.keys()), (
        'Labels in the .csv file and color_dict do not match:\n'
        f'     Labels in unique_labels but not in color_dict: {set(map(str, unique_labels)) - set(color_dict.keys())}\n'
        f'     Labels in color_dict but not in unique_labels: {set(color_dict.keys()) - set(map(str, unique_labels))}'
    )
        
else:
    if generate_eval_images:
        # Ensure args.color_dict is None
        assert args.color_dict is None, "Expected args.color_dict to be None when generating evaluation images without a provided color dictionary."
    else:
        color_dict = None
        labels = input_df['Label'].unique().tolist()

if args.model == "sam" or args.model == "mixed":
    device = "cuda"
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    model = sam_model_registry["vit_h"](checkpoint="vit_h.pth")
    model.to(device)

    predictor = SamPredictor(model)
    LabelExpander_sam = SAMLabelExpander(color_dict, input_df, unique_labels, output_df, predictor, generate_eval_images)

if args.model == "superpixel" or args.model == "mixed":
    dataset = args.dataset
    LabelExpander_spx = SuperpixelLabelExpander(dataset, color_dict, input_df, unique_labels, output_df)


mask_dir = output_dir + 'labels/'
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

eval_images_dir = output_dir + 'eval_images/'
if not os.path.exists(eval_images_dir):
    os.makedirs(eval_images_dir)

processed_images = 0

image_names_csv = input_df['Name'].unique()

if '.' in image_path.split('/')[-1]:
    image_dir = image_path[:image_path.rfind('/') + 1]
    image_name = image_path.split('/')[-1]
    image_names_csv = [image_name]
else:
    image_dir = image_path
    image_names_dir = os.listdir(image_dir)

# checkMissmatchInFiles(image_names_csv, os.listdir(image_dir))

def get_color_hsv(index, total_colors):
    hue = (index / total_colors) * 360  # Vary hue from 0 to 360 degrees
    saturation = 1.0 if index % 2 == 0 else 0.7  # Alternate saturation levels
    value = 1.0 if index % 3 == 0 else 0.8  # Alternate value levels
    return hue, saturation, value

def hsv_to_rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    hi = int(h / 60.0) % 6
    f = (h / 60.0) - hi
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)

if not isinstance(unique_labels[0], str):
    total_colors = len(unique_labels)
    colors_hsv = [get_color_hsv(i, total_colors) for i in range(total_colors)]
    colors_rgb = [hsv_to_rgb(*color) for color in colors_hsv]

    # Sort colors by HSV values to get similar colors together
    sorted_colors_hsv = sorted(colors_hsv, key=lambda x: (x[0], x[1], x[2]))
    sorted_colors_rgb = [hsv_to_rgb(*color) for color in sorted_colors_hsv]

    # Sort the labels
    sorted_labels = sorted(unique_labels)

    # Create a dictionary to store the color for each unique label
    label_colors = {label: sorted_colors_rgb[i % total_colors] for i, label in enumerate(sorted_labels)}

    # Include the class 0 and assign it a specific color (e.g., black)
    label_colors[0] = (0, 0, 0)

for image_name in image_names_csv:
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if args.gt_images:
        # Get the base name of the image without extension
        base_image_name = os.path.splitext(image_name)[0]
        
        # List all files in the gt_images_path directory
        gt_files = os.listdir(gt_images_path)
        
        # Find the file that matches the base_image_name
        gt_image_file = next((f for f in gt_files if os.path.splitext(f)[0] == base_image_name), None)
        
        if gt_image_file:
            gt_image_path = os.path.join(gt_images_path, gt_image_file)
            gt_image = cv2.imread(gt_image_path)
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        else:
            print(f"ERROR: Ground truth image for {image_name} not found in {gt_images_path}")

    if image is None:
        print(f"ERROR: Failed to load image at {image_path}")
        continue

    print('Expanding labels for', image_name, '...')

    points = input_df[input_df['Name'] == image_name].iloc[:, 1:3].to_numpy().astype(int)
    labels = input_df[input_df['Name'] == image_name].iloc[:, 3].to_numpy()
    unique_labels_str_i = np.unique(labels)

    eval_images_dir_i = eval_images_dir + image_name + '/'

    start_expand = time.time()
    if args.model == "sam":
        output_df = LabelExpander_sam.expand_image(unique_labels_str_i, image, eval_images_dir_i)
    elif args.model == "superpixel":
        output_df = LabelExpander_spx.expand_image(unique_labels_str_i, image, eval_images_dir_i)
    elif args.model == "mixed":
        print("\tExpanding labels with SAM...")
        start_sam = time.time()
        expanded_sam = LabelExpander_sam.expand_image(unique_labels_str_i, image, eval_images_dir_i).drop_duplicates()
        end_sam = time.time()

        start_spx = time.time()
        print("\tExpanding labels with Superpixels...")
        expanded_spx = LabelExpander_spx.expand_image(unique_labels_str_i, image, eval_images_dir_i).drop_duplicates()
        print(f"\tTime taken by SAM: {end_sam - start_sam} seconds")
        print(f"\tTime taken by Superpixels: {time.time() - start_spx} seconds")

        # Remove duplicates from expanded_sam and expanded_spx based on Row and Column
        expanded_sam = expanded_sam.drop_duplicates(subset=["Row", "Column"])
        expanded_spx = expanded_spx.drop_duplicates(subset=["Row", "Column"])

        # Merge expanded_spx with expanded_sam on Row and Column only
        merged_df = expanded_spx.merge(expanded_sam, on=["Row", "Column"], how='left', indicator=True, suffixes=('_spx', '_sam'))

        # Filter merged_df to get points that are only in expanded_spx
        points_not_in_sam = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge', 'Label_sam'])

        # Rename the Label column to match the original
        points_not_in_sam = points_not_in_sam.rename(columns={'Label_spx': 'Label'})

        # Concatenate expanded_sam with points_not_in_sam and remove duplicates based on Row and Column
        output_df = pd.concat([expanded_sam, points_not_in_sam], ignore_index=True).drop_duplicates(subset=["Row", "Column"])

        rgb_flag = color_dict is not None
        background = args.background_class

        color_mask_sam = np.full((image.shape[0], image.shape[1], 3), fill_value=(background, background, background), dtype=np.uint8)
        color_mask_spx = np.full((image.shape[0], image.shape[1], 3), fill_value=(background, background, background), dtype=np.uint8)
        color_mask_mix = np.full((image.shape[0], image.shape[1], 3), fill_value=(background, background, background), dtype=np.uint8)

        mask_color_dir = os.path.join(output_dir, 'labels_mosaic')
        os.makedirs(mask_color_dir, exist_ok=True)

        for label in unique_labels_str_i:
            expanded_i_sam = expanded_sam[expanded_sam['Label'] == label].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
            expanded_i_spx = expanded_spx[expanded_spx['Label'] == label].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
            expanded_i_mix = output_df[output_df['Label'] == label].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE

            if rgb_flag:
                color = np.array(color_dict[str(label)])
            else:
                #grayscale value
                color = label_colors[label]

            color_mask_sam[expanded_i_sam[:, 0], expanded_i_sam[:, 1]] = color
            color_mask_spx[expanded_i_spx[:, 0], expanded_i_spx[:, 1]] = color
            color_mask_mix[expanded_i_mix[:, 0], expanded_i_mix[:, 1]] = color
        
        if args.gt_images:
            if not rgb_flag:
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
        
                # Create an empty RGB image
                gt_image_rgb = np.zeros((gt_image.shape[0], gt_image.shape[1], 3), dtype=np.uint8)
                
                # Assign each pixel of gt_image to the corresponding color in label_colors
                for label in np.unique(gt_image):
                    color = label_colors.get(label, (0, 0, 0))  # Default to black if label not found
                    # print(f"Label {label} -> Color {color}")
                    gt_image_rgb[gt_image == label] = color
                
                gt_image = gt_image_rgb  # Replace the grayscale image with the RGB image
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            im = axs[0].imshow(gt_image)
            axs[0].set_title("Ground Truth")
            axs[0].axis('off')
            axs[1].imshow(color_mask_sam)
            axs[1].set_title("SAM")
            axs[1].axis('off')
            axs[2].imshow(color_mask_spx)
            axs[2].set_title("Superpixels")
            axs[2].axis('off')
            axs[3].imshow(color_mask_mix)
            axs[3].set_title("Mixed")
            axs[3].axis('off')
        else:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            im = axs[0].imshow(image)
            axs[0].set_title("Original image")
            axs[0].axis('off')
            axs[1].imshow(color_mask_sam)
            axs[1].set_title("SAM")
            axs[1].axis('off')
            axs[2].imshow(color_mask_spx)
            axs[2].set_title("Superpixels")
            axs[2].axis('off')
            axs[3].imshow(color_mask_mix)
            axs[3].set_title("Mixed")
            axs[3].axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        plt.savefig(os.path.join(mask_color_dir, os.path.splitext(image_name)[0] + '.png'), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    print(f"Time taken by expand_labels: {time.time() - start_expand} seconds")
    processed_images += 1
    print(f"{processed_images}/{len(image_names_csv)}\n")

    background = args.background_class

    # Initialize the mask with the background value
    mask = np.full((image.shape[0], image.shape[1]), background, dtype=np.uint8)

    if color_dict is not None:
        color_mask = np.full((image.shape[0], image.shape[1], 3), fill_value=(64, 0, 64), dtype=np.uint8)
        mask_color_dir = os.path.join(output_dir, 'labels_rgb')
        os.makedirs(mask_color_dir, exist_ok=True)

        for i, label in enumerate(unique_labels_str_i, start=0):
            expanded_i = output_df[output_df['Label'] == label].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
            color = np.array(color_dict[str(label)])
            gray = np.clip(i, 0, 255).astype(np.uint8)
            mask[expanded_i[:, 0], expanded_i[:, 1]] = gray
            color_mask[expanded_i[:, 0], expanded_i[:, 1]] = color

        # Save color mask as PNG
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(mask_color_dir, os.path.splitext(image_name)[0] + '.png'), color_mask)
        unique_labels_int = [int(i) for i in unique_labels_str_i]
    else:
        unique_labels_int = [int(i) for i in labels]

    for label in unique_labels_int:
        expanded_i = output_df[output_df['Label'] == label].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
        label = np.clip(label, 0, 255).astype(np.uint8)
        mask[expanded_i[:, 0], expanded_i[:, 1]] = label

    # Save grayscale mask as PNG
    cv2.imwrite(os.path.join(mask_dir, os.path.splitext(image_name)[0] + '.png'), mask)

    if generate_csv:
        LabelExpander.generate_csv()

print('Images expanded!')
