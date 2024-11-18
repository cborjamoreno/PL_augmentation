
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
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, label
from scipy.spatial import KDTree

from ML_Superpixels.generate_augmented_GT.generate_augmented_GT import generate_augmented_GT

# os.environ['TORCH_CUDNN_SDPA_ENABLED'] = '1'
# print(f"TORCH_CUDNN_SDPA_ENABLED: {os.environ['TORCH_CUDNN_SDPA_ENABLED']}")

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
    
def show_points(coords, labels, ax, marker_size=300, marker_color='blue', edge_color='white'):
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

    # Assign the blended color to the pixel
    img[row, column] = blended_color


def chunk_dataframe(df, chunk_size):
    """Generator to yield chunks of the DataFrame."""
    for start in range(0, df.shape[0], chunk_size):
        yield df.iloc[start:start + chunk_size]


def gather_gt_points_from_segment_area(region_coords, gt_points, gt_labels, segment_label):
    """
    Gather ground truth points that are within the region_coords and match the segment_label.
    """
    region_coords_set = set(map(tuple, region_coords))
    segment_coords = set(map(tuple, gt_points))
    intersect_coords = region_coords_set & segment_coords

    filtered_gt_points = []
    filtered_gt_labels = []

    for coord in intersect_coords:
        idx = np.where((gt_points[:, 0] == coord[0]) & (gt_points[:, 1] == coord[1]))[0][0]
        if gt_labels[idx] == segment_label:
            filtered_gt_points.append([coord[0], coord[1]])
            filtered_gt_labels.append(segment_label)

    return np.array(filtered_gt_points), filtered_gt_labels

def calculate_closest_label(expanded_label_points, gt_points, gt_labels):
    """
    Calculate the label for each expanded point based on the closest GT point.
    """
    if len(gt_points) == 0:
        return np.zeros(len(expanded_label_points), dtype=int)  # No GT points, assign default label
    
    # Calculate distances from each expanded point to all GT points
    distances = cdist(expanded_label_points, gt_points)
    
    # Find the index of the closest GT point for each expanded point
    closest_gt_idx = np.argmin(distances, axis=1)
    
    # Get the label of the closest GT point
    closest_labels = gt_labels[closest_gt_idx]
    
    return closest_labels

def resolve_overlaps(overlapping_pixels, combined_gt_points, combined_gt_labels,
                     _, __):
    """
    Resolve overlapping region by assigning each pixel the label of the closest GT point.
    """
    expanded_label_points = overlapping_pixels[['Row', 'Column']].values.astype(float)

    combined_gt_points = np.array(combined_gt_points)
    if combined_gt_points.ndim == 1:
        combined_gt_points = combined_gt_points.reshape(-1, 2)

    # Calculate the closest label for each pixel based on GT points
    resolved_labels = calculate_closest_label(expanded_label_points, combined_gt_points, combined_gt_labels)

    # Update the labels in the overlap region
    overlapping_pixels['Label'] = resolved_labels

    return overlapping_pixels

def merge_labels(image_df, gt_points, gt_labels):
    """
    Merge overlapping segments by assigning the majority class of GT points in overlap zones.
    In case of a tie, the label of the closest GT point to the overlap centroid is assigned.
    """
    merged_df = pd.DataFrame(columns=['Name', 'Row', 'Column', 'Label'])

    segments = image_df.groupby('Segment')
    compared_pairs = set()

    segment_cache = {}
    for segment_id, segment_data in segments:
        segment_cache[segment_id] = segment_data

    # Calculate bounding boxes for all segments
    bounding_boxes = {}
    for segment_id, segment_data in segments:
        x_min, y_min = segment_data[['Row', 'Column']].min()
        x_max, y_max = segment_data[['Row', 'Column']].max()
        bounding_boxes[segment_id] = (x_min, y_min, x_max, y_max)

    # Track overlaps
    overlap_dict = {}

    # Compare segments
    for segment_id, segment_data in segments:
        bounding_box_1 = bounding_boxes[segment_id]

        for other_segment_id, other_segment_data in segments:
            if other_segment_id == segment_id:
                continue

            bounding_box_2 = bounding_boxes[other_segment_id]

            # Check for bounding box overlap
            if (bounding_box_1[0] > bounding_box_2[2] or  # left of other
                bounding_box_1[2] < bounding_box_2[0] or  # right of other
                bounding_box_1[1] > bounding_box_2[3] or  # above other
                bounding_box_1[3] < bounding_box_2[1]):   # below other
                continue  # No overlap in bounding boxes, skip pixel check

            # Create a pair of segment IDs
            pair = tuple(sorted((segment_id, other_segment_id)))

            # Skip if this pair has already been compared
            if pair in compared_pairs:
                continue

            # Check for pixel overlap directly
            overlap_pixels = segment_data.merge(other_segment_data, on=['Row', 'Column'], how='inner')

            if not overlap_pixels.empty:
                # Add to the overlap dictionary by pixel coordinate
                overlap_coords = overlap_pixels[['Row', 'Column']].values
                for coord in overlap_coords:
                    coord_tuple = tuple(coord)
                    if coord_tuple not in overlap_dict:
                        overlap_dict[coord_tuple] = set()
                    overlap_dict[coord_tuple].update([segment_id, other_segment_id])

            compared_pairs.add(pair)

    # Group pixels by the specific set of segments that overlap
    segment_overlap_dict = {}
    for coord, segs in overlap_dict.items():
        segments_tuple = tuple(sorted(segs))
        if segments_tuple not in segment_overlap_dict:
            segment_overlap_dict[segments_tuple] = []
        segment_overlap_dict[segments_tuple].append(coord)

    # Sort the segment_overlap_dict by the number of overlapping segments in descending order
    sorted_segment_overlap_dict = sorted(segment_overlap_dict.items(), key=lambda x: len(x[0]), reverse=True)

    resolved_pixels = set()
    all_regions = []

    # Pre-calculate distances for all GT points
    gt_points = np.array(gt_points)
    distance_cache = {}

    # Resolve overlaps for each group of overlapping segments
    for segments_tuple, coords in sorted_segment_overlap_dict:
        region_coords = np.array(coords)
        region_segments = set(segments_tuple)

        # Check if region_coords is not empty
        if len(region_coords) == 0:
            continue

        num_gt_per_seg = []
        gt_label_per_seg = []

        # Dictionary to hold GT points for each segment in the overlapping region
        segment_id_to_gt_points = {}

        for segment_id in region_segments:
            segment_data = segments.get_group(segment_id)
            segment_label = segment_data['Label'].iloc[0]

            # Gather GT points from the overlap region for this segment
            segment_gt_points, segment_gt_labels = gather_gt_points_from_segment_area(
                region_coords, gt_points, gt_labels, segment_label
            )

            num_gt_per_seg.append(len(segment_gt_points))
            if len(segment_gt_points) == 0:
                gt_label_per_seg.append(-1)
            else:
                gt_label_per_seg.append(segment_label)
                segment_id_to_gt_points[segment_id] = segment_gt_points  # Store GT points for each segment

        # Determine the majority label or resolve ties
        if num_gt_per_seg.count(max(num_gt_per_seg)) == 1:
            majority_label = gt_label_per_seg[num_gt_per_seg.index(max(num_gt_per_seg))]
        else:
            # In case of a tie, find the GT point closest to the overlap centroid
            combined_gt_points = []

            # Step 1: Gather GT points from the overlap area for each segment in the region
            for segment_id in region_segments:
                if segment_id in segment_id_to_gt_points:
                    combined_gt_points.append(segment_id_to_gt_points[segment_id])
                    # print('segment_id:', segment_id, 'num_gt_per_seg:', num_gt_per_seg, 'gt_label_per_seg:', gt_label_per_seg)
                    # print('combined_gt_points:', combined_gt_points)
            # if combined_gt_points:
            #     print('num_gt_per_seg:', num_gt_per_seg, 'gt_label_per_seg:', gt_label_per_seg)

            # Step 2: If no GT points in the overlap area, gather GT points from the entire segments
            if not combined_gt_points:
                for segment_id in region_segments:
                    segment_data = segment_cache[segment_id]
                    segment_gt_points, segment_gt_labels = gather_gt_points_from_segment_area(
                        segment_data[['Row', 'Column']].values, gt_points, gt_labels, segment_label
                    )
                    if len(segment_gt_points) > 0:
                        combined_gt_points.append(segment_gt_points)

            # Step 3: If still no GT points, skip this region
            if not combined_gt_points:
                continue

            # Combine all GT points from the overlap or entire segments
            combined_gt_points = np.vstack(combined_gt_points)

            # print('combined_gt_points:', combined_gt_points)

            # Compute the overlap centroid
            overlap_centroid = np.mean(region_coords, axis=0)

            # Step 4: Calculate distances between the centroid and combined GT points
            centroid_tuple = tuple(overlap_centroid)
            if centroid_tuple in distance_cache:
                if len(distance_cache[centroid_tuple]) != len(combined_gt_points):
                    distances = np.linalg.norm(combined_gt_points - overlap_centroid, axis=1)
                    distance_cache[centroid_tuple] = distances
                else:
                    distances = distance_cache[centroid_tuple]
            else:
                distances = np.linalg.norm(combined_gt_points - overlap_centroid, axis=1)
                distance_cache[centroid_tuple] = distances

            # print('centroid:', overlap_centroid)
            # print('distance of centroid to GT points:', distances)

            # Assign the label of the closest GT point
            closest_gt_index = np.argmin(distances)
            closest_gt_point = combined_gt_points[closest_gt_index]

            # Find the index of this closest GT point in the original gt_points to get the label
            closest_gt_label = gt_labels[np.where((gt_points == closest_gt_point).all(axis=1))[0][0]]

            majority_label = closest_gt_label

            # print('closest_gt_index:', closest_gt_index, 'majority_label:', majority_label)

        # Assign the label to all pixels in the overlap zone
        region_df = pd.DataFrame(region_coords, columns=['Row', 'Column'])
        region_df['Label'] = majority_label
        all_regions.append(region_df)

        # Mark the pixels in the region as resolved
        resolved_pixels.update(map(tuple, region_coords))

    merged_df = pd.concat([merged_df] + all_regions, ignore_index=True)

    # After resolving overlaps, handle non-overlapping pixels
    existing_pixels = set(map(tuple, merged_df[['Row', 'Column']].itertuples(index=False)))

    # Create a list to collect non-overlapping pixels
    non_overlapping_pixels_list = []

    for segment_id, segment_data in segments:
        non_overlapping_pixels = segment_data[
            ~segment_data[['Row', 'Column']].apply(tuple, axis=1).map(lambda x: x in existing_pixels)
        ]
        non_overlapping_pixels_list.append(non_overlapping_pixels)

    # Concatenate all non-overlapping pixels at once
    merged_df = pd.concat([merged_df] + non_overlapping_pixels_list, ignore_index=True)

    return merged_df

def generate_image(image_df, image, gt_points, color_dict, image_name, output_dir, label_str=None):
    if label_str is None:
        label_str = 'ALL'

    # Create a black image
    black = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    black = image.copy()
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

            # print('aaaa',str(row['Label']))

            # Get the color and add an alpha channel
            color = np.array([int(color_dict[str(row['Label'])][0]) / 255.0, int(color_dict[str(row['Label'])][1]) / 255.0, int(color_dict[str(row['Label'])][2]) / 255.0, 1])
            blend_translucent_color(black, point[0], point[1], color, color[3])
    # print(f"Time taken by color the points in the image: {time.time() - start} seconds")

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(black)
    show_points(gt_points, np.ones(len(gt_points), dtype=int), plt.gca(), marker_color='black', edge_color='yellow')
    plt.axis('off')

    plt.savefig(output_dir + image_name + '_' + label_str + '_expanded.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    # print('Image saved in', output_dir + image_name + '_' + label_str + '_expanded.png')

def generate_image_per_class(image_df, image, points, labels, color_dict, image_name, output_dir, label_str):

    # if label_str is not a string convert it to a string
    if not isinstance(label_str, str):
        label_str = str(label_str)
    
    # print(f"Generating images for class {label_str}")
    
    # gt_points_sortened = points[np.argsort(points[:, 0])]
    # print(f"gt_points_sortened: {gt_points_sortened}")
    
    # Create a black image
    black = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    black = image.copy()
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

    # Color the points in the image
    for _, row in image_df.iterrows():
        if row['Column'] < black.shape[1] and row['Row'] < black.shape[0]:
            point = (row['Row'] + BORDER_SIZE, row['Column'] + BORDER_SIZE)

            # Get the color and add an alpha channel
            color = np.array([int(color_dict[str(row['Label'])][0]) / 255.0, int(color_dict[str(row['Label'])][1]) / 255.0, int(color_dict[str(row['Label'])][2]) / 255.0, 1])
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
    black = image.copy()
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

    def expand_image(self, unique_labels_i, unique_labels_str_i, image, background_class, eval_images_dir_i):      
        expanded_df = self.expand_labels(points, labels, unique_labels_i, unique_labels_str_i, image, image_name, background_class, eval_images_dir_i)
        
        if self.generate_eval_images and isinstance(self, SAMLabelExpander):
            generate_image(expanded_df, image, self.gt_points, self.color_dict, image_name, eval_images_dir_i)

        if self.generate_csv:
            pass

        return expanded_df
        
    @abstractmethod
    def expand_labels(self, points, labels, unique_labels_i, unique_labels_str_i, image, image_name, background_class=None, eval_image_dir=None):
        pass

class SAMLabelExpander(LabelExpander):
    def __init__(self, color_dict, input_df, labels, output_df, predictor, generate_eval_images=False):
        super().__init__(color_dict, input_df, labels, output_df)
        self.predictor = predictor
        self.generate_eval_images = generate_eval_images

    def expand_labels(self, points, labels, unique_labels, unique_labels_str, image, image_name, background_class=None, eval_image_dir=None):
        expanded_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label", "Segment"])
        
        time_start = time.time()

        # Initialize the segment counter
        segment_counter = 0

        self.gt_points = np.array([])
        self.gt_labels = np.array([])

        # Crop the image if BORDER_SIZE > 0, otherwise use the full image
        if BORDER_SIZE > 0:
            cropped_image = image[BORDER_SIZE:image.shape[0]-BORDER_SIZE, BORDER_SIZE:image.shape[1]-BORDER_SIZE]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        else:
            cropped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(cropped_image)

        # Store points and labels within borders if BORDER_SIZE > 0
        filtered_points_all = []
        filtered_labels_all = []

        for i in range(len(unique_labels)):

            if unique_labels_str[i] == background_class:
                # print(f"Skipping background class {background}")
                continue
            
            filtered_points = points[labels == unique_labels[i]]
            filtered_labels = np.array(labels)[labels == unique_labels[i]]

            if BORDER_SIZE > 0:
                # Create a mask for points within the image borders
                inbound = (filtered_points[:, 0] > BORDER_SIZE) & (filtered_points[:, 0] < image.shape[0] - BORDER_SIZE) & (filtered_points[:, 1] > BORDER_SIZE) & (filtered_points[:, 1] < image.shape[1] - BORDER_SIZE)
                _points = filtered_points[inbound]
                _labels = filtered_labels[inbound]
            else:
                _points = filtered_points
                _labels = filtered_labels

            if len(_points) == 0:
                # print(f"No points for label {unique_labels[i]}")
                continue

            # Store filtered points and labels for the loop
            filtered_points_all.append(_points)
            filtered_labels_all.append(_labels)

        all_points = np.concatenate(filtered_points_all, axis=0)
        all_labels = np.concatenate(filtered_labels_all, axis=0)

        # Randomly select 5% of the points
        num_points = len(all_points)
        num_to_select = int(num_points * 0.0)  # Select 5% of points
        selected_indices = np.random.choice(num_points, num_to_select, replace=False)

        # Filter points and labels based on selected indices
        all_points = all_points[selected_indices]
        all_labels = all_labels[selected_indices]

        # Now expand each label's points
        for i, (_points, _labels) in enumerate(zip(filtered_points_all, filtered_labels_all)):

            _points = np.flip(_points, axis=1)

            if len(self.gt_points) == 0:
                self.gt_points = _points
            else:
                self.gt_points = np.concatenate((self.gt_points, _points), axis=0)

            # Transform the points after cropping if BORDER_SIZE > 0
            _labels_ones = np.ones(len(_points), dtype=int)

            if len(self.gt_labels) == 0:
                self.gt_labels = _labels
            else:
                self.gt_labels = np.concatenate((self.gt_labels, _labels), axis=0)

            if BORDER_SIZE > 0:
                _points_pred = _points.copy()
                _points_pred[:, 0] -= BORDER_SIZE
                _points_pred[:, 1] -= BORDER_SIZE
            else:
                _points_pred = _points.copy()

            data = []
            expanded_points_set = set()  # Optimized existence check

            for p, l in zip(_points_pred, _labels_ones):

                point_row, point_column = p[1], p[0]
                label_str = unique_labels[i]

                # Filter points where the corresponding label is not equal to label_str
                other_label_gt_points = [point for point, label in zip(all_points, all_labels) if label != label_str]
                
                # Convert other_label_gt_points to a 2D array
                other_label_gt_points = np.array(other_label_gt_points).reshape(-1, 2)
                
                # Create a list of zeros of the same length as other_label_gt_points
                zero_list = [0] * len(other_label_gt_points)

                if (point_row, point_column, label_str) in expanded_points_set:
                    continue

                # Concatenate the current point 'p' with 'other_label_gt_points'
                point_coords = np.concatenate((np.array([p]), other_label_gt_points), axis=0)
                
                # Concatenate the current label 'l' with the 'zero_list'
                point_labels = np.concatenate((np.array([l]), np.array(zero_list)), axis=0)

                # print('point_coords:', point_coords)
                # print('point_labels:', point_labels)

                _, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

                mask_input = logits[0, :, :]  # Choose the model's best mask
                # mask_input = logits[np.argmax(scores), :, :]  # Choose the mask with the highest score
                mask, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    mask_input=mask_input[None, :, :],
                    multimask_output=False,
                )

                segment_counter += 1

                # Determine new points for the current mask
                new_points = np.argwhere(mask[0])

                for point in new_points:
                    if (point[0], point[1], label_str) not in expanded_points_set:
                        expanded_points_set.add((point[0], point[1], label_str))
                        data.append({
                            "Name": image_name,
                            "Row": point[0],
                            "Column": point[1],
                            "Label": label_str,
                            "Segment": segment_counter
                        })

            if BORDER_SIZE > 0:
                _points_pred[:, 0] += BORDER_SIZE
                _points_pred[:, 1] += BORDER_SIZE

            new_data_df = pd.DataFrame(data)
            
            if self.generate_eval_images:
                generate_image_per_class(new_data_df, image, _points_pred, _labels_ones, color_dict, image_name, eval_image_dir, unique_labels[i])
            
            expanded_df = pd.concat([expanded_df, new_data_df], ignore_index=True)
            # print(f'{len(_points_pred)} points of class \'{unique_labels_str[i]}\' expanded to {len(new_points)} points')
        
        # print(f"Time taken by expand_labels: {time.time() - time_start} seconds")

        # Merge the dense labels
        if BORDER_SIZE > 0:
            gt_points = self.gt_points - BORDER_SIZE
            gt_points = np.flip(gt_points, axis=1)
        else:
            gt_points = np.flip(self.gt_points, axis=1)

        gt_labels = self.gt_labels

        time_merge_labels = time.time()
        merged_df = merge_labels(expanded_df, gt_points, self.gt_labels)
        # print(f"Time taken by merge_labels: {time.time() - time_merge_labels} seconds")

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

    def expand_labels(self, points, labels, unique_labels, unique_labels_str, image, image_name, background_class=None, eval_image_dir=None):

        # crop the image BORDER_SIZE pixels from each side
        cropped_image = image[BORDER_SIZE:image.shape[0]-BORDER_SIZE, BORDER_SIZE:image.shape[1]-BORDER_SIZE]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        numeric_labels = np.zeros(len(labels), dtype=int)
        for i, label in enumerate(unique_labels, start=1):
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

        # print("Superpixel expansion done")

        # Load the expanded image in grayscale
        expanded_image = cv2.imread("ML_Superpixels/Datasets/"+self.dataset+ "/augmented_GT/train/" + filename, cv2.IMREAD_GRAYSCALE)
        expanded_image[expanded_image == 255] = 0

        _points[:, 0] += BORDER_SIZE
        _points[:, 1] += BORDER_SIZE

        # Convert the image to dataframe
        expanded_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label"])
        for i in range(1, len(unique_labels)+1):
            expanded_points = np.argwhere(expanded_image == i)
            data = []
            for point in expanded_points:
                data.append({
                    "Name": image_name,
                    "Row": point[0],
                    "Column": point[1],
                    "Label": unique_labels[i-1]
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
    # print('color_dict:', color_dict)
    
    # Get the labels that are in self.color_dict.keys() but not in labels
    extra_labels = set(color_dict.keys()) - set(map(str, unique_labels))

    # Remove the extra labels from color_dict
    # for label in extra_labels:
    #     del color_dict[label]

    # assert set(map(str, unique_labels)) == set(color_dict.keys()), (
    #     'Labels in the .csv file and color_dict do not match:\n'
    #     f'     Labels in unique_labels but not in color_dict: {set(map(str, unique_labels)) - set(color_dict.keys())}\n'
    #     f'     Labels in color_dict but not in unique_labels: {set(color_dict.keys()) - set(map(str, unique_labels))}'
    # )
else:
    if generate_eval_images:
        # Ensure args.color_dict is None
        assert args.color_dict is None, "Expected args.color_dict to be None when generating evaluation images without a provided color dictionary."
    else:
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

# Initialize lists to store execution times
sam_times = []
spx_times = []

# Initialize progress bar
with tqdm(total=len(image_names_csv), desc="Processing images") as pbar:
    for image_name in image_names_csv:
        pbar.set_description(f"Processing {image_name}")
        
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: Failed to load image at {image_path}")
            pbar.update(1)
            continue
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
            pbar.update(1)
            continue

        points = input_df[input_df['Name'] == image_name].iloc[:, 1:3].to_numpy().astype(int)
        labels = input_df[input_df['Name'] == image_name].iloc[:, 3].to_numpy()
        unique_labels_i = np.unique(labels)
        unique_labels_str_i = unique_labels_i.astype(str)

        eval_images_dir_i = eval_images_dir + image_name + '/'

        background = args.background_class

        start_expand = time.time()
        if args.model == "sam":
            start_sam = time.time()
            output_df = LabelExpander_sam.expand_image(unique_labels_i, unique_labels_str_i, image, background_class=background, eval_images_dir_i=eval_images_dir_i)
            end_sam = time.time()
            sam_times.append(end_sam - start_sam)
        elif args.model == "superpixel":
            start_spx = time.time()
            output_df = LabelExpander_spx.expand_image(unique_labels_i, unique_labels_str_i, image, background_class=background, eval_images_dir_i=eval_images_dir_i)
            end_spx = time.time()
            spx_times.append(end_spx - start_spx)
        elif args.model == "mixed":
            start_sam = time.time()
            expanded_sam = LabelExpander_sam.expand_image(unique_labels_i, unique_labels_str_i, image, background_class=background, eval_images_dir_i=eval_images_dir_i)
            end_sam = time.time()
            sam_times.append(end_sam - start_sam)

            start_spx = time.time()
            expanded_spx = LabelExpander_spx.expand_image(unique_labels_i, unique_labels_str_i, image, background_class=background, eval_images_dir_i=eval_images_dir_i)
            end_spx = time.time()
            spx_times.append(end_spx - start_spx)

            merged_df = expanded_spx.merge(expanded_sam, on=["Row", "Column"], how='left', indicator=True, suffixes=('_spx', '_sam'))
            points_not_in_sam = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge', 'Label_sam'])
            points_not_in_sam = points_not_in_sam.rename(columns={'Label_spx': 'Label'})

            output_df = pd.concat([expanded_sam, points_not_in_sam], ignore_index=True).drop_duplicates(subset=["Row", "Column"])

            rgb_flag = color_dict is not None

            # background_gray is the index of the background label in the color_dict
            background_gray = list(color_dict.keys()).index(background)
            background_color = color_dict.get(background, (background, background, background))

            # Create color masks filled with the background color
            color_mask_sam = np.full((image.shape[0], image.shape[1], 3), fill_value=background_color, dtype=np.uint8)
            color_mask_spx = np.full((image.shape[0], image.shape[1], 3), fill_value=background_color, dtype=np.uint8)
            color_mask_mix = np.full((image.shape[0], image.shape[1], 3), fill_value=background_color, dtype=np.uint8)

            mask_color_dir = os.path.join(output_dir, 'labels_mosaic')
            os.makedirs(mask_color_dir, exist_ok=True)

            for label in unique_labels_i:
                expanded_i_sam = expanded_sam[expanded_sam['Label'] == label].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
                expanded_i_spx = expanded_spx[expanded_spx['Label'] == label].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
                expanded_i_mix = output_df[output_df['Label'] == label].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE

                if rgb_flag:
                    color = np.array(color_dict[str(label)])
                else:
                    color = label_colors[label]

                color_mask_sam[expanded_i_sam[:, 0], expanded_i_sam[:, 1]] = color
                color_mask_spx[expanded_i_spx[:, 0], expanded_i_spx[:, 1]] = color
                color_mask_mix[expanded_i_mix[:, 0], expanded_i_mix[:, 1]] = color

            if args.gt_images:
                if not rgb_flag:
                    gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
                    gt_image_rgb = np.zeros((gt_image.shape[0], gt_image.shape[1], 3), dtype=np.uint8)
                    for label in np.unique(gt_image):
                        color = label_colors.get(label, (0, 0, 0))
                        gt_image_rgb[gt_image == label] = color
                    gt_image = gt_image_rgb

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

            # Save label images for SAM and Mixed approaches
            mask_dir_sam = os.path.join(output_dir, 'labels_sam')
            mask_dir_spx = os.path.join(output_dir, 'labels_spx')
            mask_dir_mix = os.path.join(output_dir, 'labels_mix')
            os.makedirs(mask_dir_sam, exist_ok=True)
            os.makedirs(mask_dir_spx, exist_ok=True)
            os.makedirs(mask_dir_mix, exist_ok=True)

            # Initialize the masks with the background value
            mask_sam = np.full((image.shape[0], image.shape[1]), background_gray, dtype=np.uint8)
            mask_spx = np.full((image.shape[0], image.shape[1]), background_gray, dtype=np.uint8)
            mask_mix = np.full((image.shape[0], image.shape[1]), background_gray, dtype=np.uint8)

            for l, l_str in zip(unique_labels_i, unique_labels_str_i):
                expanded_i_sam = expanded_sam[expanded_sam['Label'] == l].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
                expanded_i_spx = expanded_spx[expanded_spx['Label'] == l].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
                expanded_i_mix = output_df[output_df['Label'] == l].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
                label = list(color_dict.keys()).index(l_str)
                mask_sam[expanded_i_sam[:, 0], expanded_i_sam[:, 1]] = label
                mask_spx[expanded_i_spx[:, 0], expanded_i_spx[:, 1]] = label
                mask_mix[expanded_i_mix[:, 0], expanded_i_mix[:, 1]] = label

            # Save grayscale masks as PNG
            cv2.imwrite(os.path.join(mask_dir_sam, os.path.splitext(image_name)[0] + '.png'), mask_sam)
            cv2.imwrite(os.path.join(mask_dir_spx, os.path.splitext(image_name)[0] + '.png'), mask_spx)
            cv2.imwrite(os.path.join(mask_dir_mix, os.path.splitext(image_name)[0] + '.png'), mask_mix)

        # Update progress bar
        pbar.update(1)

        # Initialize the mask with the background value
        mask = np.full((image.shape[0], image.shape[1]), background_gray, dtype=np.uint8)

        if color_dict is not None:
            background_color = color_dict.get(background, (background, background, background))
            color_mask = np.full((image.shape[0], image.shape[1], 3), fill_value=background_color, dtype=np.uint8)
            mask_color_dir = os.path.join(output_dir, 'labels_rgb')
            os.makedirs(mask_color_dir, exist_ok=True)

            for i, label in enumerate(unique_labels_i, start=0):
                expanded_i = output_df[output_df['Label'] == label].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
                color = np.array(color_dict[str(label)])
                if isinstance(label, str):
                    gray = list(color_dict.keys()).index(label)
                else:
                    gray = np.clip(i, 0, 255).astype(np.uint8)
                mask[expanded_i[:, 0], expanded_i[:, 1]] = gray
                color_mask[expanded_i[:, 0], expanded_i[:, 1]] = color

            # Save color mask as PNG
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(mask_color_dir, os.path.splitext(image_name)[0] + '.png'), color_mask)
        #     unique_labels_int = [list(color_dict.keys()).index(i) for i in unique_labels_i]
        # else:
        #     unique_labels_int = [list(color_dict.keys()).index(i) for i in labels]

        for label in unique_labels_i:
            expanded_i = output_df[output_df['Label'] == label].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
            if isinstance(label, str):
                gray = list(color_dict.keys()).index(label)
            else:
                gray = np.clip(i, 0, 255).astype(np.uint8)
            mask[expanded_i[:, 0], expanded_i[:, 1]] = gray

        # Save grayscale mask as PNG
        cv2.imwrite(os.path.join(mask_dir, os.path.splitext(image_name)[0] + '.png'), mask)

        if generate_csv:
            LabelExpander.generate_csv()

# Calculate mean and standard deviation for SAM and Superpixels
if sam_times:
    mean_sam_time = np.mean(sam_times)
    std_sam_time = np.std(sam_times)
    print(f"Mean execution time for SAM: {mean_sam_time:.2f} seconds (std: {std_sam_time:.2f})")

if spx_times:
    mean_spx_time = np.mean(spx_times)
    std_spx_time = np.std(spx_times)
    print(f"Mean execution time for Superpixels: {mean_spx_time:.2f} seconds (std: {std_spx_time:.2f})")

print('Images expanded!')
