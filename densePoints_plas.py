
import argparse
import shutil
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys, os
import pandas as pd
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod
import time
import torch
import torchvision
from tqdm import tqdm
from scipy.ndimage import label
from shapely.validation import make_valid
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from ML_Superpixels.generate_augmented_GT.generate_augmented_GT import generate_augmented_GT

BORDER_SIZE = 0

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
    
def show_points(coords, labels, ax, marker_size=30, marker_color='blue', edge_color='white'):
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

    def expand_labels(self, points, labels, unique_labels, unique_labels_str, image, image_name, background_class=None, eval_image_dir=None):

        sigma_xy = 0.631
        sigma_cnn = 0.5534
        alpha = 1140

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

        filename = image_name.split('.')[0]
        filename = filename + '.png'


        def members_from_clusters(sigma_val_xy, sigma_val_cnn, XY_features, CNN_features, clusters):
            B, K, _ = clusters.shape
            sigma_array_xy = torch.full((B, K), sigma_val_xy, device=device)
            sigma_array_cnn = torch.full((B, K), sigma_val_cnn, device=device)
            
            clusters_xy = clusters[:,:,0:2]
            dist_sq_xy = torch.cdist(XY_features, clusters_xy)**2

            clusters_cnn = clusters[:,:,2:]
            dist_sq_cnn = torch.cdist(CNN_features, clusters_cnn)**2

            soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * sigma_array_cnn**2)) , dim = 2)                # shape = [B, N, K] 
            
            return soft_memberships

        def enforce_connectivity(hard, H, W, K_max, connectivity = True):
            # INPUTS
            # 1. posteriors:    shape = [B, N, K]
            B = 1

            hard_assoc = torch.unsqueeze(hard, 0).detach().cpu().numpy()                                 # shape = [B, N]
            hard_assoc_hw = hard_assoc.reshape((B, H, W))    

            segment_size = (H * W) / (int(K_max) * 1.0)

            min_size = int(0.06 * segment_size)
            max_size = int(H*W*10)

            hard_assoc_hw = hard_assoc.reshape((B, H, W))
            
            for b in range(hard_assoc.shape[0]):
                if connectivity:
                    spix_index_connect = _enforce_label_connectivity_cython(hard_assoc_hw[None, b, :, :], min_size, max_size, 0)[0]
                else:
                    spix_index_connect = hard_assoc_hw[b,:,:]

            return spix_index_connect

        class CustomLoss(nn.Module):
            def __init__(self, clusters_init, N, XY_features, CNN_features, features_cat, labels, sigma_val_xy = 0.5, sigma_val_cnn = 0.5, alpha = 1, num_pixels_used = 1000):
                super(CustomLoss, self).__init__()
                self.alpha = alpha # Weighting for the distortion loss
                self.clusters=nn.Parameter(clusters_init, requires_grad=True)   # clusters (torch.FloatTensor: shape = [B, K, C])
                B, K, _ = self.clusters.shape

                self.N = N

                self.sigma_val_xy = sigma_val_xy
                self.sigma_val_cnn = sigma_val_cnn

                self.sigma_array_xy = torch.full((B, K), self.sigma_val_xy, device=device)
                self.sigma_array_cnn = torch.full((B, K), self.sigma_val_cnn, device=device)

                self.XY_features = XY_features
                self.CNN_features = CNN_features
                self.features_cat = features_cat

                self.labels = labels
                self.num_pixels_used = num_pixels_used

            def forward(self):
                # computes the distortion loss of the superpixels and also our novel conflict loss
                #
                # INPUTS:
                # 1) features:      (torch.FloatTensor: shape = [B, N, C]) defines for each image the set of pixel features

                # B is the batch dimension
                # N is the number of pixels
                # K is the number of superpixels

                # RETURNS:
                # 1) sum of distortion loss and conflict loss scaled by alpha (we use lambda in the paper but this means something else when coding)
                indexes = torch.randperm(self.N)[:self.num_pixels_used]

                ##################################### DISTORTION LOSS #################################################
                # Calculate the distance between pixels and superpixel centres by expanding our equation: (a-b)^2 = a^2-2ab+b^2 
                features_cat_select = self.features_cat[:,indexes,:]
                dist_sq_cat = torch.cdist(features_cat_select, self.clusters)**2

                # XY COMPONENT
                clusters_xy = self.clusters[:,:,0:2]

                XY_features_select = self.XY_features[:,indexes,:]
                dist_sq_xy = torch.cdist(XY_features_select, clusters_xy)**2

                # CNN COMPONENT
                clusters_cnn = self.clusters[:,:,2:]

                CNN_features_select = self.CNN_features[:,indexes,:]
                dist_sq_cnn = torch.cdist(CNN_features_select, clusters_cnn)**2

                B, K, _ = self.clusters.shape
                
                soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * self.sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * self.sigma_array_cnn**2)) , dim = 2)                # shape = [B, N, K]  

                # The distances are weighted by the soft memberships
                dist_sq_weighted = soft_memberships * dist_sq_cat                                           # shape = [B, N, K] 

                distortion_loss = torch.mean(dist_sq_weighted)                                          # shape = [1]

                ###################################### CONFLICT LOSS ###################################################
                # print("labels", labels.shape)                                                         # shape = [B, 1, H, W]
                
                labels_reshape = self.labels.permute(0,2,3,1).float()                                   # shape = [B, H, W, 1]   

                # Find the indexes of the class labels larger than 0 (0 is means unknown class)
                label_locations = torch.gt(labels_reshape, 0).float()                                   # shape = [B, H, W, 1]
                label_locations_flat = torch.flatten(label_locations, start_dim=1, end_dim=2)           # shape = [B, N, 1]  

                XY_features_label = (self.XY_features * label_locations_flat)[0]                        # shape = [N, 2]
                non_zero_indexes = torch.abs(XY_features_label).sum(dim=1) > 0                          # shape = [N] 
                XY_features_label_filtered = XY_features_label[non_zero_indexes].unsqueeze(0)           # shape = [1, N_labelled, 2]
                dist_sq_xy = torch.cdist(XY_features_label_filtered, clusters_xy)**2                    # shape = [1, N_labelled, K]

                CNN_features_label = (self.CNN_features * label_locations_flat)[0]                      # shape = [N, 15]
                CNN_features_label_filtered = CNN_features_label[non_zero_indexes].unsqueeze(0)         # shape = [1, N_labelled, 15]
                dist_sq_cnn = torch.cdist(CNN_features_label_filtered, clusters_cnn)**2                 # shape = [1, N_labelled, K]

                soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * self.sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * self.sigma_array_cnn**2)) , dim = 2)          # shape = [B, N_labelled, K]  
                soft_memberships_T = torch.transpose(soft_memberships, 1, 2)                            # shape = [1, K, N_labelled]

                labels_flatten = torch.flatten(labels_reshape, start_dim=1, end_dim=2)[0]               # shape = [N, 1]
                labels_filtered = labels_flatten[non_zero_indexes].unsqueeze(0)                         # shape = [1, N_labelled, 1] 

                # Use batched matrix multiplication to find the inner product between all of the pixels 
                innerproducts = torch.bmm(soft_memberships, soft_memberships_T)                         # shape = [1, N_labelled, N_labelled]

                # Create an array of 0's and 1's based on whether the class of both the pixels are equal or not
                # If they are the the same class, then we want a 0 because we don't want to add to the loss
                # If the two pixels are not the same class, then we want a 1 because we want to penalise this
                check_conflicts_binary = (~torch.eq(labels_filtered, torch.transpose(labels_filtered, 1, 2))).float()      # shape = [1, N_labelled, N_labelled]

                # Multiply these ones and zeros with the innerproduct array
                # Only innerproducts for pixels with conflicting labels will remain
                conflicting_innerproducts = torch.mul(innerproducts, check_conflicts_binary)           # shape = [1, N_labelled, N_labelled]

                # Find average of the remaining values for the innerproducts 
                # If we are using batches, then we add this value to our previous stored value for the points loss
                conflict_loss = torch.mean(conflicting_innerproducts)                                # shape = [1]

                return distortion_loss + self.alpha*conflict_loss, distortion_loss, self.alpha*conflict_loss
        
        def optimize_spix(criterion, optimizer, scheduler, norm_val_x, norm_val_y, num_iterations=1000):
            
            best_clusters = criterion.clusters
            prev_loss = float("inf")

            for i in range(1,num_iterations):
                loss, distortion_loss, conflict_loss = criterion()

                # Every ten steps we clamp the X and Y locations of the superpixel centres to within the bounds of the image
                if i % 10 == 0:
                    with torch.no_grad():
                        clusters_x_temp = torch.unsqueeze(torch.clamp(criterion.clusters[0,:,0], 0, ((image_width-1)*norm_val_x)), dim=1)
                        clusters_y_temp = torch.unsqueeze(torch.clamp(criterion.clusters[0,:,1], 0, ((image_height-1)*norm_val_y)), dim=1)
                        clusters_temp = torch.unsqueeze(torch.cat((clusters_x_temp, clusters_y_temp, criterion.clusters[0,:,2:]), dim=1), dim=0)
                    criterion.clusters.data.fill_(0)
                    criterion.clusters.data += clusters_temp 

                if loss < prev_loss:
                    best_clusters = criterion.clusters
                    prev_loss = loss.item()

                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step(loss)

                for param_group in optimizer.param_groups:
                    curr_lr = param_group['lr']

                if curr_lr < 0.001:
                    break

            return best_clusters
        
        def prop_to_unlabelled_spix_feat(sparse_labels, connected, features_cnn, H, W):
            # Detach and prepare CNN features
            features_cnn = features_cnn.detach().cpu().numpy()[0]  # shape = [N, C]
            features_cnn_reshape = np.reshape(features_cnn, (H, W, features_cnn.shape[1]))  # shape = [H, W, C]

            # Calculate unique superpixels and initialize feature array
            unique_spix = np.unique(connected)
            spix_features = np.zeros((len(unique_spix), features_cnn.shape[1] + 1))

            # Calculate average features for each superpixel
            for i, spix in enumerate(unique_spix):
                r, c = np.where(connected == spix)
                features_curr_spix = features_cnn_reshape[r, c]
                spix_features[i, 0] = spix  # store spix index
                spix_features[i, 1:] = np.mean(features_curr_spix, axis=0)  # store average feature vector

            # Label array for all labeled pixels
            mask_np = np.array(sparse_labels).squeeze()
            labelled_indices = np.argwhere(mask_np > 0)
            labels = [
                [mask_np[y, x] - 1, connected[y, x], y, x] 
                for y, x in labelled_indices
            ]
            labels_array = np.array(labels)  # shape = [num_points, 4]

            # Calculate labels for each superpixel with points in it
            spix_labels = []
            for spix in unique_spix:
                if spix in labels_array[:, 1]:
                    label_indices = np.where(labels_array[:, 1] == spix)[0]
                    labels = labels_array[label_indices, 0]
                    most_common = np.bincount(labels).argmax()
                    spix_features_row = spix_features[unique_spix == spix, 1:].flatten()
                    spix_labels.append([spix, most_common] + list(spix_features_row))
            
            spix_labels = np.array(spix_labels)

            # Prepare empty mask and propagate labels
            prop_mask = np.full((H, W), np.nan)

            for i, spix in enumerate(unique_spix):
                r, c = np.where(connected == spix)
                
                # If already labeled, use label from spix_labels
                if spix in spix_labels[:, 0]:
                    label = spix_labels[spix_labels[:, 0] == spix, 1][0]
                    prop_mask[r, c] = label
                else:
                    # Find the nearest labeled superpixel by features
                    labeled_spix_features = spix_labels[:, 2:]
                    one_spix_features = spix_features[i, 1:]
                    distances = np.linalg.norm(labeled_spix_features - one_spix_features, axis=1)
                    nearest_spix_idx = np.argmin(distances)
                    nearest_label = spix_labels[nearest_spix_idx, 1]
                    prop_mask[r, c] = nearest_label

            return prop_mask

        def generate_segmented_image(read_im, read_gt, image_name, num_labels, image_height, image_width, num_classes, unlabeled, sparse_gt=None, ensemble=False, points=False):
            # Load necessary modules and functions
            from spixel_utils import xylab, find_mean_std, img2lab, ToTensor, compute_init_spixel_feat, get_spixel_init
            from ssn import CNN
            from torch.optim import Adam, lr_scheduler

            # Initialize variables
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            k = 100
            norm_val_x = 10 / image_width
            norm_val_y = 10 / image_height
            learning_rate = 0.1
            num_iterations = 50
            num_pixels_used = 3000

            # Load sparse ground truth if provided
            if sparse_gt:
                input_df = pd.read_csv(sparse_gt)
                sparse_coords = np.zeros((image_height, image_width), dtype=int)
                first_image_points = input_df[input_df['Name'] == input_df['Name'].iloc[0]]
                for _, row in first_image_points.iterrows():
                    sparse_coords[row['Row'], row['Column']] = 1

            # Load image and ground truth
            pil_img = Image.open(os.path.join(read_im, image_name))
            GT_pil_img = Image.open(os.path.join(read_gt, image_name))
            image = np.array(pil_img)
            GT_mask_np = np.array(GT_pil_img)
            GT_mask = torch.from_numpy(GT_mask_np)
            GT_mask_torch = np.expand_dims(GT_mask, axis=2)
            transform = transforms.Compose([ToTensor()])
            GT_mask_torch = transform(GT_mask_torch)

            # Prepare sparse labels
            if points == False:
                sparse_mask = np.zeros(image_height * image_width, dtype=int)
                if sparse_gt:
                    sparse_mask = sparse_coords
                else:
                    sparse_mask[:num_labels] = 1
                    np.random.shuffle(sparse_mask)
                sparse_mask = np.reshape(sparse_mask, (image_height, image_width))
                sparse_mask = np.expand_dims(sparse_mask, axis=0)
                sparse_labels = torch.add(GT_mask_torch, 1) * sparse_mask
                sparse_labels = torch.unsqueeze(sparse_labels, 0).to(device)
            else:
                sparse_labels = torch.unsqueeze(GT_mask_torch, 0).to(device)

            # Standardize image
            means, stds = find_mean_std(image)
            image = (image - means) / stds
            transform = transforms.Compose([img2lab(), ToTensor()])
            img_lab = transform(image)
            img_lab = torch.unsqueeze(img_lab, 0)

            # Obtain features
            xylab_function = xylab(1.0, norm_val_x, norm_val_y)
            CNN_function = CNN(5, 64, 100)
            model_dict = CNN_function.state_dict()
            ckp_path = "standardization_C=100_step70000.pth"
            obj = torch.load(ckp_path)
            pretrained_dict = obj['net']
            pretrained_dict = {key[4:]: val for key, val in pretrained_dict.items() if key[4:] in model_dict}
            model_dict.update(pretrained_dict)
            CNN_function.load_state_dict(pretrained_dict)
            CNN_function.to(device)
            CNN_function.eval()

            spixel_centres = get_spixel_init(k, image_width, image_height)
            XYLab, X, Y, Lab = xylab_function(img_lab)
            XYLab = XYLab.to(device)
            X = X.to(device)
            Y = Y.to(device)

            with torch.no_grad():
                features = CNN_function(XYLab)
            
            # change dtype of features to float32
            features = features.float()
            features_magnitude_mean = torch.mean(torch.norm(features, p=2, dim=1))
            features_rescaled = (features / features_magnitude_mean)
            features_cat = torch.cat((X, Y, features_rescaled), dim=1)
            XY_cat = torch.cat((X, Y), dim=1)
            mean_init = compute_init_spixel_feat(features_cat, torch.from_numpy(spixel_centres[0].flatten()).long().to(device), k)
            CNN_features = torch.flatten(features_rescaled, start_dim=2, end_dim=3)
            CNN_features = torch.transpose(CNN_features, 2, 1)
            XY_features = torch.flatten(XY_cat, start_dim=2, end_dim=3)
            XY_features = torch.transpose(XY_features, 2, 1)
            features_cat = torch.flatten(features_cat, start_dim=2, end_dim=3)
            features_cat = torch.transpose(features_cat, 2, 1)

            torch.backends.cudnn.benchmark = True

            if ensemble:
                sigma_xy_1, sigma_cnn_1, alpha_1 = 0.5597, 0.5539, 1500
                sigma_xy_2, sigma_cnn_2, alpha_2 = 0.5309, 0.846, 1590
                sigma_xy_3, sigma_cnn_3, alpha_3 = 0.631, 0.5534, 1140

                criterion_1 = CustomLoss(mean_init, image_height * image_width, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_1, sigma_val_cnn=sigma_cnn_1, alpha=alpha_1, num_pixels_used=num_pixels_used).to(device)
                optimizer_1 = Adam(criterion_1.parameters(), lr=learning_rate)
                scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer_1, factor=0.1, patience=1, min_lr=0.0001)

                criterion_2 = CustomLoss(mean_init, image_height * image_width, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_2, sigma_val_cnn=sigma_cnn_2, alpha=alpha_2, num_pixels_used=num_pixels_used).to(device)
                optimizer_2 = Adam(criterion_2.parameters(), lr=learning_rate)
                scheduler_2 = lr_scheduler.ReduceLROnPlateau(optimizer_2, factor=0.1, patience=1, min_lr=0.0001)

                criterion_3 = CustomLoss(mean_init, image_height * image_width, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_3, sigma_val_cnn=sigma_cnn_3, alpha=alpha_3, num_pixels_used=num_pixels_used).to(device)
                optimizer_3 = Adam(criterion_3.parameters(), lr=learning_rate)
                scheduler_3 = lr_scheduler.ReduceLROnPlateau(optimizer_3, factor=0.1, patience=1, min_lr=0.0001)

                best_clusters_1 = optimize_spix(criterion_1, optimizer_1, scheduler_1, norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
                best_members_1 = members_from_clusters(sigma_xy_1, sigma_cnn_1, XY_features, CNN_features, best_clusters_1)

                best_clusters_2 = optimize_spix(criterion_2, optimizer_2, scheduler_2, norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
                best_members_2 = members_from_clusters(sigma_xy_2, sigma_cnn_2, XY_features, CNN_features, best_clusters_2)

                best_clusters_3 = optimize_spix(criterion_3, optimizer_3, scheduler_3, norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
                best_members_3 = members_from_clusters(sigma_xy_3, sigma_cnn_3, XY_features, CNN_features, best_clusters_3)

                best_members_1_max = torch.squeeze(torch.argmax(best_members_1, 2))
                best_members_2_max = torch.squeeze(torch.argmax(best_members_2, 2))
                best_members_3_max = torch.squeeze(torch.argmax(best_members_3, 2))

                connected_1 = enforce_connectivity(best_members_1_max, image_height, image_width, k, connectivity=True)
                connected_2 = enforce_connectivity(best_members_2_max, image_height, image_width, k, connectivity=True)
                connected_3 = enforce_connectivity(best_members_3_max, image_height, image_width, k, connectivity=True)

                prop_1 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_1, CNN_features, image_height, image_width)
                prop_2 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_2, CNN_features, image_height, image_width)
                prop_3 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_3, CNN_features, image_height, image_width)

                prop_1_onehot = np.eye(num_classes, dtype=np.int32)[prop_1.astype(np.int32)]
                prop_2_onehot = np.eye(num_classes, dtype=np.int32)[prop_2.astype(np.int32)]
                prop_3_onehot = np.eye(num_classes, dtype=np.int32)[prop_3.astype(np.int32)]

                prop_count = prop_1_onehot + prop_2_onehot + prop_3_onehot

                if unlabeled == 0:
                    propagated_full = np.argmax(prop_count[:, :, 1:], axis=-1) + 1
                    propagated_full[prop_count[:, :, 0] == 3] = 0
                else:
                    propagated_full = np.argmax(prop_count[:, :, :-1], axis=-1)
                    propagated_full[prop_count[:, :, unlabeled] == 3] = unlabeled

            else:
                criterion = CustomLoss(mean_init, image_height * image_width, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy, sigma_val_cnn=sigma_cnn, alpha=alpha, num_pixels_used=num_pixels_used).to(device)
                optimizer = Adam(criterion.parameters(), lr=learning_rate)
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, min_lr=0.0001)
                best_clusters = optimize_spix(criterion, optimizer, scheduler, norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
                best_members = members_from_clusters(sigma_xy, sigma_cnn, XY_features, CNN_features, best_clusters)
                connected = enforce_connectivity(torch.squeeze(torch.argmax(best_members, 2)), image_height, image_width, k, connectivity=True)
                propagated_full = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected, CNN_features, image_height, image_width)

            return propagated_full

        read_im = image_dir
        read_gt = gt_images_dir
        image_name = filename
        num_labels = len(points)
        image_width = image.shape[1]
        image_height = image.shape[0]
        num_classes = NUM_CLASSES

        # Iterate through the keys of color_dict to find the index of background_class
        for idx, key in enumerate(color_dict.keys()):
            if key == background_class:
                unlabeled = idx
                break
        # get index of unlabeled class in the color_dict


        sparse_gt = args.ground_truth

        expanded_image = generate_segmented_image(read_im, read_gt, image_name, num_labels, image_height, image_width, num_classes, unlabeled, sparse_gt=sparse_gt, ensemble=ensemble)

        # Convert the image to dataframe
        expanded_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label"])
        for l in unique_labels:
            expanded_points = np.argwhere(expanded_image == l)
            data = []
            l_str = None
            for idx, (key, value) in enumerate(color_dict.items()):
                if idx == l:
                    l_str = key
                    break

            for point in expanded_points:
                data.append({
                    "Name": image_name,
                    "Row": point[0],
                    "Column": point[1],
                    "Label": l
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
parser.add_argument("--generate_eval_images", help="Generate evaluation images for the expansion (sparse and expanded images for all the classes)", required=False, action='store_true')
parser.add_argument("--color_dict", help="CSV file containing the color dictionary", required=False)
parser.add_argument("--generate_csv", help="Generate a sparse csv file", required=False, action='store_true')
parser.add_argument("--frame", help="Frame size to crop the images", required=False, type=int, default=0)
parser.add_argument("--gt_images", help="Directory containing the ground truth images.", required=False)
parser.add_argument("--gt_images_colored", help="Directory containing the ground truth images. Just for visual comparison", required=False)

parser.add_argument("-b", "--background_class", help="background class value (for grayscale, provide an integer; for color, provide a tuple)", required=True, default=0)
parser.add_argument("-n", "--num_classes", help="Number of classes in the dataset", required=True, type=int, default=35)
parser.add_argument('--ensemble', action='store_true', dest='ensemble', help='use this flag when you would like to use an ensemble of 3 classifiers, otherwise the default is to use a single classifier')
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

ensemble = args.ensemble

image_path = args.input_dir
print("Image directory:", image_path)
if not os.path.exists(image_path):
    parser.error(f"The directory {image_path} does not exist")

if args.frame:
    BORDER_SIZE = args.frame

if args.gt_images_colored:
    gt_images_colored_dir = args.gt_images_colored
    if not os.path.exists(gt_images_colored_dir):
        parser.error(f"The directory {gt_images_colored_dir} does not exist")

if args.gt_images:
    gt_images_dir = args.gt_images
    if not os.path.exists(gt_images_dir):
        parser.error(f"The directory {gt_images_dir} does not exist")

NUM_CLASSES = args.num_classes

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
else:
    if generate_eval_images:
        # Ensure args.color_dict is None
        assert args.color_dict is None, "Expected args.color_dict to be None when generating evaluation images without a provided color dictionary."
    else:
        labels = input_df['Label'].unique().tolist()

if args.model == "sam" or args.model == "mixed":
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())


    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2_model)
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

        if args.gt_images_colored:
            # Get the base name of the image without extension
            base_image_name = os.path.splitext(image_name)[0]
            
            # List all files in the gt_images_path directory
            gt_files = os.listdir(gt_images_colored_dir)
            
            # Find the file that matches the base_image_name
            gt_image_file = next((f for f in gt_files if os.path.splitext(f)[0] == base_image_name), None)
            
            if gt_image_file:
                gt_image_path = os.path.join(gt_images_colored_dir, gt_image_file)
                gt_image = cv2.imread(gt_image_path)
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            else:
                print(f"ERROR: Ground truth image for {image_name} not found in {gt_images_colored_dir}")

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

            if args.gt_images_colored:
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
                axs[1].set_title("SAM2.1")
                axs[1].axis('off')
                axs[2].imshow(color_mask_spx)
                axs[2].set_title("Point Label Aware Superpixels")
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
                axs[1].set_title("SAM2.1")
                axs[1].axis('off')
                axs[2].imshow(color_mask_spx)
                axs[2].set_title("Point Label Aware Superpixels")
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
        
        # Update progress bar
        pbar.update(1)

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
