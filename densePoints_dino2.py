
import argparse
import itertools
import math
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from abc import ABC, abstractmethod
import time
import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from scipy.ndimage import label
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from skimage.measure import perimeter
from skimage.segmentation import find_boundaries
from sklearn.metrics.pairwise import cosine_similarity

BORDER_SIZE = 0

def blend_translucent_color(img, row, column, color, alpha):
    # Get the existing color
    existing_color = img[row, column]
    
    # Blend the new color with the existing color
    blended_color = existing_color * (1 - alpha) + color * alpha

    # Assign the blended color to the pixel
    img[row, column] = blended_color

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
            segment_gt_points, _ = gather_gt_points_from_segment_area(
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

def show_points(coords, ax, marker_size=30, marker_color='blue', edge_color='white'):
    ax.scatter(coords[:, 0], coords[:, 1], color=marker_color, marker='*', s=marker_size, edgecolor=edge_color, linewidth=1.25)

def generate_image(image_df, image, points, color_dict, image_name, output_dir, label_str=None):

    image_df_str = image_df.copy()
    image_df_str['Label'] = image_df_str['Label'].apply(lambda x: list(color_dict.keys())[x])

    # If label_str is None, default to 'ALL'
    if label_str is None:
        label_str = 'ALL'

    # Create a black image (RGBA format)
    black = image.copy()
    black = black.astype(float) / 255
    black = np.dstack((black, np.ones((black.shape[0], black.shape[1]))))  # Add alpha channel

    # Image dimensions (1024x768)
    height, width, _ = black.shape

    dpi = 100  # Dots per inch (or use 200 or higher for finer quality)

    # Set the figure size to match the image size (1024x768) in pixels
    figsize = (width / dpi, height / dpi)  # Use the exact image dimensions in inches

    # Prepare output directory
    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate marker size proportional to image size
    marker_size = (height * width) / 5000  # Adjust the divisor to control the marker size

    # Save sparse points image
    plt.figure(figsize=figsize, dpi=dpi)  # Use the exact figure size
    plt.imshow(black)
    show_points(points, plt.gca(), marker_size, marker_color='black', edge_color='yellow')
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure the aspect ratio is 1:1
    plt.xlim(0, width)  # Set x-axis limit to image width
    plt.ylim(height, 0)  # Set y-axis limit to image height (reverse y-axis)
    plt.axis('off')  # Hide the axes
    plt.tight_layout(pad=0)  # Adjust layout to fit the image within bounds
    plt.savefig(output_dir + image_name + '_' + label_str + '_sparse.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Color the points in the image (using image_df)
    for _, row in image_df_str.iterrows():
        if row['Column'] < black.shape[1] and row['Row'] < black.shape[0]:
            point = (row['Row'] + BORDER_SIZE, row['Column'] + BORDER_SIZE)
            color = np.array([int(color_dict[str(row['Label'])][0]) / 255.0, 
                              int(color_dict[str(row['Label'])][1]) / 255.0, 
                              int(color_dict[str(row['Label'])][2]) / 255.0, 1])
            blend_translucent_color(black, point[0], point[1], color, color[3])

    # Save expanded points image
    plt.figure(figsize=figsize, dpi=dpi)  # Use the exact figure size
    plt.imshow(black)
    show_points(points, plt.gca(), marker_size, marker_color='black', edge_color='yellow')
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure the aspect ratio is 1:1
    plt.xlim(0, width)  # Set x-axis limit to image width
    plt.ylim(height, 0)  # Set y-axis limit to image height (reverse y-axis)
    plt.axis('off')  # Hide the axes
    plt.tight_layout(pad=0)  # Adjust layout to fit the image within bounds
    plt.savefig(output_dir + image_name + '_' + label_str + '_expanded.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

class LabelExpander(ABC):
    def __init__(self, color_dict, labels, output_df): 
        self.color_dict = color_dict
        self.unique_labels_str = labels
        self.output_df = output_df
        self.remove_far_points = remove_far_points
        self.eval_images = eval_images
        self.generate_csv = generate_csv

    def generate_csv(self):
        # TODO: 
        pass

    def expand_image(self, points, unique_labels_i, unique_labels_str_i, image, background_class, eval_images_dir_i):      
        expanded_df = self.expand_labels(points, labels, unique_labels_i, unique_labels_str_i, image, image_name, background_class, eval_images_dir_i)
        
        if self.eval_images and isinstance(self, SAMLabelExpander):
            generate_image(expanded_df, image, np.flip(points, axis=1), self.color_dict, image_name, eval_images_dir_i)

        if self.generate_csv:
            pass

        return expanded_df
        
    @abstractmethod
    def expand_labels(self, points, labels, unique_labels_i, unique_labels_str_i, image, image_name, background_class=None, eval_image_dir=None):
        pass

class SAMLabelExpander(LabelExpander):
    def __init__(self, color_dict, labels, output_df, mask_generator, out_features_path=None, eval_images=False):
        super().__init__(color_dict, labels, output_df)
        self.mask_generator = mask_generator
        self.eval_images = eval_images
        self.out_features_path = out_features_path

    def __compute_mask_metrics(self, mask, score):
        """
        Compute and normalize mask metrics: compactness, size penalty, and score.

        Args:
            mask (np.array): Binary mask for the segment.
            score (float): Score assigned by SAM for the mask.
            image_shape (tuple): Shape of the image as (height, width).

        Returns:
            tuple: Normalized compactness, size penalty, and score.
        """

        # Extract image dimensions

        # Mask metrics
        mask_area = mask.sum()  # Total pixels in the mask
        mask_perimeter = perimeter(mask)  # Perimeter of the mask

        # Compactness: Avoid divide-by-zero errors
        if mask_area > 0:
            # Ideal perimeter for a circle with the same area
            ideal_perimeter = 2 * np.sqrt(np.pi * mask_area)
            
            # Compactness: The ratio of the perimeter to the ideal perimeter (closer to 1 is more compact)
            if mask_perimeter > 0:
                raw_compactness = ideal_perimeter / mask_perimeter  # Inverse, so lower perimeter = higher compactness
            else:
                raw_compactness = 0  # Handle the case when mask_perimeter is 0
        else:
            raw_compactness = 0

        # Normalize compactness (keeping compactness between 0 and 1)
        # Higher compactness for well-defined, continuous masks, lower for scattered/irregular ones
        compactness = min(raw_compactness, 1)  # Ensure compactness doesn't exceed 1

        # Normalized size penalty
        total_pixels = (HEIGHT - BORDER_SIZE*2) * (WIDTH - BORDER_SIZE*2)

        normalized_area = mask_area / total_pixels  # Fraction of the image covered by the mask

        # Gentle penalty for very small masks (e.g., < 1% of image)
        if normalized_area < 0.001:  # Only apply penalty for masks smaller than 1% of the image
            small_mask_penalty = (normalized_area) ** 4  # Soft quadratic penalty
        else:
            small_mask_penalty = 0  # No penalty for larger masks

        # Large mask penalty (unchanged)
        large_mask_penalty = (normalized_area - 0.4) ** 4 if normalized_area > 0.5 else 0

        # Combine penalties gently
        size_penalty = normalized_area + small_mask_penalty + large_mask_penalty

        # Return normalized metrics
        return compactness, size_penalty, score

    
    def __weighted_mask_selection(self, masks, scores, weights=(1.0, 0.8, 1.4)):
        best_score = -np.inf
        best_index = -1  # Initialize with an invalid index
        
        w_s, w_c, w_a = weights  # Weights for SAM Score, Compactness, and Size
        
        for i, mask in enumerate(masks):
            # Compute metrics
            compactness, size_penalty, sam_score = self.__compute_mask_metrics(mask, scores[i])

            
            # Weighted score (nonlinear terms)
            weighted_score = (
                w_s * sam_score +               # Higher SAM score is better
                w_c * np.log(1 + compactness) - # Higher compactness is better (log smoothing)
                w_a * size_penalty              # Lower size penalty is better
            )

            # print(f"Mask {i+1} - SAM Score: {sam_score:.4f} - Compactness: {compactness:.4f} - Size Penalty: {size_penalty:.4f} \nWeighted Score: {weighted_score:.4f}")

            
            # Select best mask
            if weighted_score > best_score:
                best_score = weighted_score
                best_index = i  # Store the index of the best mask
        
        # print("\n")
                
        return best_index

    def expand_labels(self, points, labels, unique_labels, unique_labels_str, image, image_name, background_class=None, eval_image_dir=None):
        expanded_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label", "Segment"])
        
        time_start = time.time()

        # Initialize the segment counter
        segment_counter = 0

        gt_points = np.array([])
        gt_labels = np.array([])

        # Crop the image if BORDER_SIZE > 0, otherwise use the full image
        if BORDER_SIZE > 0:
            cropped_image = image[BORDER_SIZE:image.shape[0]-BORDER_SIZE, BORDER_SIZE:image.shape[1]-BORDER_SIZE]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        else:
            cropped_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Store points and labels within borders if BORDER_SIZE > 0
        filtered_points_all = []
        filtered_labels_all = []

        for idx, key in enumerate(self.color_dict.keys()):
            if key == background_class:
                unlabeled = idx
                break

        # Filter out background labels from unique_labels and unique_labels_str
        unique_labels = [label for i, label in enumerate(unique_labels) if unique_labels_str[i] != str(unlabeled)]
        unique_labels_str = [label_str for label_str in unique_labels_str if label_str != str(unlabeled)]

        for i in range(len(unique_labels)):
            
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

        # Get masks and embeddings from automatic mask generator
        masks, _ = self.mask_generator.generate(cropped_image)

        np.random.seed(3)

        def show_anns(anns, borders=True):
            if len(anns) == 0:
                return
            sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
            ax = plt.gca()
            ax.set_autoscale_on(False)

            img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
            img[:, :, 3] = 0
            for ann in sorted_anns:
                m = ann['segmentation']
                color_mask = np.concatenate([np.random.random(3), [0.5]])
                img[m] = color_mask 
                if borders:
                    import cv2
                    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                    # Try to smooth contours
                    contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                    cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

            ax.imshow(img)

        # plt.figure(figsize=(20, 20))
        # plt.imshow(cropped_image)
        # show_anns(masks)
        # plt.axis('off')
        # plt.show()

        class CenterPadding(torch.nn.Module):
            def __init__(self, multiple = 14):
                super().__init__()
                self.multiple = multiple

            def _get_pad(self, size):
                new_size = math.ceil(size / self.multiple) * self.multiple
                pad_size = new_size - size
                pad_size_left = pad_size // 2
                pad_size_right = pad_size - pad_size_left
                return pad_size_left, pad_size_right

            @torch.inference_mode()
            def forward(self, x):
                pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
                output = F.pad(x, pads)
                return output


        dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dinov2_vitg14.to(device)

        transform = T.Compose([
            T.ToTensor(),
            lambda x: x.unsqueeze(0),
            CenterPadding(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        
        cropped_image_pil = Image.fromarray(cropped_image)
        
        with torch.inference_mode():
            layers = eval("[23]")
            # intermediate layers does not use a norm or go through the very last layer of output
            img = transform(cropped_image_pil).to(device=device,dtype=torch.bfloat16)
            features_out = dinov2_vitg14.get_intermediate_layers(img, n=layers,reshape=True)    
            features = torch.cat(features_out, dim=1) # B, C, H, W

        # Transform embedding into a spatial representation
        embedding = features[0].cpu().detach().unsqueeze(0).numpy()  # Shape: (1, 1000)
        embedding_tensor = torch.tensor(embedding)
        upsampled_embedding = torch.nn.functional.interpolate(embedding_tensor, size=(cropped_image.shape[0], cropped_image.shape[1]), mode='bilinear', align_corners=False)
        upsampled_embedding = upsampled_embedding.squeeze(0).cpu().detach().numpy()

        embedding_info = {}

        num_masks = len(masks)
        num_features, height, width = upsampled_embedding.shape

        # Initialize a binary mask to track assigned pixels
        assigned_mask = np.zeros((height, width), dtype=bool)

        # Sort masks by area in ascending order
        masks = sorted(masks, key=lambda x: np.sum(x['segmentation']))

        for mask_idx, mask_dict in enumerate(masks):
            mask = mask_dict["segmentation"]

            # Only consider unassigned pixels for this mask
            unassigned_mask = mask & ~assigned_mask

            # Skip this mask if all its pixels are already assigned
            if not unassigned_mask.any():
                continue

            # Extract embeddings for unassigned pixels
            mask_flat = unassigned_mask.flatten()
            embedding_flat = upsampled_embedding.reshape(-1, num_features)
            mask_embedding = embedding_flat[mask_flat == 1]

            # Update embedding info
            embedding_info[mask_idx] = {
                'mask': unassigned_mask,  # Store the adjusted mask
                'embedding': mask_embedding  # Store embeddings for unassigned pixels
            }

            # Determine GT labels for unassigned pixels
            mask_coords = np.argwhere(unassigned_mask)
            mask_coords_set = set(map(tuple, mask_coords))

            gt_points_in_mask = []
            gt_labels_in_mask = []

            for points, labels in zip(filtered_points_all, filtered_labels_all):
                for point, label in zip(points, labels):
                    point_tuple = tuple(point)
                    if point_tuple in mask_coords_set:
                        gt_points_in_mask.append(point)
                        gt_labels_in_mask.append(label)

            # Assign label based on GT points
            if len(gt_labels_in_mask) == 0:
                mask_label = -1  # No label for this mask
            else:
                gt_labels_in_mask_flat = [item for sublist in gt_labels_in_mask for item in sublist]
                u_labels, counts = np.unique(gt_labels_in_mask_flat, return_counts=True)
                if len(u_labels) == 1:
                    mask_label = u_labels[0]
                else:
                    mask_label = u_labels[np.argmax(counts)]  # Majority label

            embedding_info[mask_idx]['label'] = mask_label

            # Mark pixels as assigned
            assigned_mask |= unassigned_mask
        
        print("Number of masks with labels:", len([mask for mask in embedding_info.values() if mask['label'] != -1]))
        print("Number of masks without labels:", len([mask for mask in embedding_info.values() if mask['label'] == -1]))

        def calculate_similarity(embedding_info):
            """
            Calculate the similarity between labeled and unlabeled masks using cosine similarity.

            Args:
                embedding_info (dict): Dictionary containing mask embeddings and labels.

            Returns:
                dict: Updated embedding_info with assigned labels for unlabeled masks.
            """

            # Calculate mean features for each mask
            for mask_idx, mask_info in embedding_info.items():
                mask_embedding = mask_info['embedding']
                mean_features = np.mean(mask_embedding, axis=0)
                embedding_info[mask_idx]['mean_features'] = mean_features

            # Calculate similarities for unlabeled masks
            for mask_idx, mask_info in embedding_info.items():
                if mask_info['label'] == -1:
                    mean_features = mask_info['mean_features']

                    # Calculate similarity with each labeled mask
                    similarities = []
                    for labeled_mask_idx, labeled_mask_info in embedding_info.items():
                        if labeled_mask_info['label'] != -1:
                            labeled_mean_features = labeled_mask_info['mean_features']
                            similarity = cosine_similarity([mean_features], [labeled_mean_features])[0][0]
                            similarities.append((similarity, labeled_mask_info['label']))

                    # Find the most similar labeled mask
                    if similarities:
                        if max(similarities, key=lambda x: x[0])[0] > 0.5:
                            most_similar_label = max(similarities, key=lambda x: x[0])[1]
                            mask_info['label'] = most_similar_label

            # Plot similarities after all labels have been recalculated
            for reference_mask_idx, reference_mask_info in embedding_info.items():
                reference_mean_features = reference_mask_info['mean_features']

                similarities = []
                for other_mask_idx, other_mask_info in embedding_info.items():
                    other_mean_features = other_mask_info['mean_features']
                    similarity = cosine_similarity([reference_mean_features], [other_mean_features])[0][0]
                    similarities.append((other_mask_idx, similarity))

                # Normalize similarities to [0, 1] for colormap
                similarities = np.array(similarities)
                min_similarity = np.min(similarities[:, 1])
                max_similarity = np.max(similarities[:, 1])
                normalized_similarities = (similarities[:, 1] - min_similarity) / (max_similarity - min_similarity)

                # # Create a colormap
                # colormap = plt.cm.get_cmap('viridis')
                # # Plot masks with colors based on similarity
                # similarity_image = np.zeros_like(image, dtype=np.float32)
                # for (other_mask_idx, similarity) in similarities:
                #     mask = embedding_info[other_mask_idx]['mask']
                #     color = colormap(normalized_similarities[int(other_mask_idx)])
                #     similarity_image[mask == 1] = color[:3]  # Use RGB channels

                # # Display the similarity image with colorbar
                # plt.figure(figsize=(10, 10))
                # plt.imshow(similarity_image)
                # plt.title(f'Similarity of all masks with respect to mask {reference_mask_idx}')
                # plt.axis('off')
                
                # # Add colorbar
                # cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=plt.gca(), orientation='vertical')
                # cbar.set_label('Similarity')
                # cbar.set_ticks([0, 0.5, 1])
                # cbar.set_ticklabels(['-1', '0', '1'])
                
                # plt.show()
            
            return embedding_info

        # Calculate similarity and assign labels to unlabeled masks
        embedding_info = calculate_similarity(embedding_info)

        # Create a final mask to ensure each pixel is assigned only one label
        final_mask = np.full((cropped_image.shape[0], cropped_image.shape[1]), -1, dtype=int)

        # Sort masks by area in ascending order
        sorted_masks = sorted(embedding_info.items(), key=lambda x: np.sum(x[1]['mask']))

        for mask_idx, mask_info in sorted_masks:
            mask = mask_info['mask']
            label = mask_info['label']

            if label != background_class:
                # Use bitwise operations to update the final mask
                final_mask = np.where((final_mask == -1) & mask, label, final_mask)

        # Convert final mask to DataFrame
        data = []
        rows, cols = np.where(final_mask != -1)
        for row, col in zip(rows, cols):
            data.append({
                "Row": row,
                "Column": col,
                "Label": final_mask[row, col],
                "Segment": segment_counter
            })

        new_data_df = pd.DataFrame(data)
        
        if self.eval_images:
            for i in range(len(unique_labels)):
                _points_l = filtered_points_all[i]
            generate_image(new_data_df, image, _points_l, color_dict, image_name, eval_image_dir, unique_labels_str[i])
        
        expanded_df = pd.concat([expanded_df, new_data_df], ignore_index=True)
        

        return expanded_df

class SuperpixelLabelExpander(LabelExpander):
    def __init__(self, color_dict, labels, output_df):
        super().__init__(color_dict, labels, output_df)
        self.color_dict = color_dict

    def expand_labels(self, points, labels, unique_labels, unique_labels_str, image, image_name, background_class=None, eval_image_dir=None):

        sigma_xy = 0.631
        sigma_cnn = 0.5534
        alpha = 1140

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

        def generate_segmented_image(read_im, image_name, num_labels, image_height, image_width, num_classes, unlabeled, sparse_gt=None, ensemble=False, points=False):
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

            # Load sparse ground truth from `input_df`
            if sparse_gt:
                input_df = pd.read_csv(sparse_gt)
                sparse_coords = np.zeros((image_height, image_width), dtype=int)
                image_points = input_df[input_df['Name'] == image_name]
                
                # Populate sparse_coords with labels at the specified points
                for _, row in image_points.iterrows():
                    label_str = str(row['Label'])
                    label_int = list(color_dict.keys()).index(label_str)
                    sparse_coords[row['Row'], row['Column']] = label_int+1

            # Load the image (if necessary, for display or other purposes)
            pil_img = Image.open(os.path.join(read_im, image_name))
            image = np.array(pil_img)

            # Prepare sparse labels
            sparse_mask = np.zeros((image_height, image_width), dtype=int)

            if sparse_gt:
                # Use sparse coordinates from the DataFrame as the sparse mask
                sparse_mask = sparse_coords
            else:
                # Random initialization if no sparse ground truth provided
                sparse_mask[:num_labels] = 1
                np.random.shuffle(sparse_mask)
                sparse_mask = np.reshape(sparse_mask, (image_height, image_width))

            # Create sparse labels tensor from the sparse mask
            sparse_mask = np.expand_dims(sparse_mask, axis=0)
            sparse_labels = torch.from_numpy(sparse_mask).to(device)

            # Expand dimensions to match the expected input format
            sparse_labels = torch.unsqueeze(sparse_labels, 0).to(device)

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

            start_ensemble = time.time()
            if ensemble:
                # print("Ensemble")
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

                def overlay_boundaries_on_image(original_image, assignments, boundary_color=(1, 0, 0)):
                    """
                    Overlay red boundaries of superpixels on the original image.

                    Parameters:
                    - original_image: Original image as a NumPy array (H, W, 3), values in [0, 255].
                    - assignments: Superpixel assignments as a NumPy array (H, W).
                    - boundary_color: Tuple of RGB values for the boundary color, in [0, 1].

                    Returns:
                    - overlaid_image: Image with red superpixel boundaries overlaid.
                    """
                    assignments = assignments.cpu().numpy().reshape(original_image.shape[:2])
                    boundaries = find_boundaries(assignments, mode='outer')

                    # Normalize the original image to [0, 1] for blending
                    original_image = original_image.astype(float) / 2
                    overlaid_image = original_image.copy()

                    # Apply the boundary color
                    for i, color in enumerate(boundary_color):
                        overlaid_image[..., i][boundaries] = color

                    # Convert back to [0, 255] for display
                    return (overlaid_image * 255).astype(np.uint8)

                def visualize_superpixel_overlays(original_image, assignments_list, titles):
                    """
                    Visualize original image with superpixel boundaries from multiple models.

                    Parameters:
                    - original_image: Original image as a NumPy array (H, W, 3).
                    - assignments_list: List of superpixel assignments.
                    - titles: List of titles corresponding to each assignment.
                    """
                    fig, axes = plt.subplots(1, len(assignments_list) + 1, figsize=(18, 6))
                    axes[0].imshow(original_image)
                    axes[0].set_title("Original Image")
                    axes[0].axis("off")

                    for i, (assignments, title) in enumerate(zip(assignments_list, titles)):
                        overlaid_image = overlay_boundaries_on_image(original_image, assignments)
                        num_superpixels = len(np.unique(assignments.cpu().numpy()))
                        print(f"{title}: {num_superpixels} superpixels")  # Print count to terminal
                        axes[i + 1].imshow(overlaid_image)
                        axes[i + 1].set_title(title)
                        axes[i + 1].axis("off")

                    plt.tight_layout()
                    plt.show()

                # Example usage
                # original_image = plt.imread(os.path.join(read_im, image_name))  # Load original image (H, W, 3)
                # visualize_superpixel_overlays(
                #     original_image,
                #     [best_members_1_max, best_members_2_max, best_members_3_max],
                #     ["Superpixels from Model 1", "Superpixels from Model 2", "Superpixels from Model 3"]
                # )

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
                # print("Single")
                criterion = CustomLoss(mean_init, image_height * image_width, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy, sigma_val_cnn=sigma_cnn, alpha=alpha, num_pixels_used=num_pixels_used).to(device)
                optimizer = Adam(criterion.parameters(), lr=learning_rate)
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, min_lr=0.0001)
                best_clusters = optimize_spix(criterion, optimizer, scheduler, norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
                best_members = members_from_clusters(sigma_xy, sigma_cnn, XY_features, CNN_features, best_clusters)
                connected = enforce_connectivity(torch.squeeze(torch.argmax(best_members, 2)), image_height, image_width, k, connectivity=True)
                propagated_full = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected, CNN_features, image_height, image_width)
            
            end_ensemble = time.time()
            # print(f"Time taken by ensemble: {end_ensemble - start_ensemble} seconds")
            return propagated_full

        read_im = image_dir
        num_labels = len(points)
        image_width = image.shape[1]
        image_height = image.shape[0]
        num_classes = NUM_CLASSES

        # Iterate through the keys of color_dict to find the index of background_class
        for idx, key in enumerate(self.color_dict.keys()):
            if key == background_class:
                unlabeled = idx
                break
        # get index of unlabeled class in the color_dict


        sparse_gt = args.ground_truth

        expanded_image = generate_segmented_image(read_im, image_name, num_labels, image_height, image_width, num_classes, unlabeled, sparse_gt=sparse_gt, ensemble=ensemble)

        # Convert the image to dataframe
        expanded_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label"])

        for l in unique_labels:

            expanded_points = np.argwhere(expanded_image == l)

            data = []
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
parser.add_argument("-i", "--input_dir", help="Directory containing images", required=True)
parser.add_argument("-o", "--output_dir", help="Directory to save the output images", required=True)
parser.add_argument("-gt", "--ground_truth", help="CSV file containing the points and labels", required=True)
parser.add_argument("--eval_images", help="Generate evaluation images for the expansion (sparse and expanded images for all the classes)", required=False, action='store_true')
parser.add_argument("-c", "--color_dict", help="CSV file containing the color dictionary", required=False)
parser.add_argument("--generate_csv", help="Generate the CSV file containing the expanded points and labels", required=False, action='store_true')
parser.add_argument("--out_features", help="Save the embedding of the images", required=False, action='store_true')
parser.add_argument("--frame", help="Frame size to crop the images", required=False, type=int, default=0)
parser.add_argument("--gt_images_colored", help="Directory containing the ground truth images. Just for visual comparison", required=False)
parser.add_argument("-b", "--background_class", help="background class value (for grayscale, provide an integer; for color, provide a tuple)", required=True, default=0)
parser.add_argument("-n", "--num_classes", help="Number of classes in the dataset", required=True, type=int, default=35)
parser.add_argument('--ensemble', action='store_true', dest='ensemble', help='use this flag when you would like to use an ensemble of 3 classifiers, otherwise the default is to use a single classifier', default=False)
args = parser.parse_args()

remove_far_points = False
eval_images = False
generate_csv = False
color_dict = {}
NUM_CLASSES = args.num_classes

# Get input points and labels from csv file
input_df = pd.read_csv(args.ground_truth)
output_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label"])

if args.generate_csv:
    generate_csv = True

ensemble = args.ensemble

image_path = args.input_dir
if not os.path.exists(image_path):
    parser.error(f"The directory {image_path} does not exist")

if args.frame:
    BORDER_SIZE = args.frame

if args.gt_images_colored:
    gt_images_colored_dir = args.gt_images_colored
    if not os.path.exists(gt_images_colored_dir):
        parser.error(f"The directory {gt_images_colored_dir} does not exist")


output_dir = args.output_dir
if output_dir[-1] != '/':
    output_dir += '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

unique_labels = input_df['Label'].unique()

if args.eval_images:
    eval_images = True

if args.color_dict:
    color_df = pd.read_csv(args.color_dict, header=None)
    keys = color_df.iloc[0].tolist()
    values = color_df.iloc[1:].values.tolist()
    
    # Create the dictionary
    color_dict = {str(keys[i]): [row[i] for row in values] for i in range(len(keys))}
    
    # Get the labels that are in self.color_dict.keys() but not in labels
    extra_labels = set(color_dict.keys()) - set(map(str, unique_labels))
else:
    if eval_images:
        # Ensure args.color_dict is None
        assert args.color_dict is None, "Expected args.color_dict to be None when generating evaluation images without a provided color dictionary."
    else:
        labels = input_df['Label'].unique().tolist()

# Convert labels to integers using the color_dict
input_df['Label'] = input_df['Label'].apply(lambda x: list(color_dict.keys()).index(str(x)))

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
mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model,
                                               points_per_side=64,
                                               points_per_patch=128,
                                               pred_iou_threshold=0.7,
                                               stability_score_thresh=0.92,
                                               stability_score_offset=0.7,
                                               crop_n_layers=1,
                                               box_nms_thresh=0.7,
                                               )

if args.out_features is not None:
    out_features_path = os.path.join(output_dir, 'features')
    if not os.path.exists(out_features_path):
        os.makedirs(out_features_path)
    LabelExpander_sam = SAMLabelExpander(color_dict, unique_labels, output_df, mask_generator, out_features_path, eval_images)
else:
    LabelExpander_sam = SAMLabelExpander(color_dict, unique_labels, output_df, mask_generator, eval_images=eval_images)

LabelExpander_spx = SuperpixelLabelExpander(color_dict, unique_labels, output_df)


mask_dir = output_dir + 'labels/'
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

if eval_images:
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

# Generate label colors
if not isinstance(unique_labels[0], str):
    print("Generating label colors")
    total_colors = len(unique_labels)
    label_colors = {
        label: hsv_to_rgb(*get_color_hsv(i, total_colors)) for i, label in enumerate(sorted(unique_labels))
    }
    label_colors[0] = (0, 0, 0)

# Initialize lists to store execution times
sam_times = []
spx_times = []

# Initialize directories outside the loop if they don't change
mask_color_dir = os.path.join(output_dir, 'labels_mosaic')
os.makedirs(mask_color_dir, exist_ok=True)

with tqdm(total=len(image_names_csv), desc="Processing images") as pbar:
    gt_files = os.listdir(gt_images_colored_dir) if args.gt_images_colored else []
    for image_name in image_names_csv:
        # Avoid frequent description updates if not essential
        if pbar.n % 10 == 0:
            pbar.set_description(f"Processing {image_name}")

        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: Failed to load image at {image_path}")
            pbar.update(1)
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        HEIGHT, WIDTH = image.shape[:2]

        if args.gt_images_colored:
            base_image_name = os.path.splitext(image_name)[0]
            gt_image_file = next((f for f in gt_files if os.path.splitext(f)[0] == base_image_name), None)
            if gt_image_file:
                gt_image_path = os.path.join(gt_images_colored_dir, gt_image_file)
                gt_image = cv2.imread(gt_image_path)
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            else:
                print(f"ERROR: Ground truth image for {image_name} not found in {gt_images_colored_dir}")

        # Extract points and labels in a single lookup
        image_data = input_df[input_df['Name'] == image_name]
        points = image_data.iloc[:, 1:3].to_numpy().astype(int)
        labels = image_data.iloc[:, 3].to_numpy()
        unique_labels_i = np.unique(labels)
        unique_labels_str_i = unique_labels_i.astype(str)

        if eval_images:
            eval_images_dir_i = eval_images_dir + image_name + '/'
        else:
            eval_images_dir_i = None

        background = args.background_class

        # background_gray is the index of the background label in the color_dict
        background_gray = list(color_dict.keys()).index(background)
        background_color = color_dict.get(background, (background, background, background))

        start_expand = time.time()

        start_sam = time.time()
        expanded_sam = LabelExpander_sam.expand_image(points, unique_labels_i, unique_labels_str_i, image, background_class=background, eval_images_dir_i=eval_images_dir_i)
        end_sam = time.time()
        sam_times.append(end_sam - start_sam)
        start_spx = time.time()
        expanded_spx = LabelExpander_spx.expand_image(points, unique_labels_i, unique_labels_str_i, image, background_class=background, eval_images_dir_i=eval_images_dir_i)
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

        for l in unique_labels_i:
            expanded_i_sam = expanded_sam[expanded_sam['Label'] == l].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
            expanded_i_spx = expanded_spx[expanded_spx['Label'] == l].iloc[:, 1:3].to_numpy().astype(int)
            expanded_i_spx = expanded_i_spx[(expanded_i_spx[:, 0] >= BORDER_SIZE) & (expanded_i_spx[:, 0] < image.shape[0] - BORDER_SIZE) & 
                                            (expanded_i_spx[:, 1] >= BORDER_SIZE) & (expanded_i_spx[:, 1] < image.shape[1] - BORDER_SIZE)]

            expanded_i_mix = output_df[output_df['Label'] == l].iloc[:, 1:3].to_numpy().astype(int)
            expanded_i_mix = expanded_i_mix[(expanded_i_mix[:, 0] >= BORDER_SIZE) & (expanded_i_mix[:, 0] < image.shape[0] - BORDER_SIZE) & 
                                            (expanded_i_mix[:, 1] >= BORDER_SIZE) & (expanded_i_mix[:, 1] < image.shape[1] - BORDER_SIZE)]

            if rgb_flag:
                l_str = list(color_dict.keys())[l]
                color = np.array(color_dict[l_str])
            else:
                color = label_colors[l]

            # if l != 25:
            #     continue

            color_mask_sam[expanded_i_sam[:, 0], expanded_i_sam[:, 1]] = color
            color_mask_spx[expanded_i_spx[:, 0], expanded_i_spx[:, 1]] = color
            color_mask_mix[expanded_i_mix[:, 0], expanded_i_mix[:, 1]] = color

        # # Keep only color of class 25 of GT image
        # gt_image_25 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        # color_25 = color_dict.get('25', (0, 0, 0))
        # mask_25_gt = np.all(gt_image == color_25, axis=-1)
        # gt_image_25[mask_25_gt] = color_25
        # gt_image_25[~mask_25_gt] = background_color

        # mask_25_overexpanded = np.all(color_mask_spx == color_25, axis=-1)
        # overexpanded = color_mask_spx

        # gt_background_mask = np.all(gt_image == background_color, axis=-1)
        # overexpanded_no_background = overexpanded.copy()
        # overexpanded_no_background = np.where(gt_background_mask[:, :, None], background_color, overexpanded_no_background)

        # mask_25_overexpanded_no_background = np.all(overexpanded_no_background == color_25, axis=-1)

        # mask_25_substraction = np.logical_and(mask_25_gt, np.logical_not(mask_25_overexpanded_no_background))

        # overexpanded_diff = np.full((image.shape[0], image.shape[1], 3), fill_value=background_color, dtype=np.uint8)
        # overexpanded_diff[mask_25_substraction] = color_25

        if args.gt_images_colored:
            if not rgb_flag:
                gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
                gt_image_rgb = np.zeros((gt_image.shape[0], gt_image.shape[1], 3), dtype=np.uint8)
                for l in np.unique(gt_image):
                    color = label_colors.get(label, (0, 0, 0))
                    gt_image_rgb[gt_image == l] = color
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

        if color_dict is not None:
            background_color = color_dict.get(background, (background, background, background))
            color_mask_sam = np.full((image.shape[0], image.shape[1], 3), fill_value=background_color, dtype=np.uint8)
            color_mask_spx = np.full((image.shape[0], image.shape[1], 3), fill_value=background_color, dtype=np.uint8)
            color_mask_mix = np.full((image.shape[0], image.shape[1], 3), fill_value=background_color, dtype=np.uint8)
            mask_color_dir_sam = os.path.join(output_dir, 'labels_rgb_sam')
            mask_color_dir_spx = os.path.join(output_dir, 'labels_rgb_spx')
            mask_color_dir_mix = os.path.join(output_dir, 'labels_rgb_mix')
            os.makedirs(mask_color_dir, exist_ok=True)
            os.makedirs(mask_color_dir_sam, exist_ok=True)
            os.makedirs(mask_color_dir_spx, exist_ok=True)
            os.makedirs(mask_color_dir_mix, exist_ok=True)

        for l in unique_labels_i:
            expanded_i_sam = expanded_sam[expanded_sam['Label'] == l].iloc[:, 1:3].to_numpy().astype(int) + BORDER_SIZE
            expanded_i_spx = expanded_spx[expanded_spx['Label'] == l].iloc[:, 1:3].to_numpy().astype(int)
            expanded_i_mix = output_df[output_df['Label'] == l].iloc[:, 1:3].to_numpy().astype(int)

            mask_sam[expanded_i_sam[:, 0], expanded_i_sam[:, 1]] = l
            mask_spx[expanded_i_spx[:, 0], expanded_i_spx[:, 1]] = l
            mask_mix[expanded_i_mix[:, 0], expanded_i_mix[:, 1]] = l

            if color_dict is not None:
                l_str = list(color_dict.keys())[l]
                color = np.array(color_dict[l_str])
                color_mask_sam[expanded_i_sam[:, 0], expanded_i_sam[:, 1]] = color
                color_mask_spx[expanded_i_spx[:, 0], expanded_i_spx[:, 1]] = color
                color_mask_mix[expanded_i_mix[:, 0], expanded_i_mix[:, 1]] = color

        # Save grayscale masks as PNG
        cv2.imwrite(os.path.join(mask_dir_sam, os.path.splitext(image_name)[0] + '.png'), mask_sam)
        cv2.imwrite(os.path.join(mask_dir_spx, os.path.splitext(image_name)[0] + '.png'), mask_spx)
        cv2.imwrite(os.path.join(mask_dir_mix, os.path.splitext(image_name)[0] + '.png'), mask_mix)

        if color_dict is not None:
            # Save color masks as PNG
            color_mask_sam = cv2.cvtColor(color_mask_sam, cv2.COLOR_RGB2BGR)
            color_mask_spx = cv2.cvtColor(color_mask_spx, cv2.COLOR_RGB2BGR)
            color_mask_mix = cv2.cvtColor(color_mask_mix, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(mask_color_dir_sam, os.path.splitext(image_name)[0] + '.png'), color_mask_sam)
            cv2.imwrite(os.path.join(mask_color_dir_spx, os.path.splitext(image_name)[0] + '.png'), color_mask_spx)
            cv2.imwrite(os.path.join(mask_color_dir_mix, os.path.splitext(image_name)[0] + '.png'), color_mask_mix)

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
