
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
import multiprocessing as mp
from tqdm import tqdm 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BORDER_SIZE = 150
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
    
def show_points(coords, labels, ax, marker_size=375, marker_color='blue', edge_color='white'):
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

def kde_subsample(points, bandwidth, subsample_indices):
    subsample_points = points[subsample_indices]
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(subsample_points)
    return kde

def estimate_density(points, bandwidth=1.0, n_jobs=4):
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(points)
    return kde

def evaluate_density(kde, points):
    densities = np.exp(kde.score_samples(points))
    return densities

def adjust_priority(args):
    label, priorities, image_df, densities, density_factor, max_priority, density_scale = args
    label_indices = image_df[image_df['Label'] == label].index
    label_density = densities[label_indices].mean() * density_scale  # Average density for the label
    print(f"Label {label}: Density - {label_density}")
    print(f"Label {label}: Priority - {max_priority + 1 - priorities[label]}")
    adjusted_priority = max_priority + 1 - priorities[label] + density_factor * label_density
    return label, adjusted_priority

def adjust_priorities_with_density(image_df, densities, density_factor=0.5):
    priorities = image_df['Label'].value_counts()
    max_priority = priorities.max()
    print('priorities:', priorities)

    # Calculate scaling factor for densities
    mean_priorities = priorities.mean()
    std_priorities = priorities.std()
    mean_densities = densities.mean()
    std_densities = densities.std()

    density_scale = std_priorities / std_densities if std_densities != 0 else 1

    with mp.Pool(mp.cpu_count()) as pool:
        args = [(label, priorities, image_df, densities, density_factor, max_priority, density_scale) for label in priorities.index]
        label_priorities = dict(pool.map(adjust_priority, tqdm(args, desc="Adjusting priorities")))

    return label_priorities

def merge_labels(image_df, gt_points, gt_labels):
    merged_df = pd.DataFrame(columns=['Row', 'Column', 'Label'])
    point_labels = {}
    label_counts = {}
    unique_labels = np.unique(gt_labels)
    kde_dict = {label: estimate_density(gt_points[gt_labels == label], bandwidth=1.0) for label in unique_labels}
    assigned_densities = np.zeros(len(image_df))

    for label in unique_labels:
        label_indices = image_df['Label'] == label
        expanded_label_points = image_df.loc[label_indices, ['Row', 'Column']].values + BORDER_SIZE
        if expanded_label_points.size == 0:
            continue
        densities_label = evaluate_density(kde_dict[label], expanded_label_points)
        assigned_densities[label_indices] = densities_label

    label_priorities = adjust_priorities_with_density(image_df, assigned_densities, density_factor=0.5)

    points_to_add = []
    for index, row in image_df.iterrows():
        adjusted_point = (row['Row'] + BORDER_SIZE, row['Column'] + BORDER_SIZE)
        label = row['Label']
        label_count = label_counts.get(adjusted_point, 0) + 1
        label_counts[adjusted_point] = label_count
        if label_count > 1 and label_priorities[label] > label_priorities.get(point_labels.get(adjusted_point, None), -1):
            point_labels[adjusted_point] = label
        elif label_count == 1:
            point_labels[adjusted_point] = label

    for point, label in point_labels.items():
        row, column = point
        points_to_add.append({'Row': row - BORDER_SIZE, 'Column': column - BORDER_SIZE, 'Label': label})

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

    # Color the points in the image
    for _, row in image_df.iterrows():
        if row['Column'] < black.shape[1] and row['Row'] < black.shape[0]:
            point = (row['Row'] + BORDER_SIZE, row['Column'] + BORDER_SIZE)

            # Get the color and add an alpha channel
            color = np.array([color_dict[row['Label']][0] / 255.0, color_dict[row['Label']][1] / 255.0, color_dict[row['Label']][2] / 255.0, 1])
            blend_translucent_color(black, point[0], point[1], color, color[3])

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(black)
    show_points(gt_points, np.ones(len(gt_points), dtype=int), plt.gca(), marker_color='black', edge_color='yellow')
    plt.axis('off')

    plt.savefig(output_dir + image_name + '_' + label_str + '_expanded.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def generate_image_per_class(image_df, image, points, labels, color_dict, image_name, output_dir, label_str):
    
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

    plt.savefig(output_dir + image_name + '_' + label_str + '_expanded.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
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

class LabelExpander(ABC):
    def __init__(self, color_dict, input_df, labels, image_dir, output_df, output_dir, remove_far_points=False, generate_eval_images=False, get_sparse_csv=False): 
        self.color_dict = color_dict
        self.input_df = input_df
        self.unique_labels_str = labels
        self.image_dir = image_dir
        self.image_names_csv = input_df['Name'].unique()
        self.image_names_dir = os.listdir(image_dir)
        self.output_dir = output_dir
        self.output_df = output_df
        self.gt_points = np.array([])
        self.gt_labels = np.array([])
        self.remove_far_points = remove_far_points
        self.generate_eval_images = generate_eval_images
        self.generate_csv = generate_csv
        self.checkMissmatchInFiles()

    def checkMissmatchInFiles(self):
        # Print images in .csv that are not in the image directory
        for image_name in self.image_names_csv:
            if image_name + '.jpg' not in self.image_names_dir:
                print(f"Image {image_name} in .csv but not in image directory")

        # Print images in the image directory that are not in the .csv
        for image_name in self.image_names_dir:
            if image_name[:-4] not in self.image_names_csv:
                print(f"Image {image_name} in image directory but not in .csv")

    def expand_images(self):

        if self.generate_csv:
            sparse_df = pd.DataFrame(columns=['Name', 'Row', 'Column', 'Label'])
        
        mask_dir = self.output_dir + 'labels/'
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        eval_images_dir = self.output_dir + 'eval_images/'
        if not os.path.exists(eval_images_dir):
            os.makedirs(eval_images_dir)

        for image_name in self.image_names_csv:

            image_path = os.path.join(self.image_dir, image_name + '.jpg')
            image = cv2.imread(image_path)

            if image is None:
                continue

            points = self.input_df[self.input_df['Name'] == image_name].iloc[:, 1:3].to_numpy().astype(int)
            labels = self.input_df[self.input_df['Name'] == image_name].iloc[:, 3].to_numpy()
            unique_labels_str_i = np.unique(labels)

            eval_images_dir_i = eval_images_dir + image_name + '/'

            # reset gt_points and gt_labels
            self.gt_points = np.array([])
            self.gt_labels = np.array([])

            if image is None:
                print(f"Failed to load image at {image_path}")
                continue

            print('Starting prediction for image', image_name)

            image_df = self.expand_labels(points, labels, unique_labels_str_i, image, image_name, eval_images_dir_i, self.remove_far_points, generate_eval_images=self.generate_eval_images)

            # Merge the dense labels
            merged_df = merge_labels(image_df, self.gt_points, self.gt_labels)

            if self.generate_eval_images:
                generate_image(merged_df, image, self.gt_points, self.color_dict, image_name, eval_images_dir_i)

            if self.generate_csv:

                # TODO: Perform Kmeans clustering separately for each class
                if not image_df.empty:
                    num_clusters = len(unique_labels_str_i)
                    kmeans = KMeans(n_clusters=num_clusters)
                    image_df['Cluster'] = kmeans.fit_predict(image_df[['Row', 'Column']])

                    # Count the number of points in each cluster
                    cluster_counts = image_df['Cluster'].value_counts()

                    # Calculate the sampling rate for each class (inverse of class count)
                    sampling_rates = 1 / cluster_counts

                    # Normalize the sampling rates so that they sum up to 1
                    sampling_rates = sampling_rates / sampling_rates.sum()

                    # Calculate the number of points to sample from each class
                    points_per_cluster = (sampling_rates * 950).round().astype(int)

                    # Create an empty DataFrame to store the sampled points
                    sampled_df = pd.DataFrame(columns=['Name', 'Row', 'Column', 'Label'])

                    # Sample points from each cluster
                    for cluster in image_df['Cluster'].unique():
                        cluster_points = image_df[image_df['Cluster'] == cluster]
                        num_points = points_per_cluster[cluster]
                        sampled_points = cluster_points.sample(min(len(cluster_points), num_points))
                        sampled_df = pd.concat([sampled_df, sampled_points])

                    # If there are still points left to sample, sample randomly from the entire DataFrame
                    if len(sampled_df) < 950:
                        remaining_points = 950 - len(sampled_df)
                        additional_points = image_df.sample(remaining_points)
                        sampled_df = pd.concat([sampled_df, additional_points])
                        
                    sparse_df = pd.concat([sparse_df, sampled_df])
                else:
                    print("Warning: image_df is empty. Skipping KMeans clustering and sampling.")

            # plot each of the classes of the image in grayscale from 1 to the number of classes. 0 is for the pixeles that are not in any class
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=float)

            point_labels = {}
            for i, label in enumerate(self.unique_labels_str, start=1):
                expanded_i = merged_df[merged_df['Label'] == label].iloc[:, 1:3].to_numpy().astype(int)
                for point in expanded_i:
                    point = point + BORDER_SIZE
                    point_tuple = tuple(point)
                    mask[point[0], point[1]] = i
                    point_labels[point_tuple] = label
                
            cv2.imwrite(mask_dir+image_name+'_labels.png', mask)

        if self.generate_csv:
                sparse_df.to_csv(self.output_dir + 'sparse.csv', index=False)
        
    @abstractmethod
    def expand_labels(self, points, labels, unique_labels_str, image, image_name, output_dir, remove_far_points=False, generate_eval_images=False):
        pass

class SAMLabelExpander(LabelExpander):
    def __init__(self, color_dict, input_df, labels, image_dir, output_df, output_dir, predictor, remove_far_points=False, generate_eval_images=False, generate_csv=False):
        super().__init__(color_dict, input_df, labels, image_dir, output_df, output_dir, remove_far_points, generate_eval_images, generate_csv)
        self.predictor = predictor
        

    def expand_labels(self, points, labels, unique_labels_str, image, image_name, output_dir, remove_far_points=False, generate_eval_images=False):
        expanded_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label"])

        # crop the image BORDER_SIZE pixels from each side
        cropped_image = image[BORDER_SIZE:image.shape[0]-BORDER_SIZE, BORDER_SIZE:image.shape[1]-BORDER_SIZE]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(cropped_image)

        # SAM-specific label expansion code here# use SAM to expand the points into masks. Artificial new GT points.
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

            for p, l in zip(_points_pred, _labels_ones):
                _, _, logits = self.predictor.predict(
                    point_coords=np.array([p]),
                    point_labels=np.array([l]),
                    multimask_output=True,
                )
                
                mask_input = logits[0, :, :] # Choose the model's best mask

                mask, _, _ = self.predictor.predict(
                    point_coords=np.array([p]),
                    point_labels=np.array([l]),
                    mask_input= mask_input[None, :, :],
                    multimask_output=True,
                )

                best_mask = np.logical_or(best_mask, mask[0])          
            
            min_distance = np.inf
            max_distance = -np.inf

            if remove_far_points:
                # Remove far points from gt_points
                height, width = best_mask.shape
                far_mask = np.zeros((height, width), dtype=bool)

                # Copy of best_mask before changes
                # best_mask_before = best_mask.copy()

                for idx in np.argwhere(best_mask):
                    row, col = idx
                    p = np.array([col, row])
                    distances = [np.linalg.norm(p - np.array(gt_p)) for gt_p in _points_pred]
                    min_distance = min(distances)
                    max_distance = max(max_distance, min_distance)
                    far_mask[row, col] = min_distance > MAX_DISTANCE

                best_mask[far_mask] = 0
            
            _points_pred[:, 0] += BORDER_SIZE
            _points_pred[:, 1] += BORDER_SIZE
            
            # _points_pred_show = _points_pred.copy()

            # Plotting
            # fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            # # Plot best_mask before changes
            # ax[0].imshow(best_mask_before, cmap='gray')
            # ax[0].scatter(_points_pred_show[:, 0], _points_pred_show[:, 1], color='red', s=1)  # plot _points_pred as red points
            # ax[0].set_title('Best Mask before changes '+unique_labels_str[i])

            # # Plot far_mask
            # ax[1].imshow(far_mask, cmap='gray')
            # ax[1].scatter(_points_pred_show[:, 0], _points_pred_show[:, 1], color='red', s=1)  # plot _points_pred as red points
            # ax[1].set_title('Far Mask with GT points '+unique_labels_str[i])

            # # Plot best_mask after changes
            # ax[2].imshow(best_mask, cmap='gray')
            # ax[2].scatter(_points_pred_show[:, 0], _points_pred_show[:, 1], color='red', s=1)  # plot _points_pred as red points
            # ax[2].set_title('Best Mask after changes '+unique_labels_str[i])

            # plt.show()

            # Add new points to the new_points array
            new_points = np.argwhere(best_mask)

            # Add the new points to the output dataframe
            data = []
            for point in new_points:
                data.append({
                    "Name": image_name,
                    "Row": point[0],
                    "Column": point[1],
                    "Label": unique_labels_str[i]
                })

            new_data_df = pd.DataFrame(data)

            
            if generate_eval_images:
                generate_image_per_class(new_data_df, image, _points_pred, _labels_ones, color_dict, image_name, output_dir, unique_labels_str[i])
            
            expanded_df = pd.concat([expanded_df, new_data_df], ignore_index=True)            
            print(f'{len(_points_pred)} {unique_labels_str[i]} points expanded to {len(new_points)}')

        return expanded_df

class SuperpixelLabelExpander(LabelExpander):
    def __init__(self, dataset, color_dict, input_df, labels, image_dir, output_df, output_dir, remove_far_points=False, generate_eval_images=False, generate_csv=False):
        super().__init__(color_dict, input_df, labels, image_dir, output_df, output_dir, remove_far_points, generate_eval_images, generate_csv)
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

    def expand_labels(self, points, labels, unique_labels_str, image, image_name, output_dir, remove_far_points=False, generate_eval_images=False):

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
        
        sparseImage = self.createSparseImage(_points, _labels, cropped_image.shape)
        # show the sparse image
        # plt.figure(figsize=(10,10))
        # plt.imshow(sparseImage)
        # plt.show()

        unique_values = np.unique(sparseImage)
        print(f"Unique values in sparse GT: {unique_values}")

        new_filename_path = "ML-Superpixels/"+self.dataset + "/sparse_GT/train/"
        if not os.path.exists(new_filename_path):
            os.makedirs(new_filename_path)
        new_filename = new_filename_path + image_name + ".png"

        # Delete existing dataset with the same name in ML-Superpixels/Datasets
        if os.path.exists("ML-Superpixels/Datasets/"+self.dataset):
            shutil.rmtree("ML-Superpixels/Datasets/"+self.dataset)

        # Create a new dataset for the images used
        os.makedirs("ML-Superpixels/Datasets/"+self.dataset+"/images/train")
        cv2.imwrite("ML-Superpixels/Datasets/"+self.dataset+"/images/train/"+image_name+".png", cropped_image)

        # Save the image
        cv2.imwrite(new_filename, sparseImage)

        print("Image saved at", new_filename)

        # Call Superpixel-Expansion
        os.system(f"python ML-Superpixels/generate_augmented_GT/generate_augmented_GT.py --dataset ML-Superpixels/Datasets/"+ self.dataset+" --number_levels 15 --start_n_superpixels 3000 --last_n_superpixels 30")

        print("Superpixel expansion done")

        # Load the expanded image in grayscale
        expanded_image = cv2.imread("ML-Superpixels/"+self.dataset+ "/augmented_GT/train/" + image_name + ".png", cv2.IMREAD_GRAYSCALE)

        #show the expanded image
        plt.figure(figsize=(10,10))
        plt.imshow(expanded_image)
        plt.show()

        # Convert to RGB using the gray-RGB correlation
        expanded_image_RGB = np.zeros((expanded_image.shape[0], expanded_image.shape[1], 3), dtype=np.uint8)
        for i in range(expanded_image.shape[0]):
            for j in range(expanded_image.shape[1]):
                expanded_image_RGB[i, j] = self.gray_RGBCorrelation[expanded_image[i, j]]

        # Save the expanded image
        plt.figure(figsize=(10,10))
        plt.imshow(expanded_image)
        show_points(points, labels, plt.gca(), marker_color='black', edge_color='yellow')
        plt.axis('off')

        label_str = 'ALL'
        if output_dir[-1] != '/':
            output_dir += '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_dir+image_name+'_'+label_str+'_expanded.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        # plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="Directory containing images", required=True)
parser.add_argument("--output_dir", help="Directory to save the output images", required=True)
parser.add_argument("--ground-truth", help="CSV file containing the points and labels", required=True)
parser.add_argument("--model", help="Model to use for prediction. If superpixel, the ML-Superpixels folder must be at the same path than this script.", default="sam", choices=["sam", "superpixel"])
parser.add_argument("--dataset", help="Dataset to use for superpixel expansion", required=False)
parser.add_argument("--max_distance", help="Maximum distance between expanded points and the seed", type=int)
parser.add_argument("--generate_eval_images", help="Generate evaluation images for the expansion (sparse and expanded images for all the classes)", required=False, action='store_true')
parser.add_argument("--color_dict", help="CSV file containing the color dictionary", required=False)
parser.add_argument("--generate_csv", help="Generate a sparse csv file", required=False, action='store_true')
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

image_dir = args.input_dir
if not os.path.exists(image_dir):
    parser.error(f"The directory {image_dir} does not exist")

output_dir = args.output_dir
if output_dir[-1] != '/':
    output_dir += '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if args.generate_eval_images:
    generate_eval_images = True
    labels = input_df['Label'].unique()
    if args.color_dict:
        color_dict = pd.read_csv(args.color_dict).to_dict()
        # Get the labels that are in self.color_dict.keys() but not in labels
        extra_labels = set(color_dict.keys()) - set(labels)

        # Remove the extra labels from color_dict
        for label in extra_labels:
            del color_dict[label]

        if set(labels) != set(color_dict.keys()):
            print('Labels in the .csv file and color_dict do not match:')
            print('     Labels in unique_labels_str but not in color_dict:', set(labels) - set(color_dict.keys()))
            print('     Labels in color_dict but not in unique_labels_str:', set(color_dict.keys()) - set(labels))

            print('Creating a new color_dict randomly...')
            color_dict = create_color_dict(labels, output_dir)
    else:
        print("--color_dict not provided. Colors will be generated randomly.")
        color_dict = create_color_dict(labels, output_dir)
    
    # Order unique_labels_str in the way that they appear in color_dict
    labels = [label for label in color_dict.keys() if label in labels]

if args.model == "sam":
    device = "cuda"
    model = sam_model_registry["vit_h"](checkpoint="vit_h.pth")
    predictor = SamPredictor(model)
    # mask_generator = SamAutomaticMaskGenerator(model=model, points_per_side=32)
    LabelExpander = SAMLabelExpander(color_dict, input_df, labels, image_dir, output_df, output_dir, predictor, remove_far_points, generate_eval_images, generate_csv)
elif args.model == "superpixel":
    dataset = args.dataset
    LabelExpander = SuperpixelLabelExpander(dataset, color_dict, input_df, labels, image_dir, output_df, output_dir, generate_eval_images)
else:
    print("Invalid model option. Please choose either 'sam' or 'superpixel'.")
    sys.exit(1)

image_df = LabelExpander.expand_images()

print('Finished expanding images')


