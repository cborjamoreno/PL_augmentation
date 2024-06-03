
import argparse
import random
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

# def estimate_density(points, bandwidth=1.0, subsample_rate=0.1, n_jobs=4):
#     n_points = len(points)
#     n_subsample = max(1, int(n_points * subsample_rate))
    
#     # Randomly sample a subset of indices
#     subsample_indices = np.random.choice(n_points, n_subsample, replace=False)
#     subsample_points = points[subsample_indices]
    
#     # Split the indices for parallel processing
#     split_indices = np.array_split(subsample_indices, n_jobs)
    
#     # Create a pool of processes
#     with mp.Pool(n_jobs) as pool:
#         async_results = []
#         for split in tqdm(split_indices, desc="Estimating density"):
#             async_results.append(pool.apply_async(kde_subsample, args=(points, bandwidth, split)))
        
#         # Get results from each process
#         kde_list = [result.get() for result in async_results]
    
#     # Aggregate results from each process
#     start_time = time.time()
#     densities = np.zeros(n_points)
#     for kde in kde_list:
#         densities += np.exp(kde.score_samples(points))
#     end_time = time.time()

#     print(f"Time taken: {end_time - start_time} seconds")
    
#     # Average the densities
#     densities /= n_jobs
#     return densities

def estimate_density(points, bandwidth=1.0, subsample_rate=0.1, n_jobs=4):
    n_points = len(points)
    n_subsample = max(1, int(n_points * subsample_rate))
    
    # Randomly sample a subset of indices
    subsample_indices = np.random.choice(n_points, n_subsample, replace=False)
    subsample_points = points[subsample_indices]
    
    # Fit KDE on the subsample
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(subsample_points)
    
    # Compute densities for all points
    start_time = time.time()
    densities = np.exp(kde.score_samples(points))
    end_time = time.time()

    print(f"Time taken: {end_time - start_time} seconds")
    
    return densities

def adjust_priority(args):
    label, priorities, image_df, densities, density_factor, max_priority = args
    label_indices = image_df[image_df['Label'] == label].index
    label_density = densities[label_indices].mean()  # Average density for the label
    adjusted_priority = max_priority + 1 - priorities[label] + density_factor * label_density
    return label, adjusted_priority

def adjust_priorities_with_density(image_df, densities, density_factor=0.5):
    priorities = image_df['Label'].value_counts()
    max_priority = priorities.max()

    with mp.Pool(mp.cpu_count()) as pool:
        args = [(label, priorities, image_df, densities, density_factor, max_priority) for label in priorities.index]
        label_priorities = dict(pool.map(adjust_priority, tqdm(args, desc="Adjusting priorities")))

    return label_priorities

def generate_image(image_df, image, points, labels, color_dict, image_name, output_dir, label_str=None):
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
    show_points(points, labels, plt.gca(), marker_color='black', edge_color='yellow')
    plt.axis('off')

    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_dir + image_name + '_' + label_str + '_sparse.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Initialize a dictionary to store the labels of the points
    point_labels = {}

    # Create a dictionary to count the number of labels for each point
    label_counts = {}

    for _, row in image_df.iterrows():
        point = (row['Row'] + BORDER_SIZE, row['Column'] + BORDER_SIZE)
        if point in label_counts:
            label_counts[point] += 1
        else:
            label_counts[point] = 1


    # Start the timer
    start_time = time.time()

    points_array = image_df[['Row', 'Column']].values
    print('len(points_array):', len(points_array))
    bandwidth  = 1.0
    densities = estimate_density(points_array, bandwidth=bandwidth, subsample_rate=0.001, n_jobs=mp.cpu_count())
    label_priorities = adjust_priorities_with_density(image_df, densities, density_factor=1)

    # Stop the timer
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"The estimation of priorities took {elapsed_time} seconds.")

    # Color the points in the image
    for _, row in image_df.iterrows():
        if row['Column'] < black.shape[1] and row['Row'] < black.shape[0]:
            point = (row['Row'] + BORDER_SIZE, row['Column'] + BORDER_SIZE)

            # If the point has more than one label or has already been labeled, check priorities
            if label_counts[point] > 1 or point in point_labels:
                # If the current label has a higher priority, color the point with the current label
                if label_priorities.get(row['Label'], 0) > label_priorities.get(point_labels.get(point, None), 0):
                    # Get the color and add an alpha channel
                    color = np.array([color_dict[row['Label']][0] / 255.0, color_dict[row['Label']][1] / 255.0, color_dict[row['Label']][2] / 255.0, 1])
                    blend_translucent_color(black, point[0], point[1], color, color[3])
                    point_labels[point] = row['Label']
            else:
                # Get the color and add an alpha channel
                color = np.array([color_dict[row['Label']][0] / 255.0, color_dict[row['Label']][1] / 255.0, color_dict[row['Label']][2] / 255.0, 1])
                blend_translucent_color(black, point[0], point[1], color, color[3])
                point_labels[point] = row['Label']

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(black)
    show_points(points, labels, plt.gca(), marker_color='black', edge_color='yellow')
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
    color_dict_df.to_csv(output_dir + 'color_dict.csv', index=False, header=True)

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
    def __init__(self, color_dict, input_df, image_dir, output_df, output_dir, remove_far_points=False, generate_eval_images=False): 
        self.color_dict = color_dict
        self.input_df = input_df
        self.image_dir = image_dir
        self.image_names_csv = input_df['Name'].unique()
        self.image_names_dir = os.listdir(image_dir)
        self.output_dir = output_dir
        self.output_df = output_df
        self.gt_points = np.array([])
        self.gt_labels = np.array([])
        self.remove_far_points = remove_far_points
        self.generate_eval_images = generate_eval_images
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
        # Get number of unique labels
        unique_labels_str = self.input_df['Label'].unique()
        print('Number of unique labels:', len(unique_labels_str))

        # Get the labels that are in self.color_dict.keys() but not in unique_labels_str
        extra_labels = set(self.color_dict.keys()) - set(unique_labels_str)

        # Remove these labels from self.color_dict
        for label in extra_labels:
            del self.color_dict[label]

        if set(unique_labels_str) != set(self.color_dict.keys()):
            print('Labels in the .csv file and color_dict do not match')
            print('Creating a new color_dict')

            # Print the labels that are in unique_labels_str but not in self.color_dict.keys()
            print('Labels in unique_labels_str but not in color_dict:', set(unique_labels_str) - set(self.color_dict.keys()))

            # Print the labels that are in self.color_dict.keys() but not in unique_labels_str
            print('Labels in color_dict but not in unique_labels_str:', set(self.color_dict.keys()) - set(unique_labels_str))

            colors = generate_distinct_colors(len(unique_labels_str))
            color_dict = {label: color for label, color in zip(unique_labels_str, colors)}
            export_colors(color_dict, self.output_dir)
            self.color_dict = color_dict

        # Order unique_labels_str in the way that they appear in color_dict
        unique_labels_str = [label for label in self.color_dict.keys() if label in unique_labels_str]

        if self.output_dir[-1] != '/':
            self.output_dir += '/'

        mask_dir = self.output_dir + 'labels/'
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        for image_name in self.image_names_csv:

            image_path = os.path.join(self.image_dir, image_name + '.jpg')
            image = cv2.imread(image_path)

            if image is None:
                continue

            points = self.input_df[self.input_df['Name'] == image_name].iloc[:, 1:3].to_numpy().astype(int)
            labels = self.input_df[self.input_df['Name'] == image_name].iloc[:, 3].to_numpy()
            unique_labels_str_i = np.unique(labels)

            output_dir = self.output_dir + image_name + '/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # reset gt_points and gt_labels
            self.gt_points = np.array([])
            self.gt_labels = np.array([])

            if image is None:
                print(f"Failed to load image at {image_path}")
                continue

            print('Starting prediction for image', image_name)

            image_df = self.expand_labels(points, labels, unique_labels_str_i, image, image_name, output_dir, self.remove_far_points, generate_eval_images=self.generate_eval_images)
            
            if generate_eval_images:
                generate_image(image_df, image, self.gt_points, self.gt_labels, self.color_dict, image_name, output_dir)

            # plot each of the classes of the image in grayscale from 1 to the number of classes. 0 is for the pixeles that are not in any class
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=float)

            point_labels = {}
            for i, label in enumerate(unique_labels_str, start=1):
                expanded_i = image_df[image_df['Label'] == label].iloc[:, 1:3].to_numpy().astype(int)
                for point in expanded_i:
                    point = point + BORDER_SIZE
                    point_tuple = tuple(point)
                    mask[point[0], point[1]] = i
                    point_labels[point_tuple] = label
                
            cv2.imwrite(mask_dir+image_name+'_labels.png', mask)
        
    @abstractmethod
    def expand_labels(self, points, labels, unique_labels_str, image, image_name, output_dir, remove_far_points=False, generate_eval_images=False):
        pass

class SAMLabelExpander(LabelExpander):
    def __init__(self, color_dict, input_df, image_dir, output_df, output_dir, predictor, remove_far_points=False, generate_eval_images=False):
        super().__init__(color_dict, input_df, image_dir, output_df, output_dir, remove_far_points, generate_eval_images)
        self.predictor = predictor
        

    def expand_labels(self, points, labels, unique_labels_str, image, image_name, output_dir, remove_far_points=False, generate_eval_images=False):
        expanded_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label"])

        # crop the image BORDER_SIZE pixels from each side
        cropped_image = image[BORDER_SIZE:image.shape[0]-BORDER_SIZE, BORDER_SIZE:image.shape[1]-BORDER_SIZE]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(cropped_image)

        # SAM-specific label expansion code here# use SAM to expand the points into masks. Artificial new GT points.
        for i in range(len(unique_labels_str)):

            # if unique_labels_str[i] == 'LIT':
            #     continue
            
            _points = points[labels==unique_labels_str[i]]
            _points = _points[(_points[:, 0] > BORDER_SIZE) & (_points[:, 0] < image.shape[0] - BORDER_SIZE) & (_points[:, 1] > BORDER_SIZE) & (_points[:, 1] < image.shape[1] - BORDER_SIZE)]

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

            _labels = np.ones(len(_points), dtype=int)

            # Store _labels in gt_labels
            if len(self.gt_labels) == 0:
                self.gt_labels = _labels
            else:
                self.gt_labels = np.concatenate((self.gt_labels, _labels), axis=0)

            _points_pred = _points.copy()
            _points_pred[:, 0] -= BORDER_SIZE
            _points_pred[:, 1] -= BORDER_SIZE

            best_mask = np.zeros((cropped_image.shape[0], cropped_image.shape[1]), dtype=bool)

            for p, l in zip(_points_pred, _labels):
                _, _, logits = self.predictor.predict(
                    point_coords=np.array([p]),
                    point_labels=np.array([l]),
                    multimask_output=True,
                )

                # best_mask = np.logical_or(best_mask1, masks[0])
                
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
                generate_image_per_class(new_data_df, image, _points_pred, _labels, color_dict, image_name, output_dir, unique_labels_str[i])
            
            expanded_df = pd.concat([expanded_df, new_data_df], ignore_index=True)            
            print(f'{len(_points_pred)} {unique_labels_str[i]} points expanded to {len(new_points)}')

        return expanded_df

class SuperpixelLabelExpander(LabelExpander):
    def __init__(self, color_dict, input_df, image_dir, output_df, output_dir):
        super().__init__(color_dict, input_df, image_dir, output_df, output_dir)
        self.sparseImage = None
        self.gray_RGBCorrelation = {}
    def createSparseImage(self, points, labels, gray_dict, image_shape=(1000, 1000)):
        # Create a white image with the specified shape
        sparse_image = np.ones((image_shape[0], image_shape[1], 3), dtype=np.uint8) * 255
        
        # Iterate over the points and labels
        for point, label in zip(points, labels):
            # Get the grayscale value from the gray_dict
            gray_value = gray_dict[label]
            
            # Set the grayscale value at the corresponding point in the sparse image
            sparse_image[point[1], point[0]] = [gray_value, gray_value, gray_value]
        
        return sparse_image

    def expand_labels(self, points, labels, unique_labels_str, image, image_name):

        # crop the image BORDER_SIZE pixels from each side
        cropped_image = image[BORDER_SIZE:image.shape[0]-BORDER_SIZE, BORDER_SIZE:image.shape[1]-BORDER_SIZE]
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(cropped_image)

        for i, label in enumerate(unique_labels_str):
            self.gray_RGBCorrelation[i] = self.color_dict[label]

        print('gray_RGBCorrelation:', self.gray_RGBCorrelation)

        self.sparseImage = self.createSparseImage(points, labels, self.gray_RGBCorrelation, cropped_image.shape)
        
        new_filename = "ML-Superpixels/Sebens_MA_LTM/sparse_GT/test/" + image_name + ".png"

        # Save the image
        cv2.imwrite(new_filename, self.sparseImage)

        # Call Superpixel-Expansion
        os.system(f"python generate_augmented_GT/generate_augmented_GT.py --dataset ML-Superpixels/Datasets/camvid --number_levels 15 --start_n_superpixels 3000 --last_n_superpixels 30")

        # Load the expanded image in grayscale
        expanded_image = cv2.imread("ML-Superpixels/Sebens_MA_LTM/augmented_GT/" + image_name + ".png", cv2.IMREAD_GRAYSCALE)

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
        # plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="Directory containing images", required=True)
parser.add_argument("--output_dir", help="Directory to save the output images", required=True)
parser.add_argument("--ground-truth", help="CSV file containing the points and labels", required=True)
parser.add_argument("--model", help="Model to use for prediction", default="sam", choices=["sam", "superpixel"])
parser.add_argument("--max_distance", help="Maximum distance between expanded points and the seed", type=int)
parser.add_argument("--generate_eval_images", help="Generate evaluation images for the expansion (sparse and expanded images for all the classes)", required=False)
args = parser.parse_args()

remove_far_points = False
generate_eval_images = False

if args.generate_eval_images:
    generate_eval_images = True

if args.max_distance:
    MAX_DISTANCE = args.max_distance
    remove_far_points = True
image_dir = args.input_dir

output_dir = args.output_dir

# Get input points and labels from csv file
input_df = pd.read_csv(args.ground_truth)
output_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label"])

# Generate distinct colors for each label
# colors = generate_distinct_colors(len(unique_labels))

# Create a dictionary mapping labels to colors
# color_dict = {label: color for label, color in zip(unique_labels, colors)}

# Export the color_dict to a file
# export_colors(color_dict, output_dir)

# Import the color_dict from a file
color_dict = pd.read_csv('color_dict.csv').to_dict()

if args.model == "sam":
    device = "cuda"
    model = sam_model_registry["vit_h"](checkpoint="vit_h.pth")
    predictor = SamPredictor(model)
    # mask_generator = SamAutomaticMaskGenerator(model=model, points_per_side=32)
    LabelExpander = SAMLabelExpander(color_dict, input_df, image_dir, output_df, output_dir, predictor, remove_far_points, generate_eval_images)
elif args.model == "superpixel":
    LabelExpander = SuperpixelLabelExpander(color_dict, input_df, image_dir, output_df, output_dir, generate_eval_images)
else:
    print("Invalid model option. Please choose either 'sam' or 'superpixel'.")
    sys.exit(1)

LabelExpander.expand_images()

print('Finished expanding images')


