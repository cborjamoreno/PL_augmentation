
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

def generate_image(image_df, image, points, labels, color_dict, image_name, output_dir, label_str=None):
    if label_str is None:
        label_str = 'ALL'
    
    # Create a black image
    black = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Create a copy of the image
    black = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # Convert the image to float type
    black = black.astype(float)

    black = black / 255

    # Convert the image to an RGBA image
    black = np.dstack((black, np.ones((black.shape[0], black.shape[1]))))

    black2 = black.copy()

    # Initialize a dictionary to store the labels of the points
    point_labels = {}

    # Create a dictionary to count the number of labels for each point
    label_counts = {}

    for index, row in image_df.iterrows():
        point = (row['Row']+200, row['Column']+200)
        if point in label_counts:
            label_counts[point] += 1
        else:
            label_counts[point] = 1

    # Color the points in the image
    for index, row in image_df.iterrows():
        # if row and column are in the image bounds
        if row['Column'] < black.shape[1] and row['Row'] < black.shape[0]:
            point = (row['Row']+200, row['Column']+200)

            # If the point has more than one label or has already been labeled, skip coloring
            if label_counts[point] > 1 or point in point_labels:
                continue

            # Get the color and add an alpha channel
            color = np.array([color_dict[row['Label']][0]/255.0, color_dict[row['Label']][1]/255.0, color_dict[row['Label']][2]/255.0, 1])
            blend_translucent_color(black, point[0], point[1], color, color[3])

            # Store the label of the point
            point_labels[point] = row['Label']

    plt.figure(figsize=(10,10))
    plt.imshow(black)
    show_points(points, labels, plt.gca(), marker_size=100, marker_color='black', edge_color='yellow')
    plt.axis('off')

    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir+image_name+'_'+label_str+'_expanded.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    plt.figure(figsize=(10,10))
    plt.imshow(black2)
    show_points(points, labels, plt.gca(), marker_size=100, marker_color='black', edge_color='yellow')
    plt.axis('off')

    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir+image_name+'_'+label_str+'_sparse.png', bbox_inches='tight', pad_inches=0)

    # plt.show()


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
    def __init__(self, color_dict, input_df, image_dir, output_df, output_dir): 
        self.color_dict = color_dict
        self.input_df = input_df
        self.image_dir = image_dir
        self.image_names_csv = input_df['Name'].unique()
        self.image_names_dir = os.listdir(image_dir)
        self.output_dir = output_dir
        self.output_df = output_df
        self.gt_points = np.array([])
        self.gt_labels = np.array([])
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
        for image_name in self.image_names_csv:
            image_path = os.path.join(self.image_dir, image_name + '.jpg')
            image = cv2.imread(image_path)

            if image is None:
                continue

            # crop the image 200 pixels from each side
            cropped_image = image[200:image.shape[0]-200, 200:image.shape[1]-200]
            
            if cropped_image is None:
                print(f"Failed to load image at {image_path}")
                continue

            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                
            points = self.input_df[self.input_df['Name'] == image_name].iloc[:, 1:3].to_numpy().astype(int)
            labels = self.input_df[self.input_df['Name'] == image_name].iloc[:, 3].to_numpy()
            unique_labels_str = np.unique(labels)
            unique_labels = range(len(unique_labels_str))
            label_dict = {}
            for i in range(len(unique_labels_str)):
                label_dict[unique_labels_str[i]] = unique_labels[i]

            print('Starting prediction for image', image_name)
            # plot_mosaic(image, unique_labels_str, 'Sebens_MA_LTM/output_test/several_points/')

            image_df = self.expand_labels(points, labels, unique_labels_str, image, cropped_image, image_name)
            
            generate_image(image_df, image, self.gt_points, self.gt_labels, color_dict, image_name, output_dir)
        
    @abstractmethod
    def expand_labels(self, points, labels, unique_labels_str, image, cropped_image, image_name):
        pass

class SAMLabelExpander(LabelExpander):
    def __init__(self, color_dict, input_df, image_dir, output_df, output_dir, predictor):
        super().__init__(color_dict, input_df, image_dir, output_df, output_dir)
        self.predictor = predictor
        

    def expand_labels(self, points, labels, unique_labels_str, image, cropped_image, image_name):
        expanded_df = pd.DataFrame(columns=["Name", "Row", "Column", "Label"])
        predictor.set_image(cropped_image)

        # SAM-specific label expansion code here# use SAM to expand the points into masks. Artificial new GT points.
        for i in range(len(unique_labels_str)):
            # print(f"Label {i+1}: {unique_labels_str[i]}")
            # get the points with the label

            # if unique_labels_str[i] != 'DROE':
            #     continue
            
            _points = points[labels==unique_labels_str[i]]

            # # Add points with different labels
            # different_labels_points = points[labels!=unique_labels_str[i]]
            # _points = np.concatenate((_points, different_labels_points), axis=0)
            _points = _points[(_points[:, 0] > 200) & (_points[:, 0] < cropped_image.shape[0]-200) & (_points[:, 1] > 200) & (_points[:, 1] < cropped_image.shape[1]-200)]

            if len(_points) == 0:
                print(f"No points for label {unique_labels_str[i]}")
                continue
            # Transform the points after cropping

            # change x and y coordinates
            _points = np.flip(_points, axis=1)

            # store _points in gt_points
            if len(self.gt_points) == 0:
                self.gt_points = _points
            else:
                self.gt_points = np.concatenate((self.gt_points, _points), axis=0)

            _labels = np.ones(len(_points), dtype=int)

            # store _labels in gt_labels
            if len(self.gt_labels) == 0:
                self.gt_labels = _labels
            else:
                self.gt_labels = np.concatenate((self.gt_labels, _labels), axis=0)

            _points_pred = _points.copy()
            _points_pred[:, 0] -= 200
            _points_pred[:, 1] -= 200

            best_mask = np.zeros((cropped_image.shape[0], cropped_image.shape[1]), dtype=bool)

            for p, l in zip(_points_pred, _labels):
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array([p]),
                    point_labels=np.array([l]),
                    multimask_output=True,
                    # return_logits=True
                )
                        
                # Merge masks[0] with best_mask
                # masks[0] = masks[0] > 2.0
                best_mask = np.logical_or(best_mask, masks[0])
            
            # remove far points from gt_points
            height, width = best_mask.shape
            far_mask = np.zeros((height, width), dtype=bool)

            min_distance = np.inf
            max_distance = -np.inf

            _points_pred[:, 0] += 200
            _points_pred[:, 1] += 200

            # for idx in np.argwhere(best_mask):
            #     k, j = idx
            #     p = np.array([k, j])
            #     is_far = True
            #     for gt_p in _points_pred:
            #         distance = np.linalg.norm(p - np.array(gt_p))
            #         min_distance = min(min_distance, distance)
            #         max_distance = max(max_distance, distance)
            #         if distance <= 200.0:
            #             is_far = False
            #             break
            #     far_mask[k, j] = is_far

            # print("Minimum distance:", min_distance)
            # print("Maximum distance:", max_distance)
            # print("Number of far points:", np.sum(far_mask))

            # best_mask[far_mask] = 0
            

            # add new points to the new_points array
            new_points = np.argwhere(best_mask)

            # add the new points to the output dataframe
            data = []
            for point in new_points:
                data.append({
                    "Name": image_name,
                    "Row": point[0],
                    "Column": point[1],
                    "Label": unique_labels_str[i]
                })

            new_data_df = pd.DataFrame(data)


            generate_image(new_data_df, image, _points_pred, _labels, color_dict, image_name, output_dir, unique_labels_str[i])
            
            expanded_df = pd.concat([expanded_df, new_data_df], ignore_index=True)
            # output_df = output_df.concat(new_data_df, ignore_index=True)
            
            print(f'{len(_points_pred)} {unique_labels_str[i]} points expanded to {len(new_points)}')
            # print(f'image_df has {len(image_df)} points')

        
        # check if there are two rows with the same point and different labels
        # print(expanded_df.duplicated(subset=['Row', 'Column'], keep=False))
        # print(expanded_df[expanded_df.duplicated(subset=['Row', 'Column'], keep=False)])

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

    def expand_labels(self, points, labels, unique_labels_str, image, cropped_image, image_name):
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
        show_points(points, labels, plt.gca(), marker_size=100, marker_color='black', edge_color='yellow')
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
args = parser.parse_args()

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
    LabelExpander = SAMLabelExpander(color_dict, input_df, image_dir, output_df, output_dir, predictor)
elif args.model == "superpixel":
    LabelExpander = SuperpixelLabelExpander(color_dict, input_df, image_dir, output_df, output_dir)
else:
    print("Invalid model option. Please choose either 'sam' or 'superpixel'.")
    sys.exit(1)

LabelExpander.expand_images()

print('Finished expanding images')


