import glob
import heapq
import os
import pandas as pd
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import random
import argparse
from scipy.spatial import distance
from skimage import draw
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--images_pth", help="path of images", required=True)
    parser.add_argument("-gt", "--ground_truth_pth", help="path of ground truth images", required=True)
    parser.add_argument("-o", "--output_file", help="name of the output csv file without extension", required=True)
    parser.add_argument("-n", "--num_labels", help="number of labels to generate", default=300, type=int)
    parser.add_argument("-t", "--image_type", help="type of ground truth images (grayscale or color)", choices=['grayscale', 'color'], default='grayscale')
    parser.add_argument("-c", "--color_dict", help="path to color dictionary CSV file", required=False)
    return parser.parse_args()

def load_color_dict(color_dict_path):
    color_dict = {}
    df = pd.read_csv(color_dict_path)
    class_names = df.columns.tolist()
    for class_name in class_names:
        rgb_values = df[class_name].tolist()
        color_dict[tuple(rgb_values)] = class_name
    return color_dict

def show_anns(anns, img):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)
    for ann in sorted_anns:
        print(f"Area: {ann['area']}")
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor('white')  # Set the figure background color to white
        ax.set_facecolor('white')  # Set the axes background color to white
        m = ann['segmentation']
        # Set the color to yellow (R=1, G=1, B=0, A=1)
        color_mask = np.array([1, 1, 0, 1])
        # Create a black background only on the size of the image
        img_mask = np.zeros((img.shape[0], img.shape[1], 4))
        img_mask[m] = color_mask
        # Create a white canvas larger than the image
        canvas = np.ones((img.shape[0], img.shape[1], 4))
        canvas[:, :, :3] = 0  # Set RGB to white
        canvas[:, :, 3] = 1  # Set alpha to 1 for opacity
        # Overlay the black background and yellow mask on the white canvas
        canvas[img_mask[:, :, 3] > 0] = img_mask[img_mask[:, :, 3] > 0]
        ax.imshow(np.ones_like(img) * 255)  # Display the original image as the background
        ax.imshow(canvas)  # Overlay the mask with some transparency
        ax.axis('off')
        plt.show()

def calculate_centroid(mask):
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None
    centroid = np.mean(indices, axis=0).astype(int)
    return tuple(centroid)

def get_majority_label_in_circular_window(gt_img, point, radius, return_percentage=False, debug=False):
    x, y = point
    mask = np.zeros_like(gt_img, dtype=bool)
    rr, cc = draw.disk((x, y), radius, shape=gt_img.shape)
    mask[rr, cc] = True
    labels, counts = np.unique(gt_img[mask], return_counts=True)
    if debug:
        # plot the point and the circular window
        plt.figure(figsize=(10, 10))
        plt.imshow(gt_img)
        plt.scatter(y, x, c='red', marker='o')  # Note: (y, x) for plotting
        plt.scatter(cc, rr, c='blue', marker='o', alpha=0.5)  # Note: (cc, rr) for plotting
        plt.title(f'Point and Circular Window (Radius={radius})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()

        print(f"Point: ({x}, {y})")
        print(f"Radius: {radius}")
        print(f"Labels: {labels}")
        print(f"Counts: {counts}")

    majority_label = labels[np.argmax(counts)]
    majority_percentage = counts[np.argmax(counts)] / np.sum(counts)
    if return_percentage:
        return majority_label, majority_percentage
    return majority_label

def is_far_enough(point, selected_points, min_distance):
    for selected_point in selected_points:
        if distance.euclidean(point, selected_point) < min_distance:
            return False
    return True

def generate_smart_points(mask_generator, img, gt_img, num_labels, radius=10, grid_size=10, min_distance=20):
    masks = mask_generator.generate(img)
    masked_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    # print(f"Number of masks: {len(sorted_masks)}")

    centroids_and_labels = []
    for mask in sorted_masks:
        m = mask['segmentation']
        centroid = calculate_centroid(m)
        if centroid is not None:
            majority_label = get_majority_label_in_circular_window(gt_img, centroid, radius)
            centroids_and_labels.append((centroid, majority_label))
            masked_image[m] = 1  # Mark as filled

    selected_points_and_labels = centroids_and_labels[:num_labels]

    grid_height, grid_width = img.shape[0] // grid_size, img.shape[1] // grid_size

    # Track the number of points in each cell using a priority queue
    cell_point_counts = {(i, j): 0 for i in range(grid_size) for j in range(grid_size)}
    for (x, y), _ in selected_points_and_labels:
        cell_i, cell_j = x // grid_height, y // grid_width
        if (cell_i, cell_j) not in cell_point_counts:
            cell_point_counts[(cell_i, cell_j)] = 0
        cell_point_counts[(cell_i, cell_j)] += 1

    # Initialize the priority queue (min-heap) with the initial counts
    pq = [(count, (i, j)) for (i, j), count in cell_point_counts.items()]
    heapq.heapify(pq)

    continue_ = False

    # Ensure even distribution by filling grid cells
    initial_points_count = len(selected_points_and_labels)
    while len(selected_points_and_labels) < num_labels:
        if not pq:
            print("Heap is empty but the required number of labels has not been reached.")
            break  # Break out of the loop if the heap is empty

        count, (i, j) = heapq.heappop(pq)
        if count == 0 or len(selected_points_and_labels) < num_labels:
            # Find a point in this cell
            cell_indices = np.argwhere(
                (i * grid_height <= np.arange(masked_image.shape[0])[:, None]) & 
                (np.arange(masked_image.shape[0])[:, None] < (i + 1) * grid_height) & 
                (j * grid_width <= np.arange(masked_image.shape[1])) & 
                (np.arange(masked_image.shape[1]) < (j + 1) * grid_width)
            )
            
            if len(cell_indices) > 0:
                max_attempts = 100  # Maximum number of attempts to find a valid point
                attempts = 0
                best_point = None
                best_majority_percentage = 0
                min_distance = 10  # Minimum distance from already selected points
                while attempts < max_attempts:
                    random_point = tuple(cell_indices[random.randint(0, len(cell_indices) - 1)])
                    
                    # Check if the random point is far from already selected points
                    distances = [np.linalg.norm(np.array(random_point) - np.array(p)) for p, _ in selected_points_and_labels]
                    if all(d > min_distance for d in distances):
                        majority_label, majority_percentage = get_majority_label_in_circular_window(gt_img, random_point, radius, return_percentage=True)
                        if majority_percentage > best_majority_percentage:
                            best_point = (random_point, majority_label)
                            best_majority_percentage = majority_percentage
                        if majority_percentage > 0.8:
                            break
                    attempts += 1

                if best_point is None:
                    # If no valid point is found, select a random point
                    random_index = random.randint(0, len(cell_indices) - 1)
                    random_point = tuple(cell_indices[random_index])
                    random_label = gt_img[random_point[0], random_point[1]]  # Assuming gt_img contains the labels
                    best_point = (random_point, random_label)
                    random_point, majority_label = best_point
                else:
                    # print(f"Selected best point {best_point[0]} with majority percentage {best_majority_percentage:.2f} in cell ({i}, {j}) after {max_attempts} attempts.")
                    random_point, majority_label = best_point

                selected_points_and_labels.append((random_point, majority_label))
                cell_point_counts[(i, j)] += 1  # Update the count for the cell
                # Re-add the cell to the heap with the updated count
                heapq.heappush(pq, (cell_point_counts[(i, j)], (i, j)))
                # print(f"Added point {random_point} with label {majority_label} in cell ({i}, {j})")

    # Ensure the number of points does not exceed num_labels
    if len(selected_points_and_labels) > num_labels:
        selected_points_and_labels = selected_points_and_labels[:num_labels]

    # print(f"Total selected points: {len(selected_points_and_labels)}")

    return selected_points_and_labels

def process_image(args):
    mask_generator, gt_filename, img_filename, num_labels, image_type, color_dict = args
    data = []

    # Ensure num_labels is an integer
    num_labels = int(num_labels)

    # Read the ground truth image
    img = cv2.imread(img_filename, cv2.COLOR_BGR2RGB)

    if image_type == 'grayscale':
        gt_img = cv2.imread(gt_filename, cv2.IMREAD_GRAYSCALE)
    else:
        gt_img = cv2.imread(gt_filename, cv2.IMREAD_COLOR)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    
    points_and_labels = generate_smart_points(mask_generator, img, gt_img, num_labels)

    # Get the extension from the img_filename
    _, ext = os.path.splitext(img_filename)

    for (pos_i, pos_j), label in points_and_labels:
        if image_type == 'grayscale':
            color = label
            image_name = os.path.basename(img_filename)
            image_name = os.path.splitext(image_name)[0] + ext
            data.append([image_name, pos_i, pos_j, color])
        else:
            print("TODO: Implement color dictionary for color images")
            color = tuple(gt_img[pos_i, pos_j])
            if color_dict and color not in color_dict:
                print(f"Color not found in dictionary: {color}")
                continue
            image_name = os.path.basename(img_filename)
            image_name = os.path.splitext(image_name)[0] + ext
            label = color_dict.get(color, label)
            # Convert label to int if it is numeric, otherwise keep it as string
            try:
                label = int(label)
            except ValueError:
                label = str(label)
            data.append([image_name, pos_i, pos_j, label])

    return data

def process_images(images_pth, ground_truth_pth, output_file, num_labels=300, image_type='grayscale', color_dict_path=None):
    data = []

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize the model and mask generator inside the process
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    sam2_model.to(device)
    mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model,
                                               points_per_side=64,
                                               points_per_patch=128,
                                               pred_iou_threshold=0.7,
                                               stability_score_thresh=0.92,
                                               stability_score_offset=0.7,
                                               crop_n_layers=1,
                                               box_nms_thresh=0.7,
                                               )

    color_dict = None
    if color_dict_path:
        color_dict = load_color_dict(color_dict_path)

    # Get the list of ground truth images
    if os.path.isfile(ground_truth_pth):
        gt_image_files = [ground_truth_pth]
    else:
        gt_image_files = glob.glob(ground_truth_pth + '/*.*')

    # Get the list of images from images_pth
    if os.path.isfile(images_pth):
        image_files = [images_pth]
    else:
        image_files = glob.glob(images_pth + '/*.*')

    # Create a mapping of GT images to corresponding images300
    gt_image_map = {os.path.splitext(os.path.basename(gt))[0]: gt for gt in gt_image_files}

    # Create a list of tuples (gt_image, corresponding_image)
    image_pairs = []
    for image_file in image_files:
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        if image_name in gt_image_map:
            gt_image = gt_image_map[image_name]
            image_pairs.append((gt_image, image_file))

    # Ensure num_labels is an integer
    num_labels = int(num_labels)

    for gt, img in tqdm(image_pairs, desc="Processing images"):
        result = process_image((mask_generator, gt, img, num_labels, image_type, color_dict))
        data.extend(result)

    # Modify the output filename to include the number of labels and add .csv extension
    modified_output_file = f"{output_file}_{num_labels}.csv"

    output_df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label'])
    output_df.to_csv(modified_output_file, index=False)

if __name__ == "__main__":
    set_start_method('spawn')
    args = parse_arguments()
    process_images(args.images_pth, args.ground_truth_pth, args.output_file, args.num_labels, args.image_type, args.color_dict)