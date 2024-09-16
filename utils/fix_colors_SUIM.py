import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from scipy.spatial import KDTree

# Paths
labels_path = '../Datasets/SUIM/test/extra_colors'
output_path = '../Datasets/SUIM/test/fixed_labels'

# Create the output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Define the allowed colors (change 1 to 255)
allowed_colors = {
    (0, 0, 0): 'Background (waterbody)',  # Black
    (0, 0, 255): 'Human divers',  # Blue
    (0, 255, 0): 'Aquatic plants and sea-grass',  # Green
    (0, 255, 255): 'Wrecks and ruins',  # Sky
    (255, 0, 0): 'Robots (AUVs/ROVs/instruments)',  # Red
    (255, 0, 255): 'Reefs and invertebrates',  # Pink
    (255, 255, 0): 'Fish and vertebrates',  # Yellow
    (255, 255, 255): 'Sea-floor and rocks'  # White
}

# Convert allowed colors to a list and create a KDTree for fast nearest neighbor search
allowed_colors_list = np.array(list(allowed_colors.keys()))
kd_tree = KDTree(allowed_colors_list)

# Function to get all files in a directory with a specific extension
def get_files_with_extension(directory, extension):
    return [os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files if file.endswith(extension)]

# Get all label files
label_files = get_files_with_extension(labels_path, '.bmp')

# Function to force colors to be within the allowed colors using a sliding window approach
def force_colors_with_window(image, radius=5, threshold=50):
    # Create an output image initialized to the original image
    output_image = image.copy()

    # Create a copy of the original image for window calculations
    original_image = image.copy()

    # Get the dimensions of the image
    height, width, _ = image.shape

    # First pass: force pixels close to allowed colors to the nearest allowed color
    for y in range(height):
        for x in range(width):
            current_color = original_image[y, x]
            distance, index = kd_tree.query(current_color)
            if distance < threshold:
                output_image[y, x] = allowed_colors_list[index]

    # Second pass: apply the window-based approach for remaining pixels
    for y in range(height):
        for x in range(width):
            # Get the current pixel color
            current_color = tuple(output_image[y, x])

            # If the current color is in the allowed set, skip the calculation
            if current_color in allowed_colors:
                continue

            # Get the window around the current pixel
            y_min = max(0, y - radius)
            y_max = min(height, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(width, x + radius + 1)
            window = original_image[y_min:y_max, x_min:x_max]

            # Get the unique colors in the window
            unique_colors = np.unique(window.reshape(-1, window.shape[2]), axis=0)

            # Find the closest allowed color for each pixel in the window
            closest_colors = []
            for color in unique_colors:
                if tuple(color) in allowed_colors:
                    closest_colors.append(tuple(color))
                else:
                    _, index = kd_tree.query(color)
                    closest_colors.append(tuple(allowed_colors_list[index]))

            # Determine the majority color in the window
            majority_color = max(set(closest_colors), key=closest_colors.count)

            # Assign the majority color to the center pixel of the window
            output_image[y, x] = majority_color

    return output_image

# Process each label image
for label_file in tqdm(label_files, desc="Checking and fixing label colors"):
    # Read the label image
    label = cv2.imread(label_file)

    # Force colors to be within the allowed colors using the sliding window approach
    fixed_label = force_colors_with_window(label)

    # Create subfolders in the output path
    relative_path = os.path.relpath(os.path.dirname(label_file), labels_path)
    output_label_dir = os.path.join(output_path, relative_path)
    os.makedirs(output_label_dir, exist_ok=True)

    # Save the fixed label
    output_label_path = os.path.join(output_label_dir, os.path.basename(label_file))
    cv2.imwrite(output_label_path, fixed_label)

print("Done checking and fixing label colors.")