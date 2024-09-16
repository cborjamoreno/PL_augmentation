import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Paths
labels_path = '../Datasets/SUIM/test/labels'
output_path = '../Datasets/SUIM/test/extra_colors'

# Create the output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Define the allowed colors
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

# Convert allowed colors to a set for fast lookup
allowed_colors_set = set(allowed_colors.keys())

# Function to get all files in a directory with a specific extension
def get_files_with_extension(directory, extension):
    return [os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files if file.endswith(extension)]

# Get all label files
label_files = get_files_with_extension(labels_path, '.bmp')

# Flag to track if any images with extra colors are found
found_extra_colors = False

# Process each label image
for label_file in tqdm(label_files, desc="Checking for extra colors"):
    # Read the label image
    label = cv2.imread(label_file)

    # Get the unique colors in the label image
    unique_colors = np.unique(label.reshape(-1, label.shape[2]), axis=0)

    # Check if there are any colors not in the allowed set
    if any(tuple(color) not in allowed_colors_set for color in unique_colors):
        found_extra_colors = True
        # Create subfolders in the output path
        relative_path = os.path.relpath(os.path.dirname(label_file), labels_path)
        output_label_dir = os.path.join(output_path, relative_path)
        os.makedirs(output_label_dir, exist_ok=True)

        # Copy the label to the output folder
        shutil.copy(label_file, os.path.join(output_label_dir, os.path.basename(label_file)))

# Print message if no images with extra colors were found
if not found_extra_colors:
    print("All images are correctly colored. No images with extra colors found.")

print("Done checking for extra colors.")