import os
import shutil
import cv2
from tqdm import tqdm

# Paths
images_path = '../Datasets/SUIM/train/images'
labels_path = '../Datasets/SUIM/train/labels'
output_path = '../Datasets/SUIM/train/output'

# Function to get all files in a directory with a specific extension
def get_files_with_extension(directory, extension):
    return [os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files if file.endswith(extension)]

# Get all image and label files
image_files = get_files_with_extension(images_path, '.jpg')
label_files = get_files_with_extension(labels_path, '.bmp')

# Create a dictionary for quick lookup of label files by name
label_files_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in label_files}

# Variable to store the size difference
size_differences = set()
mismatched_files = []

# Check sizes and copy mismatched files
for image_file in tqdm(image_files, desc="Checking image sizes"):
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    label_file = label_files_dict.get(image_name)

    if label_file:
        # Read the image and label
        image = cv2.imread(image_file)
        label = cv2.imread(label_file)

        # Check if sizes differ
        if image.shape[:2] != label.shape[:2]:

            os.makedirs(output_path, exist_ok=True)

            # Calculate the size difference
            size_diff = (image.shape[0] - label.shape[0], image.shape[1] - label.shape[1])
            size_differences.add(size_diff)
            mismatched_files.append((image_file, label_file))

            # Create subfolders in the output path
            relative_path = os.path.relpath(os.path.dirname(image_file), images_path)
            output_image_dir = os.path.join(output_path, 'images', relative_path)
            output_label_dir = os.path.join(output_path, 'labels', relative_path)
            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)

            # Copy the files to the output folder
            shutil.copy(image_file, os.path.join(output_image_dir, os.path.basename(image_file)))
            shutil.copy(label_file, os.path.join(output_label_dir, os.path.basename(label_file)))

# Check if there are any mismatched files
if mismatched_files:
    # Check if all size differences are the same
    if len(size_differences) == 1:
        print(f"All mismatched images have the same size difference: {size_differences.pop()}")
    else:
        print(f"Mismatched images have different size differences: {size_differences}")
else:
    print("No images with different sizes found.")

print("Done checking and copying mismatched files.")