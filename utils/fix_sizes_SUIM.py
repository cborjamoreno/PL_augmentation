import os
import cv2
from tqdm import tqdm

# Paths
images_path = '../Datasets/SUIM/hard/images'
labels_path = '../Datasets/SUIM/hard/labels'
output_path = '../Datasets/SUIM/hard/cropped_labels'

# Create the output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

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

# Crop labels and save them
for image_file in tqdm(image_files, desc="Cropping label images"):
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    label_file = label_files_dict.get(image_name)

    if label_file:
        # Read the image and label
        image = cv2.imread(image_file)
        label = cv2.imread(label_file)

        # Check if the label is 55 pixels higher than the image
        if label.shape[0] == image.shape[0] + 55 and label.shape[1] == image.shape[1]:
            # Crop 55 pixels from the top of the label
            cropped_label = label[:-55:, :]

            # Create subfolders in the output path
            relative_path = os.path.relpath(os.path.dirname(label_file), labels_path)
            output_label_dir = os.path.join(output_path, relative_path)
            os.makedirs(output_label_dir, exist_ok=True)

            # Save the cropped label
            output_label_path = os.path.join(output_label_dir, os.path.basename(label_file))
            cv2.imwrite(output_label_path, cropped_label)

print("Done cropping and saving label images.")