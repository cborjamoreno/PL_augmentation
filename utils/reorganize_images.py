import os
import re
import shutil

# Define the main folder and the output folder
main_folder = '/home/cbm/BOSTON/CoralNet_expansion/Sebens_MA_LTM/out_superpixels/eval_images/'
output_folder = '/home/cbm/BOSTON/CoralNet_expansion/Sebens_MA_LTM/out_superpixels/reorganized/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize an empty set to store the unique class names
class_names = set()

# Iterate over each subfolder in the main folder
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    
    # Skip if not a directory
    if not os.path.isdir(subfolder_path):
        continue

    # For each subfolder, iterate over each file
    for filename in os.listdir(subfolder_path):
        # Extract the class name from the file name using a regular expression
        match = re.search(r'_(\w+)_expanded', filename)
        if match:
            class_name = match.group(1)
            class_names.add(class_name)

# Print the unique class names
for class_name in class_names:
    print(class_name)

# Create a folder for each class in the output folder
for class_name in class_names:
    os.makedirs(os.path.join(output_folder, class_name), exist_ok=True)

# Iterate over each subfolder in the main folder
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    
    # Skip if not a directory
    if not os.path.isdir(subfolder_path):
        continue

    # For each subfolder, iterate over each file
    for filename in os.listdir(subfolder_path):
        # Check if the file name contains the class name
        for class_name in class_names:
            if f'_{class_name}_' in filename:
                # Copy the file to the corresponding class folder in the output folder
                src = os.path.join(subfolder_path, filename)
                dst = os.path.join(output_folder, class_name, filename)
                shutil.copy(src, dst)