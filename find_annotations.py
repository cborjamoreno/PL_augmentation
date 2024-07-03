import pandas as pd
import os
import shutil
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--annotations', help='Path to the CSV file with annotations')
parser.add_argument('--dataset_path', help='Path to the dataset with the images')
args = parser.parse_args()

# Path to the images
img_folder = os.path.join(args.dataset_path, 'images')

# Get a list of all image file names in the image folder
image_files = os.listdir(img_folder)

# Ensure all images have the correct '.JPG.jpg' extension
for i in range(len(image_files)):
    old_name = image_files[i]
    new_name = old_name.rsplit('.', 1)[0]  # Remove current extension(s)
    if not new_name.endswith('.JPG'):
        new_name += '.JPG'  # Add '.JPG' if it's not there
    new_name += '.jpg'  # Add '.jpg'
    if old_name != new_name:
        os.rename(os.path.join(img_folder, old_name), os.path.join(img_folder, new_name))

# Read the CSV file
df = pd.read_csv(args.annotations)

# Get the updated list of all image file names in the image folder
image_files = os.listdir(img_folder)

# Remove '.jpg' from the image file names for searching in the annotations CSV
image_files = [file.rsplit('.', 1)[0] for file in image_files]

# Filter the DataFrame
df_filtered = df[df['Name'].isin(image_files)]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('annotations_in_images.csv', index=False)

# Find images that are not in the CSV file
not_in_csv = set(image_files) - set(df['Name'])

# Add '.jpg' to the image file names
not_in_csv = [file + '.jpg' for file in not_in_csv]

# If there are images not in the CSV file, create a new directory for these images and copy them
if not_in_csv:
    print(f'{len(not_in_csv)} images are not in the CSV file')
    not_in_csv_dir = os.path.join(args.dataset_path, 'not_in_csv')
    os.makedirs(not_in_csv_dir, exist_ok=True)
    for image in not_in_csv:
        shutil.copy(os.path.join(img_folder, image), not_in_csv_dir)
else:
    print('All images are in the CSV file')