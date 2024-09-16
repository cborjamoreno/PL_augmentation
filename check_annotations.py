import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Paths
images_folder = "../Datasets/SUIM/hard/masks"
csv_file = "SUIM_hard_100_annotations.csv"
output_folder = "SUIM/SUIM_mixed_100_pre_enhanced/gt_points"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Function to plot points on an image
def plot_points_on_image(image_path, points, output_path):
    # Read the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

    # Plot each point
    for _, row in points.iterrows():
        # Draw a star marker with yellow edges and black fill
        cv2.drawMarker(img, (int(row['Column']), int(row['Row'])), (0, 0, 0), markerType=cv2.MARKER_STAR, 
                       markerSize=5, thickness=2, line_type=cv2.LINE_AA)
        cv2.drawMarker(img, (int(row['Column']), int(row['Row'])), (255, 255, 0), markerType=cv2.MARKER_STAR, 
                       markerSize=5, thickness=1, line_type=cv2.LINE_AA)

    # Convert back to BGR for saving with OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)

# Process each unique image in the CSV
for image_name in df['Name'].unique():
    # Change the file extension from .jpg to .bmp
    image_name_bmp = image_name.replace('.jpg', '.bmp')
    image_path = os.path.join(images_folder, image_name_bmp)

    # Filter points for the current image
    image_points = df[df['Name'] == image_name]

    # Define the output path
    output_path = os.path.join(output_folder, image_name_bmp)

    # Plot points on the image and save it
    if os.path.exists(image_path):
        plot_points_on_image(image_path, image_points, output_path)
    else:
        print(f"Image {image_path} not found.")