import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open the image file
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                # Resize the image
                img_resized = img.resize(target_size, Image.ANTIALIAS)
                
                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)
                print(f'Resized and saved {filename} to {output_folder}')

# Example usage
input_folder = '../Datasets/'
output_folder = 'path/to/output/folder'
target_size = (800, 600)  # Width, Height

resize_images(input_folder, output_folder, target_size)