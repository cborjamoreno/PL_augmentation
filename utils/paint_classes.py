from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def create_color_mapping(image_path, colormap_name='viridis'):
    # Open the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    # Get unique grayscale values
    unique_colors = np.unique(image_array)

    # Create a colormap
    colormap = plt.get_cmap(colormap_name, len(unique_colors) - 1)
    colormap_colors = colormap(np.arange(len(unique_colors) - 1))

    # Create a new colormap with black as the first color
    new_colormap_colors = np.vstack(([0, 0, 0, 1], colormap_colors))
    new_colormap = ListedColormap(new_colormap_colors)

    # Map grayscale values to RGB colors
    color_mapping = {color: new_colormap(i) for i, color in enumerate(unique_colors)}

    return color_mapping

def apply_color_mapping(image_path, color_mapping):
    # Open the image and convert to grayscale
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)

    # Create a new RGB image
    rgb_image_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)
    for grayscale_value, rgb_color in color_mapping.items():
        rgb_image_array[image_array == grayscale_value] = (np.array(rgb_color[:3]) * 255).astype(np.uint8)

    # Convert the RGB array to an image
    rgb_image = Image.fromarray(rgb_image_array)

    return rgb_image

# Example usage
image_path_1 = 'MosaicsUCSD/MosaicsUCSD_sam_100/labels/FR3_512_1024_1536_2048.png'
image_path_2 = 'MosaicsUCSD/MosaicsUCSD_superpixel_100/labels/FR3_512_1024_1536_2048.png'
image_path_3 = '../MosaicsUCSD/train/labels/FR3_512_1024_1536_2048.png'

# Create color mapping from the first image
color_mapping = create_color_mapping(image_path_1)

# Apply the same color mapping to the second and third images
rgb_image_1 = apply_color_mapping(image_path_1, color_mapping)
rgb_image_2 = apply_color_mapping(image_path_2, color_mapping)
rgb_image_3 = apply_color_mapping(image_path_3, color_mapping)

# Display the images
# rgb_image_1.show()
# rgb_image_2.show()
# rgb_image_3.show()

# Save the images
rgb_image_1.save('color_sam.png')
rgb_image_2.save('color_superpixel.png')
rgb_image_3.save('color_gt.png')
