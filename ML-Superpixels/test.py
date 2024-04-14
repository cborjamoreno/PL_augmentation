from PIL import Image

def get_unique_colors(image_path):
    image = Image.open(image_path)
    colors = set()

    # Iterate over each pixel in the image
    for pixel in image.getdata():
        colors.add(pixel)

    return colors

# Replace 'image_path' with the path to your image file
image_path = 'camvid/sparse_GT/train/0001TP_006690.png'
unique_colors = get_unique_colors(image_path)

print(unique_colors)