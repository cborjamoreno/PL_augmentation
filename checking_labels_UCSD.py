import glob
import cv2
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--images_pth", help="path of images folder")
    parser.add_argument("-i", "--image_path", help="path of a single image")
    return parser.parse_args()

def get_unique_grayscale_values_from_folder(image_pth):
    image_files = glob.glob(image_pth + '/*.*')
    unique_values = set()

    for filename in image_files:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            unique_values.update(np.unique(img))

    return unique_values

def get_unique_grayscale_values_from_image(image_path):
    unique_values = set()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        unique_values.update(np.unique(img))
    else:
        print(f"Failed to load image: {image_path}")
    return unique_values

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.images_pth:
        unique_grayscale_values = get_unique_grayscale_values_from_folder(args.images_pth)
        print(f"Unique grayscale values in the folder: {sorted(unique_grayscale_values)}")
    elif args.image_path:
        unique_grayscale_values = get_unique_grayscale_values_from_image(args.image_path)
        print(f"Unique grayscale values in the image: {sorted(unique_grayscale_values)}")
    else:
        print("Please provide either a folder path with -p or an image path with -i.")