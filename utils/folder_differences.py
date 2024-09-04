import os

def extract_base_name(filename):
    # Split the filename by underscore and take the first three parts if available
    parts = filename.split('_')[:3]
    base_name = '_'.join(parts)
    return base_name

def find_missing_images(folder1, folder2):
    images1 = {extract_base_name(f) for f in os.listdir(folder1)}
    images2 = {extract_base_name(f) for f in os.listdir(folder2)}
    
    missing_images = images1 - images2
    
    return missing_images

folder1 = '../SegFormer_segmentation/SUIM_superpixel/train/masks'
folder2 = '../SegFormer_segmentation/SUIM/train/images'

missing_images = find_missing_images(folder1, folder2)

print("Missing images:")
for image in missing_images:
    print(image)

    # remove the image from the folder
    os.remove(os.path.join(folder1, image + '_.bmp'))