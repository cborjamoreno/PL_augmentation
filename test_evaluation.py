import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

BACKGROUND_CLASS = 0
NUM_CLASSES = 4  # Including background

def create_artificial_image(width, height):
    """
    Create an artificial image with 4 quadrants, each representing a different class.
    """
    image = np.zeros((height, width), dtype=np.uint8)
    half_height = height // 2
    half_width = width // 2

    image[:half_height, :half_width] = 1  # Top-left quadrant
    image[:half_height, half_width:] = 2  # Top-right quadrant
    image[half_height:, :half_width] = 3  # Bottom-left quadrant
    image[half_height:, half_width:] = BACKGROUND_CLASS  # Bottom-right quadrant (background)

    return image

def modify_image_deterministic(image, quadrant):
    """
    Modify one quadrant of one of the quadrants, so 1/16 of the image in a square is different.
    """
    modified_image = image.copy()
    height, width = image.shape
    quarter_height = height // 4
    quarter_width = width // 4

    if quadrant == 1:
        start_i, start_j = 0, 0
    elif quadrant == 2:
        start_i, start_j = 0, width // 2
    elif quadrant == 3:
        start_i, start_j = height // 2, 0
    elif quadrant == 4:
        start_i, start_j = height // 2, width // 2

    # Modify the top-left quadrant of the specified quadrant (1/16 of the image)
    for i in range(start_i, start_i + quarter_height):
        for j in range(start_j, start_j + quarter_width):
            if image[i, j] != BACKGROUND_CLASS:  # Ensure not modifying background
                current_class = image[i, j]
                new_class = (current_class + 1) % NUM_CLASSES
                if new_class == BACKGROUND_CLASS:
                    new_class = (new_class + 1) % NUM_CLASSES
                modified_image[i, j] = new_class
    
    return modified_image

def save_image(image, filename):
    """
    Save the image to a file.
    """
    cv2.imwrite(filename, image)

def display_images_in_pairs(image, ground_truth, index):
    """
    Display the generated image and its ground truth side by side using matplotlib.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap='tab20')
    axes[0].set_title(f"Generated Image {index}")
    axes[0].axis('off')
    axes[1].imshow(ground_truth, cmap='tab20')
    axes[1].set_title(f"Ground Truth Image {index}")
    axes[1].axis('off')
    plt.show()

def calculate_metrics(img, gt):
    # Calculate PA (Pixel Accuracy)
    correct_pixels = np.sum(img == gt)
    total_pixels = gt.size
    pa = correct_pixels / total_pixels

    # Exclude background class for MPA and IoU calculation
    valid_pixels_mask = gt != BACKGROUND_CLASS

    # Apply mask to exclude background
    img_valid = img[valid_pixels_mask]
    gt_valid = gt[valid_pixels_mask]

    # Calculate per-class IoU and MPA
    unique_classes = np.unique(gt_valid)

    class_iou = {}
    class_mpa = []

    for cls in unique_classes:
        gt_cls = gt_valid == cls
        img_cls = img_valid == cls

        mpa_cls = np.sum(gt_cls & img_cls) / np.sum(gt_cls) if np.sum(gt_cls) != 0 else 0
        class_mpa.append(mpa_cls)

        intersection = np.sum(gt_cls & img_cls)
        union = np.sum(gt_cls | img_cls)

        iou = intersection / union if union != 0 else 0
        class_iou[cls] = iou

    mean_mpa = np.mean(class_mpa) if class_mpa else 0
    mean_iou = np.mean(list(class_iou.values())) if class_iou else 0
    return pa, mean_mpa, mean_iou, class_iou, class_mpa

def main():
    output_dir = "artificial_images"
    output_dir_gt = "artificial_gt"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_gt, exist_ok=True)

    width, height = 100, 100  # Image dimensions

    all_pa = []
    all_mpa = []
    all_miou = []

    for i in range(4):  # Generate 4 images
        gt = create_artificial_image(width, height)
        img = modify_image_deterministic(gt, i + 1)

        img_filename = os.path.join(output_dir, f"image_{i}.png")
        gt_filename = os.path.join(output_dir_gt, f"image_{i}.png")

        save_image(img, img_filename)
        save_image(gt, gt_filename)

        print(f"Saved {img_filename} and {gt_filename}")

        # Calculate and print metrics
        pa, mean_mpa, mean_iou, class_iou, class_mpa = calculate_metrics(img, gt)
        all_pa.append(pa)
        all_mpa.append(mean_mpa)
        all_miou.append(mean_iou)
        print(f"Image {i} - PA: {pa}, MPA: {mean_mpa}, MIoU: {mean_iou}")
        print(f"Class PA: {class_mpa}")
        print(f"Class IoU: {class_iou}")

        # Display the images in pairs
        display_images_in_pairs(img, gt, i)

    # Calculate final measurements
    final_pa = np.mean(all_pa)
    final_mpa = np.mean(all_mpa)
    final_miou = np.mean(all_miou)

    print(f"Final PA: {final_pa}")
    print(f"Final MPA: {final_mpa}")
    print(f"Final MIoU: {final_miou}")

if __name__ == "__main__":
    main()