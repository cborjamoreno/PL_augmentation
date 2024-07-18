import numpy as np
import cv2

def create_test_images():
    # Create blank images
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gt = np.zeros((100, 100, 3), dtype=np.uint8)

    # Red class
    img[10:50, 10:50] = [255, 0, 0]
    gt[10:50, 10:50] = [255, 0, 0]
    
    # Green class
    img[60:90, 60:90] = [0, 255, 0]
    gt[60:90, 60:90] = [0, 255, 0]
    
    # Errors
    img[15:20, 15:20] = [0, 0, 255]  # False positive
    gt[80:85, 80:85] = [0, 0, 255]   # False negative
    
    return img, gt

img, gt = create_test_images()

# Save images for visual verification
cv2.imwrite("test_img.png", img)
cv2.imwrite("test_gt.png", gt)

def calculate_metrics(img, gt):
    assert img.shape == gt.shape, "Image and GT dimensions do not match"
    
    # Total number of pixels
    total_pixels = img.shape[0] * img.shape[1]
    
    # Pixel accuracy (PA)
    correct_pixels = np.sum(np.all(gt == img, axis=-1))
    pixel_accuracy = correct_pixels / total_pixels
    
    # Unique classes
    unique_classes = np.unique(gt.reshape(-1, gt.shape[-1]), axis=0)
    class_iou = []
    class_pa = []

    for cls in unique_classes:
        if np.all(cls == [0, 0, 0]):  # Skip background
            continue

        gt_cls = np.all(gt == cls, axis=-1)
        img_cls = np.all(img == cls, axis=-1)

        intersection = np.sum(gt_cls & img_cls)
        union = np.sum(gt_cls | img_cls)

        iou = intersection / union if union != 0 else 0
        pa_cls = np.sum(gt_cls & img_cls) / np.sum(gt_cls) if np.sum(gt_cls) != 0 else 0

        class_iou.append(iou)
        class_pa.append(pa_cls)

    mean_iou = np.mean(class_iou) if class_iou else 0
    mean_pa = np.mean(class_pa) if class_pa else 0

    return pixel_accuracy, mean_pa, mean_iou

# Calculate metrics
pa, mpa, miou = calculate_metrics(img, gt)

print(f"Test Image - Mean Pixel Accuracy (PA): {pa}")
print(f"Test Image - Mean Pixel Accuracy (MPA): {mpa}")
print(f"Test Image - Mean Intersection over Union (MIoU): {miou}")

expected_pa = 0.25
expected_mpa = 0.6667
expected_miou = 0.6667

print(f"Expected - Mean Pixel Accuracy (PA): {expected_pa}")
print(f"Expected - Mean Pixel Accuracy (MPA): {expected_mpa}")
print(f"Expected - Mean Intersection over Union (MIoU): {expected_miou}")

# Ensure the calculated metrics are as expected
assert np.isclose(pa, expected_pa, atol=1e-4), f"PA calculation is incorrect: {pa} vs {expected_pa}"
assert np.isclose(mpa, expected_mpa, atol=1e-4), f"MPA calculation is incorrect: {mpa} vs {expected_mpa}"
assert np.isclose(miou, expected_miou, atol=1e-4), f"MIoU calculation is incorrect: {miou} vs {expected_miou}"

print("All metrics calculated correctly!")