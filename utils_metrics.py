import numpy as np

def calculate_global_accuracy(confusion_matrix, ignore_class=None):
    if ignore_class is not None:
        confusion_matrix = np.delete(confusion_matrix, ignore_class, axis=0)
        confusion_matrix = np.delete(confusion_matrix, ignore_class, axis=1)
    correct = np.diag(confusion_matrix).sum()
    total = confusion_matrix.sum()
    return correct / total if total > 0 else 0

def calculate_pixel_accuracy(confusion_matrix, ignore_class=None):
    if ignore_class is not None:
        confusion_matrix = np.delete(confusion_matrix, ignore_class, axis=0)
        confusion_matrix = np.delete(confusion_matrix, ignore_class, axis=1)
    class_accuracies = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    class_accuracies = np.where(np.isnan(class_accuracies), 0, class_accuracies)  # Replace NaNs with 0
    mean_pixel_accuracy = np.nanmean(class_accuracies)
    return class_accuracies, mean_pixel_accuracy

def calculate_iou(confusion_matrix, ignore_class=None):
    if ignore_class is not None:
        confusion_matrix = np.delete(confusion_matrix, ignore_class, axis=0)
        confusion_matrix = np.delete(confusion_matrix, ignore_class, axis=1)
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection
    iou_scores = intersection / union
    iou_scores = np.where(np.isnan(iou_scores), 0, iou_scores)  # Replace NaNs with 0
    mean_iou = np.nanmean(iou_scores)
    return iou_scores, mean_iou