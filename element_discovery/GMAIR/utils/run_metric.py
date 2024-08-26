import numpy as np
from metric_icdar import *
import torch

# Sample input sizes
batch_size = 3
num_predicted_boxes = 20
num_ground_truth_boxes = 15
num_classes = 20

# Randomly generate predicted bounding box coordinates
z_where_bbox = np.random.rand(batch_size, num_predicted_boxes, 4)
z_where_class = np.random.rand(batch_size, num_predicted_boxes, num_classes)
score = np.random.rand(batch_size, num_predicted_boxes, 1)

# Randomly generate ground truth bounding box coordinates
ground_truth_bbox = np.random.rand(batch_size, num_ground_truth_boxes, 5)  # Assuming the last column is the class index

# Randomly generate the count of true positive bounding boxes for each class
truth_bbox_digit_count = np.random.randint(0, 10, (batch_size, num_classes))

# Call the mAP function with the numpy inputs
mAP_value = mAP((z_where_bbox, z_where_class), score, ground_truth_bbox, truth_bbox_digit_count)
print("mAP:", mAP_value)
