import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Example of broadcasting (multiplying vectors of different sizes)
boxes_center = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)

box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
box_class_prob = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
box_scores = box_confidence * box_class_prob

box_classes = tf.argmax(box_scores, axis=-1)
box_class_scores = tf.reduce_max(box_scores, axis=-1)
print(f"Box Scores: {box_scores} / Box Scores Shape {box_scores.shape}")
print("----------------------------------")
print(f"Box Classes: {box_classes} / Box Classes Shape {box_classes.shape}")
print("----------------------------------")
print(f"Box Class Scores: {box_class_scores} / Box Class Scores Shape {box_class_scores.shape}")

threshold = 0.6
mask = box_class_scores >= threshold

classes = tf.boolean_mask(tensor=box_classes, mask=mask)
boxes_center = tf.boolean_mask(tensor=boxes_center, mask=mask)
scores = tf.boolean_mask(tensor=box_scores, mask=mask)

print(f"classes shape: {classes.shape}\n {classes}\n, boxes center shape: {boxes_center.shape}\n {boxes_center}\n, scores shape: {scores.shape}\n {scores}")


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):
    """
    Filters YOLO boxes by thresholding on object and class confidence.
    """
    # TODO: Compute the box scores
    # Score = the probability that there is an object 𝑝𝑐 times the probability that the object is a certain class 𝑐𝑖. 
    box_scores = box_confidence * box_class_probs

    # TODO: Find the index of the class with the max box score. "tf.math.argmax"
    box_classes = tf.argmax(box_scores, axis=-1)
    
    # TODO: Find the corresponding box score. "tf.math.reduce_max"
    box_class_scores = tf.reduce_max(box_scores, axis=-1)

    # TODO: Create a mask by using a threshold.
    mask = box_class_scores >= threshold

    # TODO: Use TF to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes you don't want. "tf.boolean_mask"
    classes = tf.boolean_mask(tensor=box_classes, mask=mask)
    boxes = tf.boolean_mask(tensor=boxes, mask=mask)
    scores = tf.boolean_mask(tensor=box_scores, mask=mask)

    return classes, boxes, scores

# Intersection over Union
box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)

(box1_x1, box1_y1, box1_x2, box1_y2) = box1
(box2_x1, box2_y1, box2_x2, box2_y2) = box2

xi1 = max(box1_x1, box2_x1)
yi1 = max(box1_y1, box2_y1)
xi2 = min(box1_x2, box2_x2)
yi2 = min(box1_y2, box2_y2)

inter_width = xi2 - xi1
inter_height =  yi2 - yi1

inter_area = max(inter_width, 0) * max(inter_height, 0)
# Plot the rectangles
fig, ax = plt.subplots()
rect1 = patches.Rectangle((box1_x1, box1_y1), box1_x2 - box1_x1, box1_y2 - box1_y1, linewidth=1, edgecolor='r', facecolor='none')
rect2 = patches.Rectangle((box2_x1, box2_y1), box2_x2 - box2_x1, box2_y2 - box2_y1, linewidth=1, edgecolor='b', facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)

# Plot the intersection rectangle
if inter_area > 0:
    rect_inter = patches.Rectangle((xi1, yi1), inter_width, inter_height, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect_inter)

# Set the aspect of the plot to be equal
ax.set_aspect('equal', 'box')

# Set limits
plt.xlim(0, 10)
plt.ylim(0, 10)

# Show plot
plt.show()

print("Intersection Area:", inter_area) 


def iou(box1, box2):
    """
    Implement the intersection over union (IOU) between box 1 and box 2
    """
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # TODO: Calculate the coordinates of the intersection of box1 and box 2, anc calculate its area.
    xi1 = max(box1_x1, box2_x1)
    xi2 = max(box1_x2, box2_x2)
    yi1 = max(box1_y1, box2_y1)
    yi2 = max(box1_y2, box2_y2)

    intersection_w = xi2 - xi1
    intersection_h = yi2 - yi1

    intersection_area = max(intersection_w, 0) * max(intersection_h, 0)

    # TODO: Calculate the Union area by using Formula: Union(A, B) = A + B - Inter(A, B)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - intersection_area

    # TODO: Compute the IoU (intersection area/union area)
    iou = intersection_area / union_area

    return iou

scores = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
boxes = tf.random.normal([54, 4], mean=1, stddev=4, seed = 1)
classes = tf.random.normal([54,], mean=1, stddev=4, seed = 1)

print(f"Scores: {scores}\n Boxes: {boxes}\n Classes: {classes}")

max_boxes = 10
iou_threshold = 0.5

# TODO: Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
selected_indices = tf.image.non_max_suppression(
    boxes=boxes,
    scores=scores,
    max_output_size=max_boxes,
    iou_threshold=iou_threshold
    )

# TODO: Use tf.gather() to select only nms_indices from scores, boxes and classes
scores = tf.gather(scores, indices=selected_indices)
boxes = tf.gather(boxes, indices=selected_indices)
classes = tf.gather(classes, indices=selected_indices)

print(f"After NMS Scores: {scores}\n After NMS Boxes: {boxes}\n After NMS Classes: {classes}")

# TODO: Select the box that has the highest score.

# TODO: Compute the overlap of this box with all other boxes, and remove boxes that overlap significantly (iou >= iou_threshold).


# TODO: Go back to step 1 and iterate until there are no more boxes with a lower score than the currently selected box.


