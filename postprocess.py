import numpy as np

def convert_xywh_to_xyxy(bbox_array: np.array) -> np.array:
    """
    Convert bounding boxes from (x, y, width, height) format to (x1, y1, x2, y2) format.

    Args:
        bbox_array (np.array): Array of bounding boxes in (x, y, width, height) format.

    Returns:
        np.array: Array of bounding boxes in (x1, y1, x2, y2) format.
    """
    converted_boxes = np.zeros_like(bbox_array)
    converted_boxes[:, 0] = bbox_array[:, 0] - bbox_array[:, 2] / 2  # x1 (top-left x)
    converted_boxes[:, 1] = bbox_array[:, 1] - bbox_array[:, 3] / 2  # y1 (top-left y)
    converted_boxes[:, 2] = bbox_array[:, 0] + bbox_array[:, 2] / 2  # x2 (bottom-right x)
    converted_boxes[:, 3] = bbox_array[:, 1] + bbox_array[:, 3] / 2  # y2 (bottom-right y)
    return converted_boxes

def calculate_iou(box1: np.array, box2: np.array) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (np.array): First bounding box in (x1, y1, x2, y2) format.
        box2 (np.array): Second bounding box in (x1, y1, x2, y2) format.

    Returns:
        float: IoU value between the two bounding boxes.
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the coordinates of the intersection rectangle
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

    # Calculate the area of both input rectangles
    area1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    # Calculate IoU
    iou = intersection_area / float(area1 + area2 - intersection_area)
    return iou

def nms(bboxes: np.array, scores: np.array, iou_threshold: float) -> np.array:
    """
    Apply Non-Maximum Suppression (NMS) to a set of bounding boxes.

    Args:
        bboxes (np.array): Bounding boxes in (x1, y1, x2, y2) format.
        scores (np.array): Confidence scores for each bounding box.
        iou_threshold (float): IoU threshold for NMS.

    Returns:
        np.array: Indices of selected bounding boxes after NMS.
    """
    selected_indices = []

    # Sort bounding boxes by decreasing confidence scores
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        selected_indices.append(current_index)

        # Remove the current box from the sorted list
        sorted_indices.pop(0)

        indices_to_remove = []
        for index in sorted_indices:
            iou = calculate_iou(bboxes[current_index], bboxes[index])
            if iou >= iou_threshold:
                indices_to_remove.append(index)

        # Remove overlapping boxes from the sorted list
        sorted_indices = [i for i in sorted_indices if i not in indices_to_remove]

    return selected_indices

def postprocess(prediction: np.array, conf_thres: float = 0.15, iou_thres: float = 0.45, max_det: int = 300) -> np.array:
    """
    Perform post-processing on object detection predictions.

    Args:
        prediction (np.array): Detection predictions in a specific format.
        conf_thres (float): Confidence threshold for object detection.
        iou_thres (float): IoU threshold for NMS.
        max_det (int): Maximum number of detections to retain.

    Returns:
        np.array: Processed detection results.
    """
    bs = prediction.shape[0]  # batch size
    max_nms = 300  # maximum number of boxes into NMS
    max_wh = 7680
    output = [None] * bs

    for xi, x in enumerate(prediction):
        x = x.T
        if len(x) == 0:
            continue
        x[:, 4:] *= x[:, 3:4]
        box = convert_xywh_to_xyxy(x[:, :4])

        conf = x[:, 4:].max(1)
        max_conf_indices = x[:, 4:].argmax(1)
        x = np.column_stack((box, conf, max_conf_indices.astype(float)))[conf > conf_thres]

        n = len(x)
        if n == 0:
            continue
        elif n > max_nms:
            sorted_indices = np.argsort(-x[:, 4])
            x = x[sorted_indices[:max_nms]]

        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thres)
        if len(i) > max_det:
            i = i[:max_det]
        output[xi] = x[i]
    return output
