import torch


def intersection_over_union(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor,
                            box_format="midpoint") -> torch.Tensor:
    """
    Calculates intersection over union
    :param boxes_preds: Predictions of Bounding Boxes (BATCH_SIZE, 4)
    :param boxes_labels: Correct labels of Bounding Boxes (BATCH_SIZE, 4)
    :param box_format: midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    :return: Intersection over union for all examples
    """
    # box_preds = shape=(1, 7, 7, 4)

    # boxes_preds[..., 0:1] --> (1, 7, 7, 1)
    # boxes_preds[..., 0] --> (1, 7, 7)
    x_hat = boxes_preds[..., 0:1]  # (1, 7, 7, 1)
    y_hat = boxes_preds[..., 1:2]
    w_hat = boxes_preds[..., 2:3]
    h_hat = boxes_preds[..., 3:4]

    x = boxes_labels[..., 0:1]  # (1, 7, 7, 1)
    y = boxes_labels[..., 1:2]
    w = boxes_labels[..., 2:3]
    h = boxes_labels[..., 3:4]

    if box_format == "midpoint":
        # lower left corner
        box1_x1 = x_hat - w_hat / 2
        box1_y1 = y_hat - h_hat / 2

        box2_x1 = x - w / 2
        box2_y1 = y - h / 2

        # upper right corner
        box1_x2 = x_hat + w_hat / 2
        box1_y2 = y_hat + h_hat / 2

        box2_x2 = x + w / 2
        box2_y2 = y + h / 2

    if box_format == "corners":
        # centre
        box1_x1 = x
        box1_y1 = y

        box2_x1 = x_hat
        box2_y1 = y_hat

        # upper right corner
        box1_x2 = w
        box1_y2 = h  # (N, 1)

        box2_x2 = w_hat
        box2_y2 = h_hat

    # find intersection corners
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
