"""
Implementation of Yolo Loss Function from the original yolo paper
"""

import torch
import torch.nn as nn
from .functions import intersection_over_union


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, split_size=7, num_boxes=2, num_classes=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # DEFAULT: S=7 B=2, C=20   -->  (... 20 classes ... , prob1, x1, y1, w1, h1, prob2, x2, y2, w2, h2)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)  # (BATCH_SIZE, S*S(C+B*5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        # (1, 7, 7, 30) --> predictions[..., 21:25] --> shape=(1, 7, 7, 4)
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])  # 21 ~ 24 = (x, y, w, h)
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])  # 26 ~ 29 = (x, y, w, h)

        # iou_b1.shape --> (1, 7, 7, 1)

        # unsqueeze() = Returns a tensor with a dimension of size one inserted at the specified position
        # (1, 7, 7, 1) --> unsqueeze(0) --> (1, 1, 7, 7, 1)
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)  # (2, 1, 7, 7, 1)

        # iou_maxes(_) = highest IoU value out of the two prediction
        # bestbox will be indices of 0 or 1 (2 predicted boxes) for which bbox was best (argmax)
        _, bestbox = torch.max(ious, dim=0)  # (2, 1, 7, 7, 1) --> (1, 7, 7, 1), (1, 7, 7, 1)

        # look up expected prob (21st index) ** 1 or 0 **
        identify_box = target[..., 20].unsqueeze(3)  # identity obj_i (is there an object in cell i) --> (1, 7, 7, 1)
        # identify_box = target[..., 20:21]

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # set boxes with no object in them to 0
        # we only take out the highest IOU among the 2 box candidates.
        box_predictions = identify_box * (
            (
                    bestbox * predictions[..., 26:30]  # bestbox = either 0 or 1
                    + (1 - bestbox) * predictions[..., 21:25]
            )
        )  # (1, 7, 7, 4): contains best box coordinates (x, y, w, h)

        box_targets = identify_box * target[..., 21:25]  # (1, 7, 7, 4)

        # take sqrt of width, height  |  torch.sign() --> return +1/-1/0
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )  # (1, 7, 7, 4): contains best box coordinates (x, y, sqrt(w), sqrt(h))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),  # squeeze 0, 1, 2 dimensions
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        pred_box = (
                bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )  # confidence score for the highest IoU box --> (1, 7, 7, 1)

        object_loss = self.mse(
            torch.flatten(identify_box * pred_box),
            torch.flatten(identify_box * target[..., 20:21]),
        )  # (BATCH_SIZE, 7, 7)

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        # max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
        # no_object_loss = self.mse(
        #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
        #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        # )

        # box1 MSE
        no_object_loss = self.mse(
            # MSE( predicted_confidence * identify_func(1 or 0) , expected_confidence * identify_func(1 or 0) )
            torch.flatten((1 - identify_box) * predictions[..., 20:21], start_dim=1),  # (1, 7, 7, 1) --flat--> (1, 49)
            torch.flatten((1 - identify_box) * target[..., 20:21], start_dim=1),
        )

        # accumulate box2 MSE
        no_object_loss += self.mse(
            torch.flatten((1 - identify_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - identify_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(identify_box * predictions[..., :20], end_dim=-2, ),
            torch.flatten(identify_box * target[..., :20], end_dim=-2, ),
        )

        loss = (
                self.lambda_coord * box_loss  # first two rows in paper
                + object_loss  # third row in paper
                + self.lambda_noobj * no_object_loss  # forth row
                + class_loss  # fifth row
        )

        return loss
