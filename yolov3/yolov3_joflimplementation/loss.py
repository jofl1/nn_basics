import torch
import torch.nn as nn

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    intersection = (torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)).clamp(0) * \
                   (torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

class YOLOv3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.entropy_loss = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # No Object Loss
        no_object_loss = self.bce_loss(
            (predictions[..., 0:1][no_obj]), (torch.zeros_like(target[..., 0:1][no_obj])),
        )

        # Object Loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce_loss((predictions[..., 0:1][obj]), (ious * torch.ones_like(target[..., 0:1][obj])))

        # Box Coordinate Loss
        pred_boxes = predictions[..., 1:5].clone()
        pred_boxes[..., 0:2] = self.sigmoid(pred_boxes[..., 0:2])
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5]) / anchors
        )
        box_loss = self.mse_loss(pred_boxes[obj], target[..., 1:5][obj])

        # Class Loss
        class_loss = self.entropy_loss(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )

