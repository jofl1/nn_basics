

import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class YOLODataset(Dataset):
    """
    Creates a PyTorch Dataset for YOLOv3.
    """
    def __init__(self, img_dir, label_dir, anchors, image_size=416, S=None, C=20, transform=None):
        """
        Args:
            img_dir (str): Path to the directory with images.
            label_dir (str): Path to the directory with labels.
            anchors (list): A list of anchor boxes.
            image_size (int): The size to which images are resized.
            S (list): A list of grid sizes for each scale (e.g., [13, 26, 52]).
            C (int): The number of classes.
            transform: Albumentations transform
        """
        if S is None:
            S = [13, 26, 52]
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        label_path = os.path.join(self.label_dir, self.images[index].replace("jpg", "txt"))

        image = np.array(Image.open(img_path).convert("RGB"))
        bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [torch.zeros((self.num_anchors_per_scale, s, s, 6)) for s in self.S]

        for box in bboxes:
            iou_anchors = self.iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            
            class_label, x, y, width, height = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = width * S, height * S
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)

    def iou(self, box1_wh, box2_wh):
        """
        Calculates Intersection over Union for two sets of boxes, given as (w, h).
        This is used to find the best anchor for a ground truth box.
        """
        intersection = torch.min(box1_wh[..., 0], box2_wh[..., 0]) * torch.min(box1_wh[..., 1], box2_wh[..., 1])
        union = (box1_wh[..., 0] * box1_wh[..., 1] + 1e-6) + \
                (box2_wh[..., 0] * box2_wh[..., 1]) - intersection
        return intersection / union
