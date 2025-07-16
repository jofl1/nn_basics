import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from model import Darknet
from dataset import YOLODataset
from loss import YOLOv3Loss
from utils import (
    load_checkpoint,
    save_checkpoint,
)

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
IGNORE_THRESH = 0.5

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model & Data Configuration ---
NUM_CLASSES = 20  # Number of classes for PASCAL VOC, change as needed.
# The anchors are ordered from largest to smallest feature map.
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
IMAGE_SIZE = 416
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

# --- Dataset Paths ---
IMAGE_DIR = "datasets/images"
LABEL_DIR = "datasets/labels"

# --- Checkpoint & Weight Paths ---
CHECKPOINT_FILE = "yolov3.pth.tar"

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    """
    Defines the training loop for one epoch.
    """
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y0, y1, y2 = (
            y[0].to(DEVICE),
            y[1].to(DEVICE),
            y[2].to(DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update the progress bar description.
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)

def main():
    """
    Main function to run the training process.
    """
    model = Darknet(config_path="yolov3.cfg").to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    loss_fn = YOLOv3Loss()
    scaler = torch.cuda.amp.GradScaler()

    train_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
    )

    train_dataset = YOLODataset(
        img_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        anchors=ANCHORS,
        transform=train_transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
    )

    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1)
    ).to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if epoch > 0 and epoch % 10 == 0:
            save_checkpoint(model, optimizer, filename=f"checkpoint_epoch_{epoch}.pth.tar")

if __name__ == "__main__":
    main()