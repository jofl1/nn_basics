import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import argparse
import time
from model import Darknet
from dataset import YOLODataset
from loss import YOLOv3Loss
from utils import (
    load_checkpoint,
    save_checkpoint,
)

# --- Training Hyperparameters ---
learning_rate = 1e-4
batch_size = 16
num_epochs = 100
conf_threshold = 0.9
nms_threshold = 0.5
ignore_thresh = 0.5
num_workers = 4  # Added for faster data loading

# --- Device Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model & Data Configuration ---
# The anchors are ordered from largest to smallest feature map.
anchors = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
image_size = 416
s = [image_size // 32, image_size // 16, image_size // 8]

# --- Checkpoint & Weight Paths ---
checkpoint_file = "yolov3.pth"


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    """
    Defines the training loop for one epoch.
    """
    model.train()
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        with torch.amp.autocast('cuda'):
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
    
    return mean_loss


def validate_fn(val_loader, model, loss_fn, scaled_anchors):
    """
    Validation loop to monitor performance.
    """
    model.eval()
    loop = tqdm(val_loader, leave=True, desc="Validating")
    losses = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(device)
            y0, y1, y2 = (
                y[0].to(DEVICE),
                y[1].to(DEVICE),
                y[2].to(DEVICE),
            )
            
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
            
            losses.append(loss.item())
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(val_loss=mean_loss)
    
    model.train()
    return mean_loss


def main(opt):
    """
    Main function to run the training process.
    """
    print(f"Training on {device}")
    
    model = Darknet(cfg_path="yolov3.cfg", num_classes=opt.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    loss_fn = YOLOv3Loss()
    scaler = torch.amp.GradScaler('cuda',)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[50, 80], 
        gamma=0.1
    )
    
    train_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo", 
            min_visibility=0.4, 
            label_fields=[],
            clip=True  # Add this line
        ),
    )
    
    # Training dataset
    train_dataset = YOLODataset(
        img_dir=opt.img_dir,
        label_dir=opt.label_dir,
        anchors=anchors,
        transform=train_transform,
        num_classes=opt.num_classes
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,  # Added for faster data loading
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
    )
    
    # Optional Validation dataset (using same transform for simplicity)
    val_dataset = None
    val_loader = None
    if opt.val_img_dir and opt.val_label_dir:
        val_dataset = YOLODataset(
            img_dir=opt.val_img_dir,
            label_dir=opt.val_label_dir,
            anchors=anchors,
            transform=train_transform,  # You might want a separate val_transform without augmentation
            num_classes=opt.num_classes
        )
        
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            persistent_workers=True if NUM_WORKERS > 0 else False,
        )
    
    # Scale anchors
    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(s).unsqueeze(1).unsqueeze(1)
    ).to(device)
    
    # Load checkpoint if specified
    start_epoch = 0
    if opt.load_checkpoint:
        checkpoint = torch.load(opt.load_checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Loaded checkpoint from epoch {start_epoch - 1}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch [{epoch}/{num_epochs}]")
        
        # Training with timing
        epoch_start_time = time.time()
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
        epoch_time = time.time() - epoch_start_time
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Epoch Time: {epoch_time:.2f} seconds")
        
        # Validation
        if val_loader is not None:
            val_loss = validate_fn(val_loader, model, loss_fn, scaled_anchors)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, 
                    optimizer, 
                    filename="best_model.pth",
                    epoch=epoch
                )
                print("Saved best model!")
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Regular checkpoint saving
        if epoch % 10 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            save_checkpoint(
                model, 
                optimizer, 
                filename=f"checkpoint_epoch_{epoch}.pth",
                epoch=epoch
            )
            # Also save as .weights format
            model.save_darknet_weights(f"yolov3_epoch_{epoch}.weights")
            print(f"Saved checkpoint and weights at epoch {epoch}")
    
    print("\nTraining completed!")
    
    # Save final weights
    model.save_darknet_weights("yolov3_final.weights")
    print("Saved final weights as yolov3_final.weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, default='datasets/images', 
                        help='path to training image directory')
    parser.add_argument('--label-dir', type=str, default='datasets/labels', 
                        help='path to training label directory')
    parser.add_argument('--val-img-dir', type=str, default=None, 
                        help='path to validation image directory (optional)')
    parser.add_argument('--val-label-dir', type=str, default=None, 
                        help='path to validation label directory (optional)')
    parser.add_argument('--num-classes', type=int, default=20, 
                        help='number of classes in the dataset')
    parser.add_argument('--load-checkpoint', type=str, default=None, 
                        help='path to checkpoint to resume training from')
    opt = parser.parse_args()
    
    main(opt)