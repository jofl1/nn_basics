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
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
IGNORE_THRESH = 0.5
NUM_WORKERS = 4  # Added for faster data loading

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model & Data Configuration ---
# The anchors are ordered from largest to smallest feature map.
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
IMAGE_SIZE = 416
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

# --- Checkpoint & Weight Paths ---
CHECKPOINT_FILE = "yolov3.pth.tar"


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    """
    Defines the training loop for one epoch.
    """
    model.train()
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
            x = x.to(DEVICE)
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
    print(f"Training on {DEVICE}")
    
    model = Darknet(cfg_path="yolov3.cfg", num_classes=opt.num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    loss_fn = YOLOv3Loss()
    scaler = torch.amp.GradScaler('cuda',)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[50, 80], 
        gamma=0.1
    )
    
    # Data augmentation
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
    
    # Training dataset
    train_dataset = YOLODataset(
        img_dir=opt.img_dir,
        label_dir=opt.label_dir,
        anchors=ANCHORS,
        transform=train_transform,
        num_classes=opt.num_classes
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,  # Added for faster data loading
        persistent_workers=True if NUM_WORKERS > 0 else False,  # Keep workers alive
    )
    
    # Optional: Validation dataset (using same transform for simplicity)
    val_dataset = None
    val_loader = None
    if opt.val_img_dir and opt.val_label_dir:
        val_dataset = YOLODataset(
            img_dir=opt.val_img_dir,
            label_dir=opt.val_label_dir,
            anchors=ANCHORS,
            transform=train_transform,  # You might want a separate val_transform without augmentation
            num_classes=opt.num_classes
        )
        
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
            persistent_workers=True if NUM_WORKERS > 0 else False,
        )
    
    # Scale anchors
    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1)
    ).to(DEVICE)
    
    # Load checkpoint if specified
    start_epoch = 0
    if opt.load_checkpoint:
        try:
            checkpoint = torch.load(opt.load_checkpoint)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"Loaded checkpoint from epoch {start_epoch - 1}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    # Training loop with error handling
    best_val_loss = float('inf')
    
    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\nEpoch [{epoch}/{NUM_EPOCHS}]")
            
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
                        filename="best_model.pth.tar",
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
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }
                save_checkpoint(
                    model, 
                    optimizer, 
                    filename=f"checkpoint_epoch_{epoch}.pth.tar",
                    epoch=epoch
                )
                # Also save as .weights format
                model.save_darknet_weights(f"yolov3_epoch_{epoch}.weights")
                print(f"Saved checkpoint and weights at epoch {epoch}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        save_checkpoint(
            model, 
            optimizer, 
            filename="interrupt_checkpoint.pth.tar",
            epoch=epoch
        )
        print("Saved interrupt checkpoint")
    
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    
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