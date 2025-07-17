# YOLOv3 Implementation

A clean, simplified implementation of YOLOv3 for custom dataset training and inference.

## Requirements

- PyTorch
- torchvision
- opencv-python
- numpy
- tqdm
- albumentations

## Training

To train on a custom dataset:

```bash
python train.py --img-dir path/to/images --label-dir path/to/labels --num-classes 20
```

Key features:
- Epoch timing for performance monitoring
- Saves weights in both .pth.tar and .weights format
- Automatic checkpoint saving every 10 epochs
- Learning rate scheduling

## Inference

To run inference with pretrained COCO weights:

```bash
python detect.py --weights yolov3.weights --image test_image.jpg --output result.jpg
```

To run inference with custom trained weights:

```bash
python detect.py --weights yolov3_final.weights --image test_image.jpg --output result.jpg --num-classes 20
```

## Dataset Format

Expected format for custom datasets:
- Images: JPG files in the images directory
- Labels: TXT files in the labels directory (same filename as image)
- Label format: `class_index x_centre y_centre width height` (normalised to 0-1)

## Output

Training produces:
- `yolov3_final.weights` - Final trained weights in Darknet format
- `yolov3_epoch_N.weights` - Checkpoint weights every 10 epochs
- `checkpoint_epoch_N.pth.tar` - PyTorch checkpoints for resuming training