import torch
import time
import cv2
import argparse

from model import Darknet
from utils import preprocess_image, non_max_suppression, draw_detections

# COCO class names - 80 object categories
coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def detect_image(cfg_path, weights_path, img_path, output_path, conf_thres=0.5, nms_thres=0.4, num_classes=80, class_names=None):
    """
    Run YOLO object detection on a single image.
    
    Args:
        cfg_path: Path to YOLO configuration file (.cfg)
        weights_path: Path to pre-trained weights file (.weights)
        img_path: Path to input image
        output_path: Path to save output image with detections
        conf_thres: Confidence threshold for object detection
        nms_thres: IoU threshold for Non-Maximum Suppression
        
    Returns:
        detections: Tensor of final detections after NMS
    """
    # Check CUDA availability and set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
   
    # Load and initialise model
    model = Darknet(cfg_path, num_classes=num_classes)
    model.load_darknet_weights(weights_path)
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    model = model.to(device)  # Move model to GPU if available
   
    # Preprocess input image
    img_tensor, original_img = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)  # Move input tensor to same device as model
   
    # Warm up GPU 
    # Run a few forward passes to ensure CUDA kernels are loaded
    if device.type == 'cuda':
        for _ in range(3):
            _ = model(img_tensor)
        torch.cuda.synchronize()  # Wait for all GPU operations to complete
   
    # Run actual inference with timing
    start_time = time.time()
   
    with torch.no_grad():  # Disable gradient computation for inference
        detections = model(img_tensor)  # Forward pass through network
        # Concatenate outputs from all YOLO layers
        detections = torch.cat(detections, 1)
        detections = non_max_suppression(detections, conf_thres, nms_thres)  # Apply NMS
   
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Ensure GPU operations are complete for accurate timing
   
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time*1000:.2f} ms")
   
    # Draw detections on original image
    if len(detections) > 0:
        # Draw_detections modifies the image in-place
        # Move detections to CPU for drawing (OpenCV uses CPU)
        if class_names is None:
            class_names = coco_classes
        result_img = draw_detections(original_img, detections.cpu(), class_names=class_names)
    else:
        result_img = original_img
        print("No objects detected")
    
    # Save result image
    # OpenCV expects BGR format for saving, so convert from RGB
    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"Result saved to {output_path}")
    
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3.cfg', help='path to model config file')
    parser.add_argument('--weights', type=str, default='yolov3.weights', help='path to weights file')
    parser.add_argument('--image', type=str, default='test_image.jpg', help='path to input image')
    parser.add_argument('--output', type=str, default='detected_image.jpg', help='path to output image')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--num-classes', type=int, default=80, help='number of classes (80 for COCO, custom for your dataset)')
    opt = parser.parse_args()

    # For single class bike detection
    if opt.num_classes == 1:
        custom_class_names = ['bike']
    else:
        custom_class_names = None
        
    detections = detect_image(
        cfg_path=opt.cfg,
        weights_path=opt.weights,
        img_path=opt.image,
        output_path=opt.output,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        num_classes=opt.num_classes,
        class_names=custom_class_names
    )

    if detections is not None and len(detections) > 0:
        print(f"\nDetected {len(detections)} objects:")
        for det in detections:
            cls = int(det[6])
            conf = det[4]
            if custom_class_names:
                print(f"- {custom_class_names[cls]}: {conf:.2f}")
            elif opt.num_classes == 80:
                print(f"- {coco_classes[cls]}: {conf:.2f}")
            else:
                print(f"- Class {cls}: {conf:.2f}")

