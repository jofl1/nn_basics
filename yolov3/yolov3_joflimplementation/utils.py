import torch
import torchvision.ops as ops
import cv2
import numpy as np

def preprocess_image(img_path, img_size=416):
    """
    Load and preprocess image for YOLO inference.
    YOLO expects square inputs with letterbox padding to maintain aspect ratio.
    
    Args:
        img_path: Path to input image
        img_size: Target size for the square image (default 416x416)
        
    Returns:
        img_tensor: Preprocessed image tensor [1, 3, img_size, img_size]
        img: Original image in RGB format (for visualisation)
    """
    # Load image using OpenCV (loads as BGR by default)
    img = cv2.imread(img_path)
    # Convert BGR to RGB for consistency
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    # Calculate scaling factor to fit image in square while maintaining aspect ratio
    h, w = img.shape[:2]
    scale = min(img_size/w, img_size/h)  # Scale to fit within img_size
    new_w = int(w * scale)
    new_h = int(h * scale)
   
    # Resize image maintaining aspect ratio
    img_resized = cv2.resize(img, (new_w, new_h))
   
    # Create grey padded square image (128 is grey in 0-255 range)
    img_padded = np.full((img_size, img_size, 3), 128, dtype=np.uint8)
    # Calculate padding to centre the image
    dw = (img_size - new_w) // 2
    dh = (img_size - new_h) // 2
    # Place resized image in centre of padded image
    img_padded[dh:dh+new_h, dw:dw+new_w] = img_resized
   
    # Convert to PyTorch tensor and normalise to [0, 1]
    img_tensor = torch.from_numpy(img_padded).float().div(255.0)
    # Reorder dimensions from HWC to CHW and add batch dimension
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
   
    return img_tensor, img

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Perform Non-Maximum Suppression (NMS) to remove overlapping detections.
    Uses torchvision's optimised NMS implementation.
    
    Args:
        prediction: Raw YOLO output [batch_size, num_detections, 5+num_classes]
                   Format: [x_centre, y_centre, width, height, objectness, ...class_scores]
        conf_thres: Minimum objectness confidence threshold
        nms_thres: IoU threshold for NMS
        
    Returns:
        List of filtered detections with format:
        [x1, y1, x2, y2, objectness_conf, class_score, class_pred]
    """
   
    # Get batch size
    batch_size = prediction.size(0)
   
    # Convert from centre format to corner format for NMS
    # From (centre x, centre y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
   
    output = []
   
    # Process each image in the batch independently
    for image_i in range(batch_size):
        image_pred = prediction[image_i]  # Get predictions for this image
       
        # Filter out detections with low objectness confidence
        conf_mask = (image_pred[:, 4] >= conf_thres)
        image_pred = image_pred[conf_mask]
       
        # If no detections remain after confidence filtering, skip this image
        if not image_pred.size(0):
            continue
           
        # Get class with highest confidence for each detection
        # class_confs: highest class confidence scores
        # class_preds: indices of classes with highest confidence
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
       
        # Concatenate box coordinates, objectness, class confidence, and class prediction
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
       
        # Get unique classes detected in this image
        unique_classes = detections[:, -1].unique()
        
        # Perform NMS separately for each class (standard practice in object detection)
        for c in unique_classes:
            # Get detections for this specific class
            detections_class = detections[detections[:, -1] == c]
            
            # Use torchvision's optimised NMS implementation
            # NMS expects: boxes [N, 4], scores [N], iou_threshold
            keep = ops.nms(
                detections_class[:, :4],  # Bounding boxes [x1, y1, x2, y2]
                detections_class[:, 4],   # Objectness scores
                nms_thres                 # IoU threshold
            )
            
            # Add kept detections to output
            if len(keep) > 0:
                output.extend(detections_class[keep])
           
    return torch.stack(output) if output else torch.FloatTensor(0, 7)

def xywh2xyxy(x):
    """
    Convert bounding box format from centre coordinates to corner coordinates.
    
    Args:
        x: Tensor of boxes in [x_centre, y_centre, width, height] format
        
    Returns:
        Tensor of boxes in [x1, y1, x2, y2] format (top-left and bottom-right corners)
    """
    y = x.new(x.shape)  # Create new tensor on same device as input
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = x_centre - width/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = y_centre - height/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = x_centre + width/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = y_centre + height/2
    return y

def draw_detections(img, detections, img_size=416, class_names=None):
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        img: Original image in RGB format (numpy array)
        detections: Tensor of detections [num_detections, 7]
                   Format: [x1, y1, x2, y2, objectness, class_conf, class_id]
        img_size: Size of the padded square image used for inference
        class_names: List of class names for labeling
        
    Returns:
        img: Image with drawn detections (modified in-place)
    """
    
    # Calculate scaling factors to map from padded image back to original image
    # Need to reverse the letterbox transformation
    h, w = img.shape[:2]
    scale = min(img_size / w, img_size / h)  # Same scale used in preprocessing
    new_w = int(w * scale)
    new_h = int(h * scale)
    # Calculate padding that was added during preprocessing
    dw = (img_size - new_w) // 2
    dh = (img_size - new_h) // 2
    
    
    colour = (0, 255, 0)  # Green in RGB format
    
    # Process each detection
    for det in detections:
        x1, y1, x2, y2, conf, cls_conf, cls = det
        
        # Transform coordinates from padded image space back to original image space
        # First, subtract padding offset, then scale back to original size
        x1 = int((x1 - dw) / scale)
        y1 = int((y1 - dh) / scale)
        x2 = int((x2 - dw) / scale)
        y2 = int((y2 - dh) / scale)
        
        # Draw bounding box rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        
        # Create label with class name and confidence score
        if class_names:
            label = f'{class_names[int(cls)]}: {conf:.2f}'
            # Draw label above the bounding box
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
        
    # Image is modified in-place
    return img

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar", epoch=None):
    """Saves the model state."""
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if epoch is not None:
        checkpoint["epoch"] = epoch
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """Loads the model state."""
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we are loading a checkpoint, we might need to update the learning rate.
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
