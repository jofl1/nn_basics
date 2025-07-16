import torch
import torch.nn as nn
import torchvision.ops as ops  # Added for optimised NMS
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import argparse

class YOLOLayer(nn.Module):
    """
YOLO detection layer that processes feature maps and outputs bounding box predictions.
This layer is responsible for converting the raw CNN output into interpretable object detections.
    """
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors  # Pre-defined anchor box dimensions (width, height) in pixels
        self.num_anchors = len(anchors)  # Number of anchor boxes per grid cell (typically 3)
        self.num_classes = num_classes  # Number of object classes (80 for COCO)
        self.img_size = img_size  # Input image size (416x416)
        self.grid_size = 0  # Will be set dynamically based on feature map size
        self.stride = 0  # Pixel stride between grid cells
       
    def forward(self, x):
        """
        Forward pass through YOLO layer.
        
        Args:
            x: Feature map tensor of shape [batch_size, num_anchors*(5+num_classes), grid_size, grid_size]
               where 5 represents: x, y, width, height, objectness confidence
        
        Returns:
            Tensor of shape [batch_size, num_grid_cells*num_anchors, 5+num_classes] containing:
            - Scaled bounding box coordinates (x1, y1, x2, y2) in image space
            - Objectness confidence score
            - Class probability scores
        """
        batch_size = x.size(0)
        grid_size = x.size(2)  # Feature map is square, so height = width = grid_size
       
        # Reshape predictions from flat channel dimension to structured format
        # From: [batch, channels, height, width]
        # To: [batch, num_anchors, 5+num_classes, height, width]
        prediction = x.view(batch_size, self.num_anchors,
                          self.num_classes + 5, grid_size, grid_size)
        # Reorder dimensions for easier processing
        # To: [batch, num_anchors, height, width, 5+num_classes]
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
       
        # Extract and apply activation functions to predictions
        # Sigmoid constrains x,y to [0,1] within each grid cell
        x = torch.sigmoid(prediction[..., 0])  # Centre x coordinate (relative to grid cell)
        y = torch.sigmoid(prediction[..., 1])  # Centre y coordinate (relative to grid cell)
        # Width and height are in log space 
        w = prediction[..., 2]  # Width 
        h = prediction[..., 3]  # Height 
        # Objectness: probability that this anchor contains an object
        conf = torch.sigmoid(prediction[..., 4])  # Confidence/objectness score
        # Class predictions: probability distribution over classes
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class probabilities
       
        # Calculate stride: how many pixels in the original image correspond to one grid cell
        # E.g., if image is 416x416 and grid is 13x13, stride = 32
        stride = self.img_size // grid_size
       
        # Create grids of x,y coordinates for each cell
        # These represent the top-left corner of each grid cell
        # grid_x: [[0,1,2,...,12], [0,1,2,...,12], ...] for a 13x13 grid
        grid_x = torch.arange(grid_size, dtype=torch.float32, device=x.device).repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
        # grid_y: [[0,0,0,...,0], [1,1,1,...,1], ..., [12,12,12,...,12]] for a 13x13 grid
        grid_y = torch.arange(grid_size, dtype=torch.float32, device=x.device).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])
       
        # Scale anchor boxes from pixel coordinates to grid coordinates
        # This converts anchors from image space to feature map space
        scaled_anchors = [(a[0]/stride, a[1]/stride) for a in self.anchors]
        # Extract widths and heights separately, creating tensors on the correct device
        anchor_w = torch.tensor([a[0] for a in scaled_anchors], dtype=torch.float32, device=x.device)
        anchor_h = torch.tensor([a[1] for a in scaled_anchors], dtype=torch.float32, device=x.device)
        # Reshape for broadcasting: [batch, num_anchors, 1, 1]
        anchor_w = anchor_w.repeat(batch_size, 1).view(batch_size, self.num_anchors, 1, 1)
        anchor_h = anchor_h.repeat(batch_size, 1).view(batch_size, self.num_anchors, 1, 1)
       
        # Convert predictions to bounding boxes in grid space
        pred_boxes = torch.zeros_like(prediction[..., :4])
        # x,y predictions are relative to grid cell, add grid coordinates for absolute position
        pred_boxes[..., 0] = x + grid_x  # Absolute x in grid coordinates
        pred_boxes[..., 1] = y + grid_y  # Absolute y in grid coordinates
        # Width/height use exponential to ensure positive values, multiplied by anchor dimensions
        pred_boxes[..., 2] = torch.exp(w) * anchor_w  # Absolute width in grid coordinates
        pred_boxes[..., 3] = torch.exp(h) * anchor_h  # Absolute height in grid coordinates
       
        # Reshape and scale outputs to image coordinates
        # Flatten spatial dimensions: [batch, num_anchors*grid*grid, 4]
        # Multiply by stride to convert from grid coordinates to pixel coordinates
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * stride,
                           conf.view(batch_size, -1, 1),
                           pred_cls.view(batch_size, -1, self.num_classes)), -1)
       
        return output

class Darknet(nn.Module):
    """
    Darknet neural network architecture for YOLO object detection.
    Parses configuration file and builds the network dynamically.
    """
    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()
        self.blocks = self.parse_cfg(cfg_path)  # Parse network architecture from config
        self.img_size = img_size
        self.module_list = self.create_modules(self.blocks)  # Build PyTorch modules
       
    def parse_cfg(self, cfg_path):
        """
        Parse Darknet configuration file (.cfg) into a list of layer dictionaries.
        
        Args:
            cfg_path: Path to the .cfg file
            
        Returns:
            List of dictionaries, each representing a network layer/block
        """
        with open(cfg_path, 'r') as f:
            lines = f.read().split('\n')
        # Remove empty lines and comments
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.strip() for x in lines]
       
        blocks = []
        block = {}
       
        for line in lines:
            if line.startswith('['):
                # New block starts - save previous block if it exists
                if block:
                    blocks.append(block)
                block = {}
                block['type'] = line[1:-1]  # Extract block type (e.g., 'convolutional', 'yolo')
            else:
                # Parse key=value pairs within a block
                key, value = line.split('=')
                block[key.strip()] = value.strip()
        blocks.append(block)  # Don't forget the last block
       
        return blocks
   
    def create_modules(self, blocks):
        """
        Convert parsed configuration blocks into PyTorch modules.
        
        Args:
            blocks: List of configuration dictionaries
            
        Returns:
            nn.ModuleList containing the network layers
        """
        net_info = blocks[0]  # First block contains network hyperparameters
        module_list = nn.ModuleList()
        prev_filters = 3  # RGB input has 3 channels
        output_filters = []  # Track output channels for each layer (needed for route/shortcut layers)
       
        # Iterate through blocks (skip net_info at index 0)
        for idx, block in enumerate(blocks[1:]):
            module = nn.Sequential()
           
            if block['type'] == 'convolutional':
                # Standard convolutional layer
                filters = int(block['filters'])  # Number of output channels
                kernel_size = int(block['size'])  # Kernel dimensions (square)
                stride = int(block['stride'])  # Convolution stride
                # Calculate padding to maintain spatial dimensions (same padding)
                pad = (kernel_size - 1) // 2 if block.get('pad') else 0
               
                # Bias is only used when there's no batch normalisation
                # (Batch norm includes its own bias term)
                has_bias = 'batch_normalize' not in block
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=has_bias)
                module.add_module(f'conv_{idx}', conv)
               
                # Batch normalisation (if specified)
                if 'batch_normalize' in block:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module(f'batch_norm_{idx}', bn)
               
                # Activation function (typically leaky ReLU for YOLO)
                if block['activation'] == 'leaky':
                    activn = nn.LeakyReLU(0.1, inplace=True)  # Negative slope of 0.1
                    module.add_module(f'leaky_{idx}', activn)
                   
            elif block['type'] == 'upsample':
                # Upsampling layer (used in YOLOv3 for feature pyramid)
                upsample = nn.Upsample(scale_factor=int(block['stride']), mode='nearest')
                module.add_module(f'upsample_{idx}', upsample)
               
            elif block['type'] == 'route':
                # Route layer concatenates features from one or more previous layers
                layers = block['layers'].split(',')
                layers = [int(x) for x in layers]
               
                if len(layers) == 1:
                    # Single route: just pass through features from specified layer
                    filters = output_filters[layers[0]]
                else:
                    # Multiple routes: concatenate features (sum of channels)
                    filters = sum([output_filters[l] for l in layers])
               
                # Use Identity as placeholder (actual routing happens in forward pass)
                module.add_module(f'route_{idx}', nn.Identity())
               
            elif block['type'] == 'shortcut':
                # Shortcut/residual connection (adds features from previous layer)
                module.add_module(f'shortcut_{idx}', nn.Identity())
               
            elif block['type'] == 'yolo':
                # YOLO detection layer
                # Parse which anchors to use (mask indices)
                mask = block['mask'].split(',')
                mask = [int(x) for x in mask]
               
                # Parse all anchor boxes and select ones specified by mask
                anchors = block['anchors'].split(',')
                anchors = [(int(anchors[i]), int(anchors[i+1]))
                          for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
               
                num_classes = int(block['classes'])
                img_size = int(net_info['height'])
               
                yolo = YOLOLayer(anchors, num_classes, img_size)
                module.add_module(f'yolo_{idx}', yolo)
               
            module_list.append(module)
            # Track output channels for each layer
            output_filters.append(filters if block['type'] != 'yolo' else prev_filters)
            prev_filters = filters if block['type'] != 'yolo' else prev_filters
           
        return module_list
   
    def forward(self, x):
        """
        Forward pass through the entire network.
        
        Args:
            x: Input image tensor [batch_size, 3, height, width]
            
        Returns:
            Concatenated detections from all YOLO layers
        """
        outputs = []  # Collect outputs from YOLO layers
        layer_outputs = []  # Store outputs from all layers
       
        # Process each layer sequentially
        for i, (block, module) in enumerate(zip(self.blocks[1:], self.module_list)):
            if block['type'] in ['convolutional', 'upsample']:
                # Standard forward pass
                x = module(x)
               
            elif block['type'] == 'route':
                # Concatenate features from specified layers
                layers = block['layers'].split(',')
                layers = [int(x) for x in layers]
               
                if len(layers) == 1:
                    # Single route: use features from specified layer
                    x = layer_outputs[layers[0]]
                else:
                    # Multiple routes: concatenate along channel dimension
                    x = torch.cat([layer_outputs[l] for l in layers], 1)
                   
            elif block['type'] == 'shortcut':
                # Add features from specified previous layer 
                from_layer = int(block['from'])
                x = layer_outputs[-1] + layer_outputs[from_layer]
               
            elif block['type'] == 'yolo':
                # YOLO detection layer
                x = module[0](x)  # module is nn,seq ibject that holds layers for current block, x is used to call modules forward method - x is feature map tensor passed as input
                outputs.append(x) # Fromatted bounding box predicitions at specific scael 
               
            layer_outputs.append(x)  # Save output for potential route/shortcut layers
           
        # Concatenate all YOLO outputs along the detection dimension - net depths 82, 94 and 106 are output, outpus contain these three prediction tensors. tich.cat combines
        return torch.cat(outputs, 1)
   
    def load_darknet_weights(self, weights_path):
        """
        Load pre-trained weights from official Darknet format.
        Darknet stores weights as a binary file with a specific ordering.
        
        Args:
            weights_path: Path to .weights file
        """
        with open(weights_path, 'rb') as f:
            # First 5 int32 values are header information
            header = np.fromfile(f, dtype=np.int32, count=5)
            # Rest of file contains float32 weight values
            weights = np.fromfile(f, dtype=np.float32)
           
        print(f"Loading weights from {weights_path}")
        print(f"Total weights in file: {len(weights)}")
           
        ptr = 0  # Pointer to current position in weights array
        
        # Iterate through layers and load weights
        for i, (block, module) in enumerate(zip(self.blocks[1:], self.module_list)):
            if block['type'] == 'convolutional':
                conv_layer = module[0]
                if 'batch_normalize' in block:
                    # Batch normalised layer: load BN parameters first, then conv weights
                    bn_layer = module[1]
                   
                    # Batch norm parameters are stored in order:
                    # 1. bias, 2. weight (scale), 3. running mean, 4. running variance
                    num_bn_biases = bn_layer.bias.numel()
                   
                    # Check if we have enough weights remaining
                    if ptr + num_bn_biases > len(weights):
                        raise RuntimeError(f"Not enough weights for BN bias at layer {i}")
                   
                    # Load batch norm bias
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                   
                    # Load batch norm weights (scale factors)
                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                   
                    # Load batch norm running mean
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                   
                    # Load batch norm running variance
                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                   
                    # Copy loaded values to model parameters
                    bn_layer.bias.data.copy_(bn_biases.view_as(bn_layer.bias.data))
                    bn_layer.weight.data.copy_(bn_weights.view_as(bn_layer.weight.data))
                    bn_layer.running_mean.copy_(bn_running_mean.view_as(bn_layer.running_mean))
                    bn_layer.running_var.copy_(bn_running_var.view_as(bn_layer.running_var))
                else:
                    # No batch norm: load convolutional bias
                    num_biases = conv_layer.bias.numel()
                   
                    # Check if we have enough weights remaining
                    if ptr + num_biases > len(weights):
                        raise RuntimeError(f"Not enough weights for conv bias at layer {i}")
                   
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases
                    conv_layer.bias.data.copy_(conv_biases.view_as(conv_layer.bias.data))
               
                # Load convolutional weights (same for both cases)
                num_weights = conv_layer.weight.numel()
               
                # Check if we have enough weights remaining
                if ptr + num_weights > len(weights):
                    raise RuntimeError(f"Not enough weights for conv weights at layer {i}. Need {num_weights}, have {len(weights) - ptr}")
               
                try:
                    # Load and reshape weights to match layer dimensions
                    conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                    ptr += num_weights
                    conv_layer.weight.data.copy_(conv_weights.view_as(conv_layer.weight.data))
                except RuntimeError as e:
                    print(f"Error at layer {i}: {block}")
                    print(f"Conv layer shape: {conv_layer.weight.shape}")
                    print(f"Trying to load {num_weights} weights")
                    print(f"Available weights: {len(weights) - ptr}")
                    raise e
       
        print(f"Loaded weights: {ptr} / {len(weights)} values used")

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
    prediction[..., :4] = box_centre_to_corners(prediction[..., :4])
   
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
        for class_index in unique_classes:
            # Get detections for this specific class
            detections_class = detections[detections[:, -1] == class_index]
            
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

def box_centre_to_corners(x):
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

def scale_coords_to_original_image(coords, original_shape, padded_size):
    """
    Rescales coordinates from the padded image space (used for network input)
    back to the original image space.
    """
    h, w = original_shape
    scale = min(padded_size / w, padded_size / h)
    dw = (padded_size - int(w * scale)) // 2
    dh = (padded_size - int(h * scale)) // 2

    x1, y1, x2, y2 = coords

    x1 = (x1 - dw) / scale
    y1 = (y1 - dh) / scale
    x2 = (x2 - dw) / scale
    y2 = (y2 - dh) / scale
    
    return x1, y1, x2, y2


# COCO class names - 80 object categories
COCO_CLASSES = [
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

def draw_detections(img, detections, img_size=416):
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        img: Original image in RGB format (numpy array)
        detections: Tensor of detections [num_detections, 7]
                   Format: [x1, y1, x2, y2, objectness, class_conf, class_id]
        img_size: Size of the padded square image used for inference
        
    Returns:
        img: Image with drawn detections (modified in-place)
    """
    
    colour = (0, 255, 0)  # Green in RGB format
    
    # Process each detection
    for det in detections:
        x1, y1, x2, y2, conf, cls_conf, cls = det
        
        # Transform coordinates from padded image space back to original image space
        box_coords = (x1, y1, x2, y2)
        x1, y1, x2, y2 = scale_coords_to_original_image(box_coords, img.shape[:2], img_size)
        
        # Draw bounding box rectangle
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        
        # Create label with class name and confidence score
        label = f'{COCO_CLASSES[int(cls)]}: {conf:.2f}'
        # Draw label above the bounding box
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
        
    # Image is modified in-place
    return img

def visualise_raw_predictions(prediction, img, conf_threshold=0.1, max_boxes=50):
    """
    Visualise ALL predictions before NMS to see what the network actually detects.
    
    FIXED: Added proper coordinate transformation from padded space to original image space
    """
    # Convert to corner format
    pred_copy = prediction.clone()
    pred_copy[..., :4] = box_centre_to_corners(pred_copy[..., :4])
    
    h, w = img.shape[:2]
    img_size = 416  # Assuming standard YOLO input size
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    
    boxes_shown = 0
    # Get all predictions above low threshold (to see what network sees)
    for i in range(pred_copy.size(1)):
        objectness = pred_copy[0, i, 4]
        if objectness > conf_threshold and boxes_shown < max_boxes:
            box_coords = pred_copy[0, i, :4]
            
            # Transform coordinates back to original image space
            x1, y1, x2, y2 = scale_coords_to_original_image(box_coords, img.shape[:2], img_size)
            
            # Clip to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Get best class
            class_confs, class_preds = pred_copy[0, i, 5:].max(0)
            
            # Draw box with transparency based on confidence
            alpha = float(objectness) * 0.7
            colour = plt.cm.rainbow(int(class_preds) / 80)[:3]
            
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor=colour, 
                           facecolor='none', alpha=alpha)
            ax.add_patch(rect)
            
            # Add label
            label = f'{COCO_CLASSES[int(class_preds)]}: {objectness:.2f}'
            ax.text(x1, y1-5, label, color=colour, fontsize=8, alpha=alpha)
            boxes_shown += 1
    
    ax.set_title(f'Raw Predictions (before NMS) - Showing {boxes_shown}/{pred_copy.size(1)} boxes')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def print_detection_details(detections, top_k=5):
    """
    Print detailed information about each detection including full class probability vectors.
    """
    if len(detections) == 0:
        print("No detections to analyse.")
        return
        
    print("\n" + "="*80)
    print("DETAILED DETECTION ANALYSIS")
    print("="*80)
    
    for idx, det in enumerate(detections):
        x1, y1, x2, y2, objectness, cls_conf, cls_id = det
        
        print(f"\nDetection #{idx + 1}:")
        print(f"  Class: {COCO_CLASSES[int(cls_id)]} (ID: {int(cls_id)})")
        print(f"  Objectness Score: {objectness:.4f}")
        print(f"  Class Confidence: {cls_conf:.4f}")
        print(f"  Combined Score: {objectness * cls_conf:.4f}")
        print(f"  Bounding Box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
        print(f"  Box Size: {int(x2-x1)}x{int(y2-y1)} pixels")

def print_detection_statistics(predictions):
    """
    NEW FUNCTION: Print statistical analysis of detections
    """
    if len(predictions) == 0:
        print("No predictions to analyse.")
        return
    
    print("\n" + "="*60)
    print("DETECTION STATISTICS")
    print("="*60)
    
    # Objectness statistics
    objectness_scores = predictions[:, 4].cpu().numpy()
    print("\nObjectness scores:")
    print(f"  Min: {objectness_scores.min():.4f}")
    print(f"  Max: {objectness_scores.max():.4f}")
    print(f"  Mean: {objectness_scores.mean():.4f}")
    print(f"  Std: {objectness_scores.std():.4f}")
    
    # Class distribution
    best_classes = predictions[:, 5:].argmax(dim=1).cpu().numpy()
    unique_classes, counts = np.unique(best_classes, return_counts=True)
    
    print("\nClass distribution (top predicted classes):")
    for cls, count in sorted(zip(unique_classes, counts), key=lambda x: x[1], reverse=True):
        print(f"  {COCO_CLASSES[cls]:20s}: {count:3d} detections")
    
    # Box size statistics
    boxes = predictions[:, :4].cpu().numpy()
    widths = boxes[:, 2]
    heights = boxes[:, 3]
    
    print("\nBox size statistics (in network coordinates):")
    print(f"  Width:  mean={widths.mean():.1f}, std={widths.std():.1f}, "
          f"range=[{widths.min():.1f}, {widths.max():.1f}]")
    print(f"  Height: mean={heights.mean():.1f}, std={heights.std():.1f}, "
          f"range=[{heights.min():.1f}, {heights.max():.1f}]")

def analyse_predictions_interactive(model, img_tensor, img, conf_thres=0.5):
    """
    Interactive analysis mode - allows detailed inspection of network predictions.
    
    FIXED: Added proper device handling and coordinate transformations
    """
    device = img_tensor.device
    
    with torch.no_grad():
        raw_predictions = model(img_tensor)
    
    print("\n" + "="*80)
    print("INTERACTIVE YOLO ANALYSIS MODE")
    print("="*80)
    
    # Get predictions for first image in batch
    predictions = raw_predictions[0]
    
    # Filter by confidence
    conf_mask = predictions[:, 4] > conf_thres
    filtered_preds = predictions[conf_mask]
    
    print(f"\nTotal predictions: {predictions.size(0)}")
    print(f"Predictions above {conf_thres} confidence: {filtered_preds.size(0)}")
    
    # Show confidence distribution
    all_confidences = predictions[:, 4].cpu().numpy()
    print(f"Confidence range: [{all_confidences.min():.4f}, {all_confidences.max():.4f}]")
    print(f"Mean confidence: {all_confidences.mean():.4f}")
    
    if filtered_preds.size(0) == 0:
        print("\nNo predictions above threshold. Lower the threshold to see more.")
        # Show top 5 predictions regardless of threshold
        top_5_indices = predictions[:, 4].argsort(descending=True)[:5]
        print("\nTop 5 predictions by objectness:")
        for idx in top_5_indices:
            obj = predictions[idx, 4]
            best_class = predictions[idx, 5:].argmax()
            class_conf = predictions[idx, 5 + best_class]
            print(f"  {COCO_CLASSES[int(best_class)]}: obj={obj:.4f}, class={class_conf:.4f}")
        return
    
    # Interactive loop
    while True:
        print("\nOptions:")
        print("  1. Show class probability vector for a detection")
        print("  2. Show top predictions for each class")
        print("  3. Visualise confidence heatmap")
        print("  4. Show raw predictions before NMS")
        print("  5. Show detection statistics")
        print("  q. Quit interactive mode")
        
        choice = input("\nEnter choice: ").strip()
        
        if choice == 'q':
            break
            
        elif choice == '1':
            # Show full class vector for specific detection
            print("\nAvailable detections:")
            for i, pred in enumerate(filtered_preds[:20]):  # Show first 20
                objectness = pred[4]
                class_probs = pred[5:]
                best_class = class_probs.argmax()
                class_conf = class_probs[best_class]
                print(f"  {i}: {COCO_CLASSES[int(best_class)]} "
                      f"(obj: {objectness:.3f}, class: {class_conf:.3f}, "
                      f"combined: {(objectness * class_conf):.3f})")
            
            try:
                det_idx = int(input("\nEnter detection index: "))
                if 0 <= det_idx < filtered_preds.size(0):
                    show_class_probabilities(filtered_preds[det_idx])
                else:
                    print("Invalid index.")
            except:
                print("Invalid input.")
                
        elif choice == '2':
            # Show top predictions for each class
            show_top_predictions_per_class(filtered_preds)
            
        elif choice == '3':
            # Visualise confidence heatmap
            visualise_confidence_heatmap(raw_predictions, img)
            
        elif choice == '4':
            # Show raw predictions
            visualise_raw_predictions(raw_predictions, img)
            
        elif choice == '5':
            # Show detection statistics
            print_detection_statistics(filtered_preds)

def show_class_probabilities(detection, top_k=10):
    """
    Display the full 80-element class probability vector for a detection.
    
    VERIFIED: Correctly shows all 80 classes with proper highlighting
    """
    # Ensure we're working with CPU tensors
    if detection.is_cuda:
        detection = detection.cpu()
    
    class_probs = detection[5:].numpy()  # Classes start at index 5
    objectness = detection[4].numpy()
    
    print(f"\nObjectness Score: {objectness:.4f}")
    print("\nFull Class Probability Vector (80 classes):")
    print("-" * 60)
    
    # Verify we have exactly 80 classes
    assert len(class_probs) == 80, f"Expected 80 classes, got {len(class_probs)}"
    
    # Find top classes
    top_indices = class_probs.argsort()[-top_k:][::-1]
    
    # Print all probabilities with highlighting
    for i in range(80):
        prob = class_probs[i]
        class_name = COCO_CLASSES[i]
        
        # Calculate combined score (what actually matters for detection)
        combined_score = objectness * prob
        
        # Highlight top prediction
        if i == top_indices[0]:
            print(f">>> {i:2d}. {class_name:20s}: {prob:8.6f} (combined: {combined_score:.6f}) <<<  MAX")
        elif i in top_indices:
            print(f"    {i:2d}. {class_name:20s}: {prob:8.6f} (combined: {combined_score:.6f})  *")
        else:
            print(f"    {i:2d}. {class_name:20s}: {prob:8.6f}")
    
    print("\nTop-10 Classes Summary:")
    print("Rank  Class                    Prob      Combined Score")
    print("-" * 55)
    for rank, idx in enumerate(top_indices):
        combined = objectness * class_probs[idx]
        print(f"  {rank+1:2d}. {COCO_CLASSES[idx]:20s}: {class_probs[idx]:.6f}  ({combined:.6f})")


def show_top_predictions_per_class(predictions, top_k=3):
    """
    For each COCO class, show the top-k most confident predictions.
    """
    print("\nTop predictions for each class:")
    print("-" * 80)
    
    # Organise predictions by class
    class_predictions = {i: [] for i in range(80)}
    
    for pred in predictions:
        objectness = pred[4]
        class_probs = pred[5:]
        
        for class_id in range(80):
            score = objectness * class_probs[class_id]
            if score > 0.01:  # Only consider meaningful scores
                class_predictions[class_id].append({
                    'score': score.item(),
                    'objectness': objectness.item(),
                    'class_prob': class_probs[class_id].item(),
                    'bbox': pred[:4].cpu().numpy()
                })
    
    # Show top predictions for each class
    for class_id, preds in class_predictions.items():
        if preds:
            preds_sorted = sorted(preds, key=lambda x: x['score'], reverse=True)[:top_k]
            print(f"\n{COCO_CLASSES[class_id]}:")
            for i, p in enumerate(preds_sorted):
                print(f"  {i+1}. Score: {p['score']:.4f} "
                      f"(obj: {p['objectness']:.3f}, "
                      f"cls: {p['class_prob']:.3f})")

def visualise_confidence_heatmap(predictions, img):
    """
    Create a heatmap showing where the network has high confidence.
    
    FIXED: Proper coordinate transformation and boundary checking
    """
    # Get image dimensions
    h, w = img.shape[:2]
    heatmap = np.zeros((h, w))
    
    # Assuming standard YOLO input size
    img_size = 416
    
    preds = predictions[0]  # First image
    preds_copy = preds.clone()
    preds_copy[..., :4] = box_centre_to_corners(preds_copy[..., :4])
    
    # Count boxes processed
    boxes_processed = 0
    
    for pred in preds_copy:
        if pred[4] > 0.1:  # Objectness threshold
            box_coords = pred[:4]
            
            # Transform back to original image coordinates
            x1, y1, x2, y2 = scale_coords_to_original_image(box_coords, img.shape[:2], img_size)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            if x2 > x1 and y2 > y1:  # Valid box
                # Add confidence to heatmap
                heatmap[y1:y2, x1:x2] += pred[4].cpu().numpy()
                boxes_processed += 1
    
    # Normalise heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Display
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Overlay heatmap on image
    ax2.imshow(img)
    im = ax2.imshow(heatmap, cmap='hot', alpha=0.6)
    ax2.set_title(f'Objectness Confidence Heatmap ({boxes_processed} boxes)')
    ax2.axis('off')
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def print_model_stats(model):
    """
    Print statistics about the model architecture.
    """
    print("\n" + "="*80)
    print("YOLO MODEL STATISTICS")
    print("="*80)
    
    total_params = 0
    yolo_layers = []
    
    for i, (block, module) in enumerate(zip(model.blocks[1:], model.module_list)):
        if block['type'] == 'yolo':
            yolo_layers.append(i)
            anchors = module[0].anchors
            print(f"\nYOLO Layer {len(yolo_layers)} (Layer {i}):")
            print(f"  Anchors: {anchors}")
            print(f"  Stride: {module[0].stride if hasattr(module[0], 'stride') else 'N/A'}")
        
        # Count parameters
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                total_params += sum(p.numel() for p in m.parameters())
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Total YOLO Detection Layers: {len(yolo_layers)}")

# Modified main detection function with interactive mode
def detect_image_interactive(cfg_path, weights_path, img_path, output_path, 
                           conf_thres=0.5, nms_thres=0.4, interactive=False):
    """
    Enhanced detection function with interactive analysis mode.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = Darknet(cfg_path)
    model.load_darknet_weights(weights_path)
    model.eval()
    model = model.to(device)
    
    # Print model statistics
    print_model_stats(model)
    
    # Preprocess image
    img_tensor, original_img = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)
    
    if interactive:
        # Enter interactive analysis mode
        analyse_predictions_interactive(model, img_tensor, original_img, conf_thres)
    
    # Regular detection
    start_time = time.time()
    
    with torch.no_grad():
        raw_detections = model(img_tensor)
        detections = non_max_suppression(raw_detections, conf_thres, nms_thres)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    inference_time = time.time() - start_time
    print(f"\nInference time: {inference_time*1000:.2f} ms")
    
    # Print detailed detection info
    print_detection_details(detections)
    
    # Visualise raw predictions if in verbose mode
    if interactive and len(detections) > 0:
        visualise_raw_predictions(raw_detections, original_img)
    
    # Draw and save results
    if len(detections) > 0:
        result_img = draw_detections(original_img, detections.cpu())
    else:
        result_img = original_img
        print("No objects detected")
    
    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"\nResult saved to {output_path}")
    
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv3 Object Detection with Debug Features')
    parser.add_argument('--cfg', default='yolov3.cfg', help='Path to config file')
    parser.add_argument('--weights', default='yolov3.weights', help='Path to weights file')
    parser.add_argument('--image', default='test_image.jpg', help='Path to input image')
    parser.add_argument('--output', default='detected_image.jpg', help='Path to output image')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive analysis mode')
    
    args = parser.parse_args()
    
    # Run detection with interactive mode if specified
    detections = detect_image_interactive(
        args.cfg, args.weights, args.image, args.output,
        args.conf_thres, args.nms_thres, args.interactive
    )