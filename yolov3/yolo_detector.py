import torch
import torch.nn as nn
import torchvision.ops as ops  # Added for optimised NMS
import numpy as np
import time
import cv2
import os

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

def bbox_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between sets of bounding boxes.
    Used internally by NMS to determine box overlap.
    
    Args:
        box1: Tensor of shape [N, 4] in corner format [x1, y1, x2, y2]
        box2: Tensor of shape [M, 4] in corner format [x1, y1, x2, y2]
        
    Returns:
        IoU matrix of shape [N, M] where element (i,j) is IoU between box1[i] and box2[j]
    """
    # Extract coordinates for all boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
   
    # Calculate intersection area
    # Find the coordinates of the intersection rectangle
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)  # Broadcasting for pairwise comparison
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    
    # Calculate intersection area (clamp ensures non-negative)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)
   
    # Calculate union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)  # Area of boxes in set 1
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)  # Area of boxes in set 2
    # Union = Area1 + Area2 - Intersection 
    union_area = b1_area.unsqueeze(1) + b2_area - inter_area + 1e-16
   
    return inter_area / union_area

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
        label = f'{COCO_CLASSES[int(cls)]}: {conf:.2f}'
        # Draw label above the bounding box
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
        
    # Image is modified in-place
    return img

# Main detection function
def detect_image(cfg_path, weights_path, img_path, output_path, conf_thres=0.5, nms_thres=0.4):
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
    model = Darknet(cfg_path)
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
        detections = non_max_suppression(detections, conf_thres, nms_thres)  # Apply NMS
   
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Ensure GPU operations are complete for accurate timing
   
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time*1000:.2f} ms")
   
    # Draw detections on original image
    if len(detections) > 0:
        # Draw_detections modifies the image in-place
        # Move detections to CPU for drawing (OpenCV uses CPU)
        result_img = draw_detections(original_img, detections.cpu())
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
   
    # File paths
    cfg_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    img_path = "test_image.jpg"  
    output_path = "detected_image.jpg"  
   
    # Run detection
    detections = detect_image(cfg_path, weights_path, img_path, output_path)
   
    # Print detection summary
    if len(detections) > 0:
        print(f"\nDetected {len(detections)} objects:")
        for det in detections:
            cls = int(det[6])  # Class ID
            conf = det[4]      # Objectness confidence
            print(f"- {COCO_CLASSES[cls]}: {conf:.2f}")