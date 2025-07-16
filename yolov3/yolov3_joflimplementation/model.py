

import torch
import torch.nn as nn
import numpy as np

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
