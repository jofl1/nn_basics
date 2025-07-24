import torch
import torch.nn as nn
import numpy as np

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_size = 0
        self.stride = 0
        # Add this flag to control output format
        self.training = True
       
    def forward(self, x, targets=None):
        batch_size = x.size(0)
        grid_size = x.size(2)
       
        # Reshape predictions
        prediction = x.view(batch_size, self.num_anchors,
                          self.num_classes + 5, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
       
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
       
        # If we're training, return the raw predictions for loss calculation
        if self.training:
            return prediction
       
        # Otherwise, convert to bounding boxes for inference
        stride = self.img_size // grid_size
       
        # Create grids
        grid_x = torch.arange(grid_size, dtype=torch.float32, device=x.device).repeat(grid_size, 1).view([1, 1, grid_size, grid_size])
        grid_y = torch.arange(grid_size, dtype=torch.float32, device=x.device).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])
       
        # Scale anchors
        scaled_anchors = [(a[0]/stride, a[1]/stride) for a in self.anchors]
        anchor_w = torch.tensor([a[0] for a in scaled_anchors], dtype=torch.float32, device=x.device)
        anchor_h = torch.tensor([a[1] for a in scaled_anchors], dtype=torch.float32, device=x.device)
        anchor_w = anchor_w.repeat(batch_size, 1).view(batch_size, self.num_anchors, 1, 1)
        anchor_h = anchor_h.repeat(batch_size, 1).view(batch_size, self.num_anchors, 1, 1)
       
        # Convert predictions to bounding boxes
        pred_boxes = torch.zeros_like(prediction[..., :4])
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
       
        # Flatten and scale
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * stride,
                           conf.view(batch_size, -1, 1),
                           pred_cls.view(batch_size, -1, self.num_classes)), -1)
       
        return output

class Darknet(nn.Module):
    """
    Darknet neural network architecture for YOLO object detection.
    Parses configuration file and builds the network dynamically.
    """
    def __init__(self, cfg_path, num_classes=80, img_size=416):
        super(Darknet, self).__init__()
        self.blocks = self.parse_cfg(cfg_path)  # Parse network architecture from config
        self.img_size = img_size
        self.num_classes = num_classes
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
               
                num_classes = self.num_classes
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
            List of detection tensors from all YOLO layers (typically 3 for YOLOv3)
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
                x = module[0](x)  # module is nn.Sequential object that holds layers for current block
                outputs.append(x)  # Formatted bounding box predictions at specific scale
            
            layer_outputs.append(x)  # Save output for potential route/shortcut layers
        
        # Return list of YOLO outputs instead of concatenating them
        return outputs  # This will return a list with 3 tensors for YOLOv3
   
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
                   
                    
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases
                    conv_layer.bias.data.copy_(conv_biases.view_as(conv_layer.bias.data))
               
                # Load convolutional weights (same for both cases)
                num_weights = conv_layer.weight.numel()
               
                
                # Load and reshape weights to match layer dimensions
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr += num_weights
                conv_layer.weight.data.copy_(conv_weights.view_as(conv_layer.weight.data))
       
        print(f"Loaded weights: {ptr} / {len(weights)} values used")
    
    def save_darknet_weights(self, weights_path):
        """
        Save model weights in Darknet format.
        
        Args:
            weights_path: Path to save .weights file
        """
        with open(weights_path, 'wb') as f:
            # Write header (5 int32 values)
            # Major version, minor version, revision, seen (images), 0
            header = np.array([0, 2, 0, 0, 0], dtype=np.int32)
            header.tofile(f)
            
            # Save weights layer by layer
            for i, (block, module) in enumerate(zip(self.blocks[1:], self.module_list)):
                if block['type'] == 'convolutional':
                    conv_layer = module[0]
                    if 'batch_normalize' in block:
                        # Save batch norm parameters first
                        bn_layer = module[1]
                        bn_layer.bias.data.cpu().numpy().tofile(f)
                        bn_layer.weight.data.cpu().numpy().tofile(f)
                        bn_layer.running_mean.cpu().numpy().tofile(f)
                        bn_layer.running_var.cpu().numpy().tofile(f)
                    else:
                        # Save conv bias
                        conv_layer.bias.data.cpu().numpy().tofile(f)
                    
                    # Save conv weights
                    conv_layer.weight.data.cpu().numpy().tofile(f)
        
        print(f"Saved weights to {weights_path}")
