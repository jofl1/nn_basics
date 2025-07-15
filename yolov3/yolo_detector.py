import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import os

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_size = 0
        self.stride = 0
       
    def forward(self, x):
        batch_size = x.size(0)
        grid_size = x.size(2)
       
        # Reshape predictions
        prediction = x.view(batch_size, self.num_anchors,
                          self.num_classes + 5, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
       
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Confidence
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class predictions
       
        # Calculate stride
        stride = self.img_size // grid_size
       
        grid_x = torch.arange(grid_size, device=x.device).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float()
        grid_y = torch.arange(grid_size, device=x.device).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).float()
       
        # Calculate anchor boxes
        scaled_anchors = [(a[0]/stride, a[1]/stride) for a in self.anchors]
        anchor_w = torch.FloatTensor(scaled_anchors).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.FloatTensor(scaled_anchors).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).view(batch_size, self.num_anchors, 1, 1).to(x.device)
        anchor_h = anchor_h.repeat(batch_size, 1).view(batch_size, self.num_anchors, 1, 1).to(x.device)
       
        pred_boxes = torch.zeros_like(prediction[..., :4])
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
       
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * stride,
                           conf.view(batch_size, -1, 1),
                           pred_cls.view(batch_size, -1, self.num_classes)), -1)
       
        return output

class Darknet(nn.Module):
    def __init__(self, cfg_path, img_size=416):
        super(Darknet, self).__init__()
        self.blocks = self.parse_cfg(cfg_path)
        self.img_size = img_size
        self.module_list = self.create_modules(self.blocks)
       
    def parse_cfg(self, cfg_path):
        with open(cfg_path, 'r') as f:
            lines = f.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]
        lines = [x.strip() for x in lines]
       
        blocks = []
        block = {}
       
        for line in lines:
            if line.startswith('['):
                if block:
                    blocks.append(block)
                block = {}
                block['type'] = line[1:-1]
            else:
                key, value = line.split('=')
                block[key.strip()] = value.strip()
        blocks.append(block)
       
        return blocks
   
    def create_modules(self, blocks):
        net_info = blocks[0]
        module_list = nn.ModuleList()
        prev_filters = 3
        output_filters = []
       
        for idx, block in enumerate(blocks[1:]):
            module = nn.Sequential()
           
            if block['type'] == 'convolutional':
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                pad = (kernel_size - 1) // 2 if block.get('pad') else 0
               
                # Conv layers have bias only when there's no batch norm
                has_bias = 'batch_normalize' not in block
                conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=has_bias)
                module.add_module(f'conv_{idx}', conv)
               
                if 'batch_normalize' in block:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module(f'batch_norm_{idx}', bn)
               
                if block['activation'] == 'leaky':
                    activn = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module(f'leaky_{idx}', activn)
                   
            elif block['type'] == 'upsample':
                upsample = nn.Upsample(scale_factor=int(block['stride']), mode='nearest')
                module.add_module(f'upsample_{idx}', upsample)
               
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(x) for x in layers]
               
                if len(layers) == 1:
                    filters = output_filters[layers[0]]
                else:
                    filters = sum([output_filters[l] for l in layers])
               
                module.add_module(f'route_{idx}', nn.Identity())
               
            elif block['type'] == 'shortcut':
                module.add_module(f'shortcut_{idx}', nn.Identity())
               
            elif block['type'] == 'yolo':
                mask = block['mask'].split(',')
                mask = [int(x) for x in mask]
               
                anchors = block['anchors'].split(',')
                anchors = [(int(anchors[i]), int(anchors[i+1]))
                          for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
               
                num_classes = int(block['classes'])
                img_size = int(net_info['height'])
               
                yolo = YOLOLayer(anchors, num_classes, img_size)
                module.add_module(f'yolo_{idx}', yolo)
               
            module_list.append(module)
            output_filters.append(filters if block['type'] != 'yolo' else prev_filters)
            prev_filters = filters if block['type'] != 'yolo' else prev_filters
           
        return module_list
   
    def forward(self, x):
        outputs = []
        layer_outputs = []
       
        for i, (block, module) in enumerate(zip(self.blocks[1:], self.module_list)):
            if block['type'] in ['convolutional', 'upsample']:
                x = module(x)
               
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(x) for x in layers]
               
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    x = torch.cat([layer_outputs[l] for l in layers], 1)
                   
            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                x = layer_outputs[-1] + layer_outputs[from_layer]
               
            elif block['type'] == 'yolo':
                x = module[0](x)
                outputs.append(x)
               
            layer_outputs.append(x)
           
        return torch.cat(outputs, 1)
   
    def load_darknet_weights(self, weights_path):
        with open(weights_path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
           
        print(f"Loading weights from {weights_path}")
        print(f"Total weights in file: {len(weights)}")
           
        ptr = 0
        for i, (block, module) in enumerate(zip(self.blocks[1:], self.module_list)):
            if block['type'] == 'convolutional':
                conv_layer = module[0]
                if 'batch_normalize' in block:
                    bn_layer = module[1]
                   
                    # Load BN bias, weights, running mean and var
                    num_bn_biases = bn_layer.bias.numel()
                   
                    # Check if enough weights
                    if ptr + num_bn_biases > len(weights):
                        raise RuntimeError(f"Not enough weights for BN bias at layer {i}")
                   
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                   
                    bn_weights = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                   
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                   
                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                   
                    bn_layer.bias.data.copy_(bn_biases.view_as(bn_layer.bias.data))
                    bn_layer.weight.data.copy_(bn_weights.view_as(bn_layer.weight.data))
                    bn_layer.running_mean.copy_(bn_running_mean.view_as(bn_layer.running_mean))
                    bn_layer.running_var.copy_(bn_running_var.view_as(bn_layer.running_var))
                else:
                    # Load conv bias (only present when there's no batch norm)
                    num_biases = conv_layer.bias.numel()
                   
                    # Check if we have enough weights
                    if ptr + num_biases > len(weights):
                        raise RuntimeError(f"Not enough weights for conv bias at layer {i}")
                   
                    conv_biases = torch.from_numpy(weights[ptr:ptr + num_biases])
                    ptr += num_biases
                    conv_layer.bias.data.copy_(conv_biases.view_as(conv_layer.bias.data))
               
                # Load conv weights
                num_weights = conv_layer.weight.numel()
               
                # Check if we have enough weights
                if ptr + num_weights > len(weights):
                    raise RuntimeError(f"Not enough weights for conv weights at layer {i}. Need {num_weights}, have {len(weights) - ptr}")
               
                try:
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
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   
    h, w = img.shape[:2]
    scale = min(img_size/w, img_size/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
   
    img_resized = cv2.resize(img, (new_w, new_h))
   
    # Create blank image and paste resized image
    img_padded = np.full((img_size, img_size, 3), 128, dtype=np.uint8)
    dw = (img_size - new_w) // 2
    dh = (img_size - new_h) // 2
    img_padded[dh:dh+new_h, dw:dw+new_w] = img_resized
   
    # Convert to tensor
    img_tensor = torch.from_numpy(img_padded).float().div(255.0)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
   
    return img_tensor, img

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
   
    # Get batch size
    batch_size = prediction.size(0)
   
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
   
    output = []
   
    for image_i in range(batch_size):
        image_pred = prediction[image_i]  # Get predictions for this image
       
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres)
        image_pred = image_pred[conf_mask]
       
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
           
        # Object confidence times class confidence
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
       
        # Concatenate
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
       
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            # Get detection with highest score
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
           
            # Indices of boxes with high IoU and matching label
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
           
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
           
        if keep_boxes:
            output.extend(keep_boxes)
           
    return torch.stack(output) if output else torch.FloatTensor(0, 7)

def xywh2xyxy(x):
    """Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]"""
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def bbox_iou(box1, box2):
    # Get coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
   
    # Intersection area
    inter_area = torch.clamp(torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1), min=0) * \
                 torch.clamp(torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1), min=0)
   
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + 1e-16
   
    return inter_area / union_area

# COCO class names
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
    
    # Scale detections back to original image size
    h, w = img.shape[:2]
    scale = min(img_size / w, img_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    dw = (img_size - new_w) // 2
    dh = (img_size - new_h) // 2
    
    # Define font and color for OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)  # Red in RGB
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_conf, cls = det
        
        # Adjust coordinates from padded to original
        x1 = int((x1 - dw) / scale)
        y1 = int((y1 - dh) / scale)
        x2 = int((x2 - dw) / scale)
        y2 = int((y2 - dh) / scale)
        
        # Draw box with cv2.rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Create label and draw with cv2.putText
        label = f'{COCO_CLASSES[int(cls)]}: {conf:.2f}'
        cv2.putText(img, label, (x1, y1 - 10), font, 0.5, color, 2)
        
    # The 'img' array is modified in-place, so just return it
    return img

# Main detection function
def detect_image(cfg_path, weights_path, img_path, output_path, conf_thres=0.5, nms_thres=0.4):
    # Check CUDA availability
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
   
    # Load model
    model = Darknet(cfg_path)
    model.load_darknet_weights(weights_path)
    model.eval()
    model = model.to(device)  # Move model to GPU
   
    # Preprocess image
    img_tensor, original_img = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)  # Move input to GPU
   
    if device.type == 'cuda':
        for _ in range(3):
            _ = model(img_tensor)
        torch.cuda.synchronize()
   
    # Run inference with timing
    start_time = time.time()
   
    with torch.no_grad():
        detections = model(img_tensor)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
   
    if device.type == 'cuda':
        torch.cuda.synchronize()
   
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time*1000:.2f} ms")
   
 # Draw detections
    if len(detections) > 0:
        # The detections are drawn on the 'original_img' (which is in RGB format)
        result_img = draw_detections(original_img, detections.cpu()) # Ensure detections are on CPU
    else:
        result_img = original_img
        print("No objects detected")
    
    # Convert result from RGB back to BGR for OpenCV saving
    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    
    # Save result using cv2.imwrite
    cv2.imwrite(output_path, result_bgr)
    print(f"Result saved to {output_path}")
    
    return detections

if __name__ == "__main__":
   
    # Paths
    cfg_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    img_path = "test_image.jpg"  # Your input image
    output_path = "detected_image.jpg"  # Output path
   
    # Run detection
    detections = detect_image(cfg_path, weights_path, img_path, output_path)
   
    # Print detections
    if len(detections) > 0:
        print(f"\nDetected {len(detections)} objects:")
        for det in detections:
            cls = int(det[6])
            conf = det[4]
            print(f"- {COCO_CLASSES[cls]}: {conf:.2f}")
