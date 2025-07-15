import torch
import torch.nn as nn
import numpy as np
import time
import cv2
import os

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, image_size):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.image_size = image_size
        self.grid_size = 0
        self.stride = 0
        
    def forward(self, input_features):
        batch_size = input_features.size(0)
        grid_size = input_features.size(2)
        
        # Reshape the input tensor into a grid of predictions
        raw_prediction = input_features.view(
            batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size
        )
        raw_prediction = raw_prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        # Extract individual prediction components
        center_x = torch.sigmoid(raw_prediction[..., 0])
        center_y = torch.sigmoid(raw_prediction[..., 1])
        width = raw_prediction[..., 2]
        height = raw_prediction[..., 3]
        confidence_score = torch.sigmoid(raw_prediction[..., 4])
        predicted_class_scores = torch.sigmoid(raw_prediction[..., 5:])
        
        # Calculate stride for scaling
        self.stride = self.image_size // grid_size
        
        # Create grid offsets
        grid_x = torch.arange(grid_size, device=input_features.device).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float()
        grid_y = torch.arange(grid_size, device=input_features.device).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).float()
        
        # Scale anchors to the feature map size
        scaled_anchors = [(anchor_width / self.stride, anchor_height / self.stride) for anchor_width, anchor_height in self.anchors]
        anchor_widths = torch.FloatTensor(scaled_anchors).index_select(1, torch.LongTensor([0])).to(input_features.device)
        anchor_heights = torch.FloatTensor(scaled_anchors).index_select(1, torch.LongTensor([1])).to(input_features.device)
        
        anchor_widths = anchor_widths.repeat(batch_size, 1).view(batch_size, self.num_anchors, 1, 1)
        anchor_heights = anchor_heights.repeat(batch_size, 1).view(batch_size, self.num_anchors, 1, 1)
        
        # Calculate final bounding box predictions
        predicted_boxes = torch.zeros_like(raw_prediction[..., :4])
        predicted_boxes[..., 0] = center_x + grid_x
        predicted_boxes[..., 1] = center_y + grid_y
        predicted_boxes[..., 2] = torch.exp(width) * anchor_widths
        predicted_boxes[..., 3] = torch.exp(height) * anchor_heights
        
        # Combine all predictions into a single tensor
        output = torch.cat((
            predicted_boxes.view(batch_size, -1, 4) * self.stride,
            confidence_score.view(batch_size, -1, 1),
            predicted_class_scores.view(batch_size, -1, self.num_classes)
        ), -1)
        
        return output

#---------------------------------------------------------------------------------------------------

class Darknet(nn.Module):
    def __init__(self, config_path, image_size=416):
        super(Darknet, self).__init__()
        self.module_definitions = self.parse_config(config_path)
        self.image_size = image_size
        self.module_list = self.create_network_modules(self.module_definitions)
        
    def parse_config(self, config_path):
        with open(config_path, 'r') as config_file:
            lines = config_file.read().split('\n')
        lines = [line for line in lines if line and not line.startswith('#')]
        lines = [line.strip() for line in lines]
        
        module_definitions = []
        for line in lines:
            if line.startswith('['):
                if module_definitions:
                    module_definitions.append(current_block)
                current_block = {'type': line[1:-1]}
            else:
                key, value = line.split('=')
                current_block[key.strip()] = value.strip()
        module_definitions.append(current_block)
        
        return module_definitions
    
    def create_network_modules(self, module_definitions):
        network_info = module_definitions[0]
        module_list = nn.ModuleList()
        previous_filters = 3  # Initial channels for RGB image
        output_filters_history = []
        
        for index, block_def in enumerate(module_definitions[1:]):
            module = nn.Sequential()
            
            if block_def['type'] == 'convolutional':
                filters = int(block_def['filters'])
                kernel_size = int(block_def['size'])
                stride = int(block_def['stride'])
                padding = (kernel_size - 1) // 2 if int(block_def.get('pad', 0)) else 0
                has_bias = 'batch_normalize' not in block_def
                
                conv_layer = nn.Conv2d(previous_filters, filters, kernel_size, stride, padding, bias=has_bias)
                module.add_module(f'conv_{index}', conv_layer)
                
                if 'batch_normalize' in block_def:
                    bn_layer = nn.BatchNorm2d(filters)
                    module.add_module(f'batch_norm_{index}', bn_layer)
                
                if block_def['activation'] == 'leaky':
                    activation_layer = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module(f'leaky_{index}', activation_layer)
                    
            elif block_def['type'] == 'upsample':
                upsample_layer = nn.Upsample(scale_factor=int(block_def['stride']), mode='nearest')
                module.add_module(f'upsample_{index}', upsample_layer)
                
            elif block_def['type'] == 'route':
                layer_indices = [int(x) for x in block_def['layers'].split(',')]
                filters = sum(output_filters_history[i] for i in layer_indices)
                module.add_module(f'route_{index}', nn.Identity())
                
            elif block_def['type'] == 'shortcut':
                module.add_module(f'shortcut_{index}', nn.Identity())
                
            elif block_def['type'] == 'yolo':
                mask_indices = [int(x) for x in block_def['mask'].split(',')]
                anchor_coords = [int(x) for x in block_def['anchors'].split(',')]
                anchors = [(anchor_coords[i], anchor_coords[i+1]) for i in range(0, len(anchor_coords), 2)]
                masked_anchors = [anchors[i] for i in mask_indices]
                
                num_classes = int(block_def['classes'])
                image_size = int(network_info['height'])
                
                yolo_layer = YOLOLayer(masked_anchors, num_classes, image_size)
                module.add_module(f'yolo_{index}', yolo_layer)
                
            module_list.append(module)
            previous_filters = filters
            output_filters_history.append(filters)
            
        return module_list
    
    def forward(self, input_tensor):
        yolo_outputs = []
        layer_outputs = []
        
        for index, (block_def, module) in enumerate(zip(self.module_definitions[1:], self.module_list)):
            if block_def['type'] in ['convolutional', 'upsample']:
                input_tensor = module(input_tensor)
                
            elif block_def['type'] == 'route':
                layer_indices = [int(x) for x in block_def['layers'].split(',')]
                input_tensor = torch.cat([layer_outputs[i] for i in layer_indices], 1)
                
            elif block_def['type'] == 'shortcut':
                from_layer_index = int(block_def['from'])
                input_tensor = layer_outputs[-1] + layer_outputs[from_layer_index]
                
            elif block_def['type'] == 'yolo':
                yolo_output = module[0](input_tensor)
                yolo_outputs.append(yolo_output)
                
            layer_outputs.append(input_tensor)
            
        return torch.cat(yolo_outputs, 1)
    
    def load_darknet_weights(self, weights_path):
        with open(weights_path, 'rb') as weights_file:
            # First 5 values are header info
            _ = np.fromfile(weights_file, dtype=np.int32, count=5)
            weights = np.fromfile(weights_file, dtype=np.float32)
            
        print(f"Loading weights from {weights_path}")
        weights_pointer = 0
        
        for index, (block_def, module) in enumerate(zip(self.module_definitions[1:], self.module_list)):
            if block_def['type'] == 'convolutional':
                conv_layer = module[0]
                if 'batch_normalize' in block_def:
                    bn_layer = module[1]
                    num_bn_params = bn_layer.bias.numel()
                    
                    # Load BN bias, weights, running mean, and running var
                    bn_biases = torch.from_numpy(weights[weights_pointer : weights_pointer + num_bn_params])
                    weights_pointer += num_bn_params
                    bn_weights = torch.from_numpy(weights[weights_pointer : weights_pointer + num_bn_params])
                    weights_pointer += num_bn_params
                    bn_running_mean = torch.from_numpy(weights[weights_pointer : weights_pointer + num_bn_params])
                    weights_pointer += num_bn_params
                    bn_running_var = torch.from_numpy(weights[weights_pointer : weights_pointer + num_bn_params])
                    weights_pointer += num_bn_params
                    
                    # Copy loaded params to the BN layer
                    bn_layer.bias.data.copy_(bn_biases.view_as(bn_layer.bias.data))
                    bn_layer.weight.data.copy_(bn_weights.view_as(bn_layer.weight.data))
                    bn_layer.running_mean.copy_(bn_running_mean.view_as(bn_layer.running_mean))
                    bn_layer.running_var.copy_(bn_running_var.view_as(bn_layer.running_var))
                else:
                    # Load conv bias
                    num_conv_biases = conv_layer.bias.numel()
                    conv_biases = torch.from_numpy(weights[weights_pointer : weights_pointer + num_conv_biases])
                    weights_pointer += num_conv_biases
                    conv_layer.bias.data.copy_(conv_biases.view_as(conv_layer.bias.data))
                
                # Load conv weights
                num_conv_weights = conv_layer.weight.numel()
                conv_weights = torch.from_numpy(weights[weights_pointer : weights_pointer + num_conv_weights])
                weights_pointer += num_conv_weights
                conv_layer.weight.data.copy_(conv_weights.view_as(conv_layer.weight.data))
                
        print(f"Loaded weights: {weights_pointer} / {len(weights)} values used")

#---------------------------------------------------------------------------------------------------

def preprocess_image(image_path, target_size=416):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_height, original_width = image_rgb.shape[:2]
    scale = min(target_size / original_width, target_size / original_height)
    new_width, new_height = int(original_width * scale), int(original_height * scale)
    
    resized_image = cv2.resize(image_rgb, (new_width, new_height))
    
    # Create a new image with padding
    padded_image = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    width_padding = (target_size - new_width) // 2
    height_padding = (target_size - new_height) // 2
    padded_image[height_padding:height_padding + new_height, width_padding:width_padding + new_width] = resized_image
    
    # Convert to a PyTorch tensor
    image_tensor = torch.from_numpy(padded_image).float().div(255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image_rgb

#---------------------------------------------------------------------------------------------------

def non_max_suppression(predictions, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Performs Non-Maximum Suppression (NMS) on inference results.
    Returns detections with shape: (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # Convert bounding box from (center_x, center_y, width, height) to (x1, y1, x2, y2)
    predictions[..., :4] = convert_box_format_xywh_to_xyxy(predictions[..., :4])
    
    final_output = [None] * predictions.size(0)
    
    for image_index, image_predictions in enumerate(predictions):
        # Filter out low-confidence detections
        confidence_mask = image_predictions[:, 4] >= confidence_threshold
        image_predictions = image_predictions[confidence_mask]
        
        if not image_predictions.size(0):
            continue
            
        # Combine object confidence with class confidence
        class_confidences, class_predictions = image_predictions[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_predictions[:, :5], class_confidences.float(), class_predictions.float()), 1)
        
        # Sort detections by confidence score
        detections = detections[detections[:, 4].argsort(descending=True)]
        
        # Perform NMS
        final_detections = []
        while detections.size(0):
            best_detection = detections[0]
            final_detections.append(best_detection)
            
            iou = calculate_bbox_iou(best_detection.unsqueeze(0), detections)
            
            # Keep detections with low IoU or different class labels
            nms_mask = (iou < nms_threshold) | (detections[:, -1] != best_detection[-1])
            detections = detections[nms_mask]
            
        if final_detections:
            final_output[image_index] = torch.stack(final_detections)
            
    return final_output

#---------------------------------------------------------------------------------------------------

def convert_box_format_xywh_to_xyxy(box_xywh):
    """Converts bounding box from [center_x, center_y, width, height] to [x1, y1, x2, y2]."""
    box_xyxy = box_xywh.new(box_xywh.shape)
    box_xyxy[..., 0] = box_xywh[..., 0] - box_xywh[..., 2] / 2
    box_xyxy[..., 1] = box_xywh[..., 1] - box_xywh[..., 3] / 2
    box_xyxy[..., 2] = box_xywh[..., 0] + box_xywh[..., 2] / 2
    box_xyxy[..., 3] = box_xywh[..., 1] + box_xywh[..., 3] / 2
    return box_xyxy

#---------------------------------------------------------------------------------------------------

def calculate_bbox_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two sets of bounding boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Calculate intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    intersection_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union area
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = box1_area + box2_area - intersection_area + 1e-16
    
    return intersection_area / union_area

#---------------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------------

def draw_detections(image, detections, input_image_size=416):
    # Scale detection coordinates back to the original image size
    original_height, original_width = image.shape[:2]
    scale = min(input_image_size / original_width, input_image_size / original_height)
    width_padding = (input_image_size - int(original_width * scale)) // 2
    height_padding = (input_image_size - int(original_height * scale)) // 2
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0) # Red in RGB
    
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        # Rescale coordinates from padded image to original image
        x1 = int((x1 - width_padding) / scale)
        y1 = int((y1 - height_padding) / scale)
        x2 = int((x2 - width_padding) / scale)
        y2 = int((y2 - height_padding) / scale)
        
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Create and draw the label
        label = f'{COCO_CLASSES[int(cls_pred)]}: {conf:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), font, 0.5, color, 2)
        
    return image

#---------------------------------------------------------------------------------------------------

# Main detection function
def run_detection(config_path, weights_path, image_path, output_path, confidence_threshold=0.5, nms_threshold=0.4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = Darknet(config_path)
    model.load_darknet_weights(weights_path)
    model.eval()
    model.to(device)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        raw_detections = model(image_tensor)
        final_detections = non_max_suppression(raw_detections, confidence_threshold, nms_threshold)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time*1000:.2f} ms")
    
    # Draw detections on the image
    result_image = original_image.copy() # Avoid modifying the original image array
    if final_detections[0] is not None:
        detections_on_cpu = final_detections[0].cpu()
        result_image = draw_detections(result_image, detections_on_cpu)
    else:
        print("No objects detected.")
    
    # Convert from RGB (used by Pillow/Matplotlib) to BGR (used by OpenCV)
    result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"Result saved to {output_path}")
    
    return final_detections

#---------------------------------------------------------------------------------------------------

# Example usage
if __name__ == "__main__":
    # Define paths
    config_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    image_path = "test_image.jpg"
    output_path = "detected_image.jpg"
    
    # Run the detection
    all_detections = run_detection(config_path, weights_path, image_path, output_path)
    
    # Print detected objects for the first image
    image_detections = all_detections[0]
    if image_detections is not None:
        print(f"\nDetected {len(image_detections)} objects:")
        for detection in image_detections:
            # Unpack detection tensor: x1, y1, x2, y2, obj_conf, class_score, class_pred
            confidence = detection[4]
            class_index = int(detection[6])
            print(f"- {COCO_CLASSES[class_index]}: Confidence {confidence:.2f}")
