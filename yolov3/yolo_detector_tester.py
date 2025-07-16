import torch
import torch.nn as nn
import torchvision.ops as ops  # Added for optimised NMS
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NetworkVisualizer:
    """
    Comprehensive visualization toolkit for peering into YOLO's internal workings.
    Captures and visualises feature maps, statistics, and detection pipeline.
    """
    def __init__(self):
        self.layer_outputs = {}  # Store outputs from each layer
        self.layer_stats = defaultdict(dict)  # Store statistics for each layer
        self.detection_stages = {}  # Store intermediate detection results
        self.hooks = []  # Store hook handles for cleanup
        
    def clear(self):
        """Clear all stored data and remove hooks."""
        self.layer_outputs.clear()
        self.layer_stats.clear()
        self.detection_stages.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def register_hooks(self, model):
        """
        Register forward hooks on all layers to capture outputs and statistics.
        
        Args:
            model: Darknet model instance
        """
        def make_hook(name, layer_idx):
            def hook(module, input, output):
                # Store raw output
                self.layer_outputs[f'{name}_{layer_idx}'] = output.detach().cpu()
                
                # Calculate statistics
                if isinstance(output, torch.Tensor) and output.dim() >= 2:
                    with torch.no_grad():
                        # Basic statistics
                        self.layer_stats[f'{name}_{layer_idx}']['mean'] = output.mean().item()
                        self.layer_stats[f'{name}_{layer_idx}']['std'] = output.std().item()
                        self.layer_stats[f'{name}_{layer_idx}']['min'] = output.min().item()
                        self.layer_stats[f'{name}_{layer_idx}']['max'] = output.max().item()
                        
                        # Dead neurons (percentage of zeros)
                        zeros = (output == 0).float().mean().item() * 100
                        self.layer_stats[f'{name}_{layer_idx}']['dead_neurons_%'] = zeros
                        
                        # Shape information
                        self.layer_stats[f'{name}_{layer_idx}']['shape'] = list(output.shape)
            return hook
        
        # Register hooks for all convolutional and other interesting layers
        for idx, (block, module) in enumerate(zip(model.blocks[1:], model.module_list)):
            if block['type'] == 'convolutional':
                # Hook the entire sequential module to get final conv output
                handle = module.register_forward_hook(make_hook('conv', idx))
                self.hooks.append(handle)
            elif block['type'] == 'yolo':
                handle = module.register_forward_hook(make_hook('yolo', idx))
                self.hooks.append(handle)
    
    def visualize_feature_maps(self, layer_name, max_channels=64, save_path=None):
        """
        Visualise feature maps from a specific layer as a grid of images.
        
        Args:
            layer_name: Name of the layer (e.g., 'conv_0', 'conv_5')
            max_channels: Maximum number of channels to display
            save_path: Optional path to save the visualization
        """
        if layer_name not in self.layer_outputs:
            print(f"Layer {layer_name} not found. Available layers: {list(self.layer_outputs.keys())}")
            return
            
        features = self.layer_outputs[layer_name]
        
        # Handle different tensor dimensions
        if features.dim() == 4:  # [batch, channels, height, width]
            features = features[0]  # Take first batch item
        
        num_channels = min(features.shape[0], max_channels)
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_channels)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()
        
        # Plot each channel
        for i in range(num_channels):
            ax = axes[i]
            feature_map = features[i].numpy()
            
            # Normalise to [0, 1] for better visualization
            if feature_map.max() != feature_map.min():
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
            
            im = ax.imshow(feature_map, cmap='viridis')
            ax.set_title(f'Channel {i}', fontsize=8)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_channels, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Feature Maps: {layer_name} (showing {num_channels}/{features.shape[0]} channels)', 
                     fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature maps saved to {save_path}")
        plt.show()
        
    def visualize_layer_statistics(self, save_path=None):
        """
        Visualise statistics across all layers to understand network health.
        Shows activation distributions, dead neurons, and parameter magnitudes.
        """
        if not self.layer_stats:
            print("No layer statistics available. Run inference first.")
            return
            
        # Prepare data for plotting
        layer_names = list(self.layer_stats.keys())
        
        # Sort layers by their index for better visualization
        layer_names.sort(key=lambda x: int(x.split('_')[-1]))
        
        means = [self.layer_stats[name]['mean'] for name in layer_names]
        stds = [self.layer_stats[name]['std'] for name in layer_names]
        dead_neurons = [self.layer_stats[name]['dead_neurons_%'] for name in layer_names]
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Mean and standard deviation of activations
        ax1 = axes[0]
        x = np.arange(len(layer_names))
        ax1.bar(x - 0.2, means, 0.4, label='Mean', alpha=0.8)
        ax1.bar(x + 0.2, stds, 0.4, label='Std Dev', alpha=0.8)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Value')
        ax1.set_title('Layer Activation Statistics (Mean and Std Dev)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Percentage of dead neurons
        ax2 = axes[1]
        bars = ax2.bar(x, dead_neurons, alpha=0.8)
        # Colour bars based on severity
        for i, bar in enumerate(bars):
            if dead_neurons[i] > 50:
                bar.set_color('red')
            elif dead_neurons[i] > 20:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Dead Neurons per Layer (% of zero activations)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Activation ranges (min/max)
        ax3 = axes[2]
        mins = [self.layer_stats[name]['min'] for name in layer_names]
        maxs = [self.layer_stats[name]['max'] for name in layer_names]
        ax3.fill_between(x, mins, maxs, alpha=0.3, label='Activation Range')
        ax3.plot(x, mins, 'b-', label='Min', alpha=0.8)
        ax3.plot(x, maxs, 'r-', label='Max', alpha=0.8)
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Activation Value')
        ax3.set_title('Activation Value Ranges per Layer')
        ax3.set_xticks(x)
        ax3.set_xticklabels(layer_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Layer statistics saved to {save_path}")
        plt.show()
        
    def visualize_activation_histograms(self, layer_names=None, save_path=None):
        """
        Visualise activation value distributions as histograms for specified layers.
        Helps identify issues like vanishing/exploding gradients or dead ReLU.
        
        Args:
            layer_names: List of layer names to visualize (None = show first 6)
            save_path: Optional path to save the visualization
        """
        if not self.layer_outputs:
            print("No layer outputs available. Run inference first.")
            return
            
        if layer_names is None:
            # Default to first 6 layers
            layer_names = list(self.layer_outputs.keys())[:6]
        
        num_layers = len(layer_names)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, layer_name in enumerate(layer_names):
            if idx >= 6:  # Maximum 6 subplots
                break
                
            if layer_name not in self.layer_outputs:
                continue
                
            ax = axes[idx]
            activations = self.layer_outputs[layer_name].flatten().numpy()
            
            # Create histogram
            ax.hist(activations, bins=50, alpha=0.7, density=True)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero')
            ax.axvline(x=activations.mean(), color='green', linestyle='--', 
                      alpha=0.5, label=f'Mean: {activations.mean():.3f}')
            
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{layer_name}\n(σ={activations.std():.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(layer_names), 6):
            axes[idx].set_visible(False)
            
        plt.suptitle('Activation Value Distributions', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Activation histograms saved to {save_path}")
    def visualize_activation_histograms(self, layer_names=None, save_path=None):
        """
        Visualise activation value distributions as histograms for specified layers.
        Helps identify issues like vanishing/exploding gradients or dead ReLU.
        
        Args:
            layer_names: List of layer names to visualize (None = show first 6)
            save_path: Optional path to save the visualization
        """
        if not self.layer_outputs:
            print("No layer outputs available. Run inference first.")
            return
            
        if layer_names is None:
            # Default to first 6 layers
            layer_names = list(self.layer_outputs.keys())[:6]
        
        num_layers = len(layer_names)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, layer_name in enumerate(layer_names):
            if idx >= 6:  # Maximum 6 subplots
                break
                
            if layer_name not in self.layer_outputs:
                continue
                
            ax = axes[idx]
            activations = self.layer_outputs[layer_name].flatten().numpy()
            
            # Create histogram
            ax.hist(activations, bins=50, alpha=0.7, density=True)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero')
            ax.axvline(x=activations.mean(), color='green', linestyle='--', 
                      alpha=0.5, label=f'Mean: {activations.mean():.3f}')
            
            ax.set_xlabel('Activation Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{layer_name}\n(σ={activations.std():.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(layer_names), 6):
            axes[idx].set_visible(False)
            
        plt.suptitle('Activation Value Distributions', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Activation histograms saved to {save_path}")
        plt.show()
    
    def visualize_detection_pipeline(self, original_img, save_path=None):
        """
        Visualise the entire YOLO detection pipeline step by step.
        Shows confidence maps, anchor boxes, and predictions at each scale.
        
        Args:
            original_img: Original input image (numpy array)
            save_path: Optional path to save the visualization
        """
        # Find all detection scales
        grid_sizes = []
        for key in self.detection_stages.keys():
            if key.startswith('grid_size_'):
                grid_sizes.append(self.detection_stages[key])
        
        if not grid_sizes:
            print("No detection stages found. Run inference first.")
            return
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 5 * len(grid_sizes)))
        gs = GridSpec(len(grid_sizes), 4, figure=fig, hspace=0.3, wspace=0.3)
        
        for scale_idx, grid_size in enumerate(sorted(grid_sizes, reverse=True)):
            # 1. Confidence map
            ax1 = fig.add_subplot(gs[scale_idx, 0])
            conf_map = self.detection_stages[f'confidence_map_{grid_size}'][0]  # First batch
            # Average across anchors to get overall confidence
            conf_avg = conf_map.mean(dim=0).numpy()
            im1 = ax1.imshow(conf_avg, cmap='hot', interpolation='nearest')
            ax1.set_title(f'Confidence Map\n(Grid: {grid_size}x{grid_size})')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046)
            
            # 2. Top class predictions
            ax2 = fig.add_subplot(gs[scale_idx, 1])
            class_probs = self.detection_stages[f'class_probs_{grid_size}'][0]  # First batch
            # Get top class and its probability for each location
            top_probs, top_classes = class_probs.max(dim=-1)
            # Average across anchors
            top_probs_avg = top_probs.mean(dim=0).numpy()
            im2 = ax2.imshow(top_probs_avg, cmap='viridis', interpolation='nearest')
            ax2.set_title(f'Top Class Probability\n(Grid: {grid_size}x{grid_size})')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            # 3. Anchor boxes visualization
            ax3 = fig.add_subplot(gs[scale_idx, 2])
            self._visualize_anchor_boxes(ax3, grid_size, original_img.shape[:2])
            ax3.set_title(f'Anchor Boxes\n(Stride: {self.detection_stages[f"stride_{grid_size}"]})')
            
            # 4. Predicted boxes (high confidence only)
            ax4 = fig.add_subplot(gs[scale_idx, 3])
            self._visualize_predicted_boxes(ax4, grid_size, original_img)
            ax4.set_title(f'High Confidence Predictions\n(Grid: {grid_size}x{grid_size})')
        
        plt.suptitle('YOLO Detection Pipeline Visualization', fontsize=16, y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Detection pipeline saved to {save_path}")
        plt.show()
    
    def _visualize_anchor_boxes(self, ax, grid_size, img_shape):
        """Helper to visualize anchor boxes for a specific grid size."""
        stride = self.detection_stages[f'stride_{grid_size}']
        anchors = self.detection_stages[f'anchors_{grid_size}'][0]  # First batch
        
        # Create a blank image
        img_height, img_width = img_shape
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Invert y-axis
        
        # Draw grid
        for i in range(grid_size + 1):
            ax.axhline(y=i * stride, color='gray', linewidth=0.5, alpha=0.3)
            ax.axvline(x=i * stride, color='gray', linewidth=0.5, alpha=0.3)
        
        # Draw anchor boxes at center of image
        center_x = img_width // 2
        center_y = img_height // 2
        
        colors = ['red', 'green', 'blue']
        for anchor_idx in range(anchors.shape[0]):
            anchor_w = anchors[anchor_idx, 0, 0, 0].item() * stride
            anchor_h = anchors[anchor_idx, 0, 0, 1].item() * stride
            
            rect = patches.Rectangle(
                (center_x - anchor_w/2, center_y - anchor_h/2),
                anchor_w, anchor_h,
                linewidth=2, edgecolor=colors[anchor_idx],
                facecolor='none', label=f'Anchor {anchor_idx}'
            )
            ax.add_patch(rect)
        
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _visualize_predicted_boxes(self, ax, grid_size, original_img):
        """Helper to visualize high-confidence predicted boxes."""
        # Get predictions
        conf_map = self.detection_stages[f'confidence_map_{grid_size}'][0]
        pred_boxes = self.detection_stages[f'pred_boxes_grid_{grid_size}'][0]
        stride = self.detection_stages[f'stride_{grid_size}']
        
        # Display image
        ax.imshow(original_img)
        
        # Find high confidence predictions
        conf_threshold = 0.5
        high_conf_mask = conf_map > conf_threshold
        
        # Draw boxes for high confidence predictions
        for anchor_idx in range(conf_map.shape[0]):
            for y in range(grid_size):
                for x in range(grid_size):
                    if high_conf_mask[anchor_idx, y, x]:
                        # Get box coordinates (in grid space)
                        box_x = pred_boxes[anchor_idx, y, x, 0].item() * stride
                        box_y = pred_boxes[anchor_idx, y, x, 1].item() * stride
                        box_w = pred_boxes[anchor_idx, y, x, 2].item() * stride
                        box_h = pred_boxes[anchor_idx, y, x, 3].item() * stride
                        
                        # Convert to corner coordinates
                        x1 = box_x - box_w / 2
                        y1 = box_y - box_h / 2
                        
                        # Draw rectangle
                        conf_score = conf_map[anchor_idx, y, x].item()
                        rect = patches.Rectangle(
                            (x1, y1), box_w, box_h,
                            linewidth=2, edgecolor='yellow',
                            facecolor='none', alpha=conf_score
                        )
                        ax.add_patch(rect)
        
        ax.axis('off')
    
    def visualize_detections_comparison(self, original_img, detections_before_nms, 
                                      detections_after_nms, save_path=None):
        """
        Compare detections before and after NMS to understand the filtering process.
        
        Args:
            original_img: Original input image
            detections_before_nms: Raw detections before NMS
            detections_after_nms: Filtered detections after NMS
            save_path: Optional path to save visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Before NMS
        ax1.imshow(original_img)
        ax1.set_title(f'Before NMS\n({len(detections_before_nms)} detections)')
        
        # Draw all detections with transparency based on confidence
        for det in detections_before_nms:
            x1, y1, x2, y2, conf = det[:5]
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1, edgecolor='red',
                facecolor='none', alpha=min(conf.item(), 1.0)
            )
            ax1.add_patch(rect)
        
        # After NMS
        ax2.imshow(original_img)
        ax2.set_title(f'After NMS\n({len(detections_after_nms)} detections)')
        
        # Draw filtered detections
        for det in detections_after_nms:
            x1, y1, x2, y2, conf, cls_conf, cls = det
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='green',
                facecolor='none'
            )
            ax2.add_patch(rect)
            
            # Add label
            label = f'{COCO_CLASSES[int(cls)]}: {conf:.2f}'
            ax2.text(x1, y1-5, label, color='green', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax1.axis('off')
        ax2.axis('off')
        
        plt.suptitle('Non-Maximum Suppression (NMS) Effect', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"NMS comparison saved to {save_path}")
        plt.show()


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
        self.img_size = img_size  # Input image size (typically 416x416)
        self.grid_size = 0  # Will be set dynamically based on feature map size
        self.stride = 0  # Pixel stride between grid cells
       
    def forward(self, x, visualizer=None):
        """
        Forward pass through YOLO layer.
        
        Args:
            x: Feature map tensor of shape [batch_size, num_anchors*(5+num_classes), grid_size, grid_size]
               where 5 represents: x, y, width, height, objectness confidence
            visualizer: Optional NetworkVisualizer instance to capture intermediate results
        
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
       
        # Store raw predictions before activation for visualization
        if visualizer is not None:
            visualizer.detection_stages[f'raw_predictions_{grid_size}'] = prediction.detach().cpu()
       
        # Extract and apply activation functions to predictions
        # Sigmoid constrains x,y to [0,1] within each grid cell
        x = torch.sigmoid(prediction[..., 0])  # Centre x coordinate (relative to grid cell)
        y = torch.sigmoid(prediction[..., 1])  # Centre y coordinate (relative to grid cell)
        # Width and height are in log space (will be exponentiated later)
        w = prediction[..., 2]  # Width (log space)
        h = prediction[..., 3]  # Height (log space)
        # Objectness: probability that this anchor contains an object
        conf = torch.sigmoid(prediction[..., 4])  # Confidence/objectness score
        # Class predictions: probability distribution over classes
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class probabilities
       
        # Store confidence map for visualization
        if visualizer is not None:
            visualizer.detection_stages[f'confidence_map_{grid_size}'] = conf.detach().cpu()
            visualizer.detection_stages[f'class_probs_{grid_size}'] = pred_cls.detach().cpu()
       
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
       
        # Store anchor dimensions for visualization
        if visualizer is not None:
            visualizer.detection_stages[f'anchors_{grid_size}'] = torch.stack([anchor_w, anchor_h], dim=-1).detach().cpu()
            visualizer.detection_stages[f'grid_size_{grid_size}'] = grid_size
            visualizer.detection_stages[f'stride_{grid_size}'] = stride
       
        # Convert predictions to bounding boxes in grid space
        pred_boxes = torch.zeros_like(prediction[..., :4])
        # x,y predictions are relative to grid cell, add grid coordinates for absolute position
        pred_boxes[..., 0] = x + grid_x  # Absolute x in grid coordinates
        pred_boxes[..., 1] = y + grid_y  # Absolute y in grid coordinates
        # Width/height use exponential to ensure positive values, multiplied by anchor dimensions
        pred_boxes[..., 2] = torch.exp(w) * anchor_w  # Absolute width in grid coordinates
        pred_boxes[..., 3] = torch.exp(h) * anchor_h  # Absolute height in grid coordinates
       
        # Store predicted boxes in grid space for visualization
        if visualizer is not None:
            visualizer.detection_stages[f'pred_boxes_grid_{grid_size}'] = pred_boxes.detach().cpu()
       
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
   
    def forward(self, x, visualizer=None):
        """
        Forward pass through the entire network.
        
        Args:
            x: Input image tensor [batch_size, 3, height, width]
            visualizer: Optional NetworkVisualizer instance to capture intermediate results
            
        Returns:
            Concatenated detections from all YOLO layers
        """
        outputs = []  # Collect outputs from YOLO layers
        layer_outputs = []  # Store outputs from all layers (needed for route/shortcut)
       
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
                # Add features from specified previous layer (residual connection)
                from_layer = int(block['from'])
                x = layer_outputs[-1] + layer_outputs[from_layer]
               
            elif block['type'] == 'yolo':
                # YOLO detection layer - pass visualizer if available
                x = module[0](x, visualizer)  # module is Sequential, so access first element
                outputs.append(x)
               
            layer_outputs.append(x)  # Save output for potential route/shortcut layers
           
        # Concatenate all YOLO outputs along the detection dimension
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
    # Union = Area1 + Area2 - Intersection (add small epsilon to avoid division by zero)
    union_area = b1_area.unsqueeze(1) + b2_area - inter_area + 1e-16
   
    return inter_area / union_area

# COCO class names - 80 object categories
# These correspond to the class predictions from YOLO trained on COCO dataset
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
    
    # Define drawing parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    colour = (255, 0, 0)  # Red in RGB format
    
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
        cv2.putText(img, label, (x1, y1 - 10), font, 0.5, colour, 2)
        
    # Image is modified in-place, return for convenience
    return img

# Main detection function
def detect_image(cfg_path, weights_path, img_path, output_path, conf_thres=0.5, nms_thres=0.4, visualize=False):
    """
    Run YOLO object detection on a single image.
    
    Args:
        cfg_path: Path to YOLO configuration file (.cfg)
        weights_path: Path to pre-trained weights file (.weights)
        img_path: Path to input image
        output_path: Path to save output image with detections
        conf_thres: Confidence threshold for object detection
        nms_thres: IoU threshold for Non-Maximum Suppression
        visualize: Whether to generate visualization outputs
        
    Returns:
        detections: Tensor of final detections after NMS
        visualizer: NetworkVisualizer instance (if visualize=True)
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
    
    # Initialize visualizer if requested
    visualizer = NetworkVisualizer() if visualize else None
    if visualizer:
        visualizer.register_hooks(model)
   
    # Preprocess input image
    img_tensor, original_img = preprocess_image(img_path)
    img_tensor = img_tensor.to(device)  # Move input tensor to same device as model
   
    # Warm up GPU (optional but recommended for accurate timing)
    # Run a few forward passes to ensure CUDA kernels are loaded
    if device.type == 'cuda':
        for _ in range(3):
            _ = model(img_tensor)
        torch.cuda.synchronize()  # Wait for all GPU operations to complete
   
    # Run actual inference with timing
    start_time = time.time()
   
    with torch.no_grad():  # Disable gradient computation for inference
        detections = model(img_tensor, visualizer)  # Forward pass through network
        detections_before_nms = detections.clone() if visualize else None
        detections = non_max_suppression(detections, conf_thres, nms_thres)  # Apply NMS
   
    if device.type == 'cuda':
        torch.cuda.synchronize()  # Ensure GPU operations are complete for accurate timing
   
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time*1000:.2f} ms")
   
    # Draw detections on original image
    if len(detections) > 0:
        # Note: draw_detections modifies the image in-place
        # Move detections to CPU for drawing (OpenCV uses CPU)
        result_img = draw_detections(original_img.copy(), detections.cpu())
    else:
        result_img = original_img
        print("No objects detected")
    
    # Save result image
    # Note: OpenCV expects BGR format for saving, so convert from RGB
    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    print(f"Result saved to {output_path}")
    
    # Generate visualizations if requested
    if visualize and visualizer:
        print("\nGenerating visualizations...")
        
        # Create output directory for visualizations
        vis_dir = os.path.splitext(output_path)[0] + '_visualizations'
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Feature map visualizations
        print("1. Creating feature map visualizations...")
        # Visualize early, middle, and late layers
        visualizer.visualize_feature_maps('conv_0', max_channels=32, 
                                        save_path=os.path.join(vis_dir, 'features_conv_0.png'))
        visualizer.visualize_feature_maps('conv_10', max_channels=32,
                                        save_path=os.path.join(vis_dir, 'features_conv_10.png'))
        visualizer.visualize_feature_maps('conv_30', max_channels=32,
                                        save_path=os.path.join(vis_dir, 'features_conv_30.png'))
        
        # 2. Layer statistics
        print("2. Creating layer statistics visualization...")
        visualizer.visualize_layer_statistics(save_path=os.path.join(vis_dir, 'layer_statistics.png'))
        
        # 3. Activation histograms
        print("3. Creating activation histograms...")
        visualizer.visualize_activation_histograms(save_path=os.path.join(vis_dir, 'activation_histograms.png'))
        
        # 4. Detection pipeline
        print("4. Creating detection pipeline visualization...")
        visualizer.visualize_detection_pipeline(original_img, 
                                              save_path=os.path.join(vis_dir, 'detection_pipeline.png'))
        
        # 5. NMS comparison
        if detections_before_nms is not None and len(detections) > 0:
            print("5. Creating NMS comparison visualization...")
            # Filter detections before NMS for visualization
            conf_mask = detections_before_nms[0, :, 4] >= conf_thres
            filtered_before = detections_before_nms[0][conf_mask]
            
            visualizer.visualize_detections_comparison(
                original_img, filtered_before, detections,
                save_path=os.path.join(vis_dir, 'nms_comparison.png')
            )
        
        print(f"\nAll visualizations saved to: {vis_dir}/")
    
    return detections, visualizer


def create_network_analysis_report(model, img_path, output_dir='network_analysis'):
    """
    Create a comprehensive analysis report of the YOLO network.
    This function runs inference and generates all possible visualizations.
    
    Args:
        model: Trained Darknet model
        img_path: Path to test image
        output_dir: Directory to save analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Run detection with visualization
    print("Running YOLO detection with full visualization suite...")
    detections, visualizer = detect_image(
        model.cfg_path,
        model.weights_path,
        img_path,
        os.path.join(output_dir, 'detection_result.jpg'),
        visualize=True
    )
    
    # Generate summary report
    with open(os.path.join(output_dir, 'network_analysis_report.txt'), 'w') as f:
        f.write("YOLO Network Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Model architecture summary
        f.write("Model Architecture:\n")
        f.write(f"- Total layers: {len(model.module_list)}\n")
        f.write(f"- Input size: {model.img_size}x{model.img_size}\n")
        
        # Count layer types
        layer_counts = defaultdict(int)
        for block in model.blocks[1:]:
            layer_counts[block['type']] += 1
        
        f.write("\nLayer composition:\n")
        for layer_type, count in layer_counts.items():
            f.write(f"- {layer_type}: {count}\n")
        
        # Detection results
        f.write(f"\n\nDetection Results:\n")
        f.write(f"- Total detections: {len(detections)}\n")
        if len(detections) > 0:
            f.write("\nDetected objects:\n")
            for det in detections:
                cls = int(det[6])
                conf = det[4]
                f.write(f"- {COCO_CLASSES[cls]}: {conf:.3f}\n")
        
        # Layer statistics summary
        f.write("\n\nLayer Statistics Summary:\n")
        for layer_name, stats in visualizer.layer_stats.items():
            f.write(f"\n{layer_name}:\n")
            f.write(f"  Shape: {stats['shape']}\n")
            f.write(f"  Mean: {stats['mean']:.4f}\n")
            f.write(f"  Std: {stats['std']:.4f}\n")
            f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
            f.write(f"  Dead neurons: {stats['dead_neurons_%']:.1f}%\n")
    
    print(f"\nComplete network analysis saved to: {output_dir}/")
    print("Check the generated visualizations to understand:")
    print("- Feature maps: What patterns the network learns at different depths")
    print("- Layer statistics: Network health and potential issues")
    print("- Activation histograms: Distribution of values through the network")
    print("- Detection pipeline: How YOLO makes predictions at each scale")
    print("- NMS comparison: How duplicate detections are filtered")


# Example usage
if __name__ == "__main__":
    """
    Example of how to use the YOLO detector.
    Requires:
    - yolov3.cfg: Network architecture configuration
    - yolov3.weights: Pre-trained weights (download from YOLO website)
    - test_image.jpg: Input image for detection
    """
   
    # File paths
    cfg_path = "yolov3.cfg"
    weights_path = "yolov3.weights"
    img_path = "test_image.jpg"  # Your input image
    output_path = "detected_image.jpg"  # Output path
   
    # Run detection
    detections = detect_image(cfg_path, weights_path, img_path, output_path)
   
    # Print detection summary
    if len(detections) > 0:
        print(f"\nDetected {len(detections)} objects:")
        for det in detections:
            cls = int(det[6])  # Class ID
            conf = det[4]      # Objectness confidence
            print(f"- {COCO_CLASSES[cls]}: {conf:.2f}")