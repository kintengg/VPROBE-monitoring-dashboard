import torch
import torch.nn as nn 
import torchvision.transforms as T
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np
import cv2
from tqdm import tqdm
import json
import supervision as sv
import csv
from pathlib import Path
from datetime import datetime
from traffic_congestion import TrafficCongestionEstimator, RealtimeTrafficCongestionEstimator

def format_video_timestamp(frame_number, fps):
    """Convert frame number to video timestamp HH:MM:SS format"""
    total_seconds = int(frame_number / fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def calculate_gflops(model, input_size=(3, 640, 640)):
    """Calculate GFLOPs using custom implementation"""
    
    def estimate_flops(model, input_tensor):
        """Estimate FLOPs for common operations"""
        model.eval()
        with torch.no_grad():
            # Hook to count operations
            total_flops = [0]
            
            def flop_count_hook(module, input, output):
                if isinstance(module, torch.nn.Conv2d):
                    # Conv2d: (batch_size * output_height * output_width * kernel_height * kernel_width * input_channels * output_channels) / groups
                    if hasattr(output, 'shape'):
                        batch_size, out_channels, out_h, out_w = output.shape
                        kernel_h, kernel_w = module.kernel_size
                        in_channels = module.in_channels
                        groups = module.groups
                        flops = batch_size * out_h * out_w * kernel_h * kernel_w * in_channels * out_channels / groups
                        total_flops[0] += flops
                elif isinstance(module, torch.nn.Linear):
                    # Linear: input_features * output_features * batch_size
                    if hasattr(output, 'shape') and len(output.shape) >= 2:
                        batch_size = output.shape[0]
                        flops = module.in_features * module.out_features * batch_size
                        total_flops[0] += flops
                elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                    # BatchNorm: 2 * num_features * batch_size * spatial_size
                    if hasattr(output, 'numel'):
                        flops = 2 * output.numel()
                        total_flops[0] += flops
                elif isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)):
                    # Activation functions
                    if hasattr(output, 'numel'):
                        flops = output.numel()
                        total_flops[0] += flops
            
            # Register hooks
            hooks = []
            for module in model.modules():
                hooks.append(module.register_forward_hook(flop_count_hook))
            
            # Forward pass
            _ = model(input_tensor)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            return total_flops[0]
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
    
    try:
        flops = estimate_flops(model, dummy_input)
        gflops = flops / 1e9
        return gflops
    except Exception as e:
        print(f"Warning: Could not calculate GFLOPs: {e}")
        return None

def cap_from_youtube(youtube_url, resolution='best'):
    """Create video capture from YouTube URL"""
    try:
        import yt_dlp as youtube_dl
        
        ydl_opts = {
            'format': f'{resolution}[ext=mp4]' if resolution != 'best' else 'best[ext=mp4]',
            'quiet': True,
            'no_warnings': True,
        }
        
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            video_url = info['url']
            
        cap = cv2.VideoCapture(video_url)
        return cap
        
    except ImportError:
        print("Error: yt-dlp not installed. Install with: pip install yt-dlp")
        return None
    except Exception as e:
        print(f"Error loading YouTube video: {e}")
        return None

def load_class_names(ann_file):
    """Load class names from COCO annotation file
    
    Returns:
        tuple: (class_names dict, class_id_mapping dict)
            - class_names: maps annotation category IDs to names
            - class_id_mapping: maps model output indices (0-based) to annotation category IDs
    """
    try:
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        class_names = {}
        categories = data.get('categories', [])
        
        for category in categories:
            class_names[category['id']] = category['name']
        
        # Create mapping from model output indices (0-based) to annotation category IDs
        # Sort by category ID to ensure consistent ordering
        sorted_categories = sorted(categories, key=lambda x: x['id'])
        class_id_mapping = {}
        
        for model_idx, category in enumerate(sorted_categories):
            class_id_mapping[model_idx] = category['id']
        
        print(f"Loaded {len(class_names)} classes from annotation file:")
        print(f"  Model index -> Annotation ID mapping: {class_id_mapping}")
        print(f"  Class names: {class_names}")
        
        return class_names, class_id_mapping
    except Exception as e:
        print(f"Warning: Could not load class names from {ann_file}: {e}")
        return {}, {}

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)
def draw(images, labels, boxes, scores, thrh = 0.6, path = ""):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"label: {lab[j].item()} {round(scrs[j].item(),2)}", font=ImageFont.load_default(), fill='blue')
        if path == "":
            im.save(f'results_{i}.jpg')
        else:
            im.save(path)

def get_class_color(class_id, num_classes=80):
    """Generate a consistent color for each class ID"""
    import colorsys
    
    # Generate colors using HSV color space for better distribution
    hue = (class_id * 137.508) % 360  # Use golden angle for good distribution
    saturation = 0.8
    value = 0.9
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
    
    # Convert to BGR format for OpenCV (0-255 range)
    bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    
    return bgr

def draw_cv2(frame, labels, boxes, scores, threshold=0.6, class_names=None, class_id_mapping=None, border_thickness=3):
    """Draw bounding boxes on OpenCV frame"""
    if len(labels) == 0:
        return frame
    
    # Convert tensors to numpy arrays
    scr = scores[0].cpu().detach().numpy() if torch.is_tensor(scores[0]) else scores[0]
    lab = labels[0].cpu().detach().numpy() if torch.is_tensor(labels[0]) else labels[0]
    box = boxes[0].cpu().detach().numpy() if torch.is_tensor(boxes[0]) else boxes[0]
    
    # Filter by threshold
    valid_mask = scr > threshold
    lab = lab[valid_mask]
    box = box[valid_mask]
    scrs = scr[valid_mask]
    
    for j, b in enumerate(box):
        x1, y1, x2, y2 = b.astype(int)
        
        # Get model output class index
        model_class_idx = int(lab[j])
        
        # Remap to annotation category ID if mapping is available
        if class_id_mapping and model_class_idx in class_id_mapping:
            category_id = class_id_mapping[model_class_idx]
        else:
            category_id = model_class_idx
        
        # Get class-specific color (use category_id for consistent colors)
        box_color = get_class_color(category_id)
        
        # Draw bounding box with class-specific color
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, border_thickness)
        
        # Get class name if available, otherwise use category ID
        if class_names and category_id in class_names:
            class_text = class_names[category_id]
        else:
            class_text = f"class_{category_id}"
        
        label_text = f"{class_text}: {round(float(scrs[j]),2)}"
        
        # Calculate text size for background rectangle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Draw background rectangle for text (using same color as box but darker)
        label_y = max(y1 - 10, text_height + 10)  # Ensure label stays within frame
        text_bg_color = tuple(int(c * 0.7) for c in box_color)  # Darker version of box color
        # cv2.rectangle(frame, 
        #              (x1, label_y - text_height - 10), 
        #              (x1 + text_width + 10, label_y + baseline), 
        #              text_bg_color, -1)
        
        # Draw white text on colored background
        cv2.putText(frame, label_text, (x1 + 5, label_y - 1), 
                   font, font_scale, (255, 255, 255), font_thickness)
    
    return frame

def convert_to_supervision_detections(labels, boxes, scores, threshold=0.6, class_id_mapping=None):
    """Convert RT-DETR outputs to supervision Detections format"""
    # Convert tensors to numpy arrays
    scr = scores[0].cpu().detach().numpy() if torch.is_tensor(scores[0]) else scores[0]
    lab = labels[0].cpu().detach().numpy() if torch.is_tensor(labels[0]) else labels[0]
    box = boxes[0].cpu().detach().numpy() if torch.is_tensor(boxes[0]) else boxes[0]

    # Filter by threshold
    valid_mask = scr > threshold
    lab = lab[valid_mask]
    box = box[valid_mask]
    scrs = scr[valid_mask]

    # Remap class IDs if mapping is provided
    if class_id_mapping:
        remapped_lab = np.array([class_id_mapping.get(int(cls), int(cls)) for cls in lab])
    else:
        remapped_lab = lab

    # Create supervision Detections object
    detections = sv.Detections(
        xyxy=box,
        confidence=scrs,
        class_id=remapped_lab.astype(int)
    )

    return detections

def draw_tracked_boxes(frame, detections, class_names=None, border_thickness=3):
    """Draw tracked bounding boxes with track IDs on OpenCV frame
    Note: detections.class_id should already be remapped to annotation category IDs
    """
    if len(detections) == 0:
        return frame

    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)

        # Get category ID (already remapped in convert_to_supervision_detections)
        category_id = int(detections.class_id[i])
        box_color = get_class_color(category_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, border_thickness)

        # Get class name if available
        if class_names and category_id in class_names:
            class_text = class_names[category_id]
        else:
            class_text = f"class_{category_id}"

        # Get track ID if available
        track_id = detections.tracker_id[i] if detections.tracker_id is not None else None

        # Create label text with track ID
        if track_id is not None:
            label_text = f"ID:{track_id} {class_text}: {round(float(detections.confidence[i]),2)}"
        else:
            label_text = f"{class_text}: {round(float(detections.confidence[i]),2)}"

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        label_y = max(y1 - 10, text_height + 10)
        cv2.putText(frame, label_text, (x1 + 5, label_y - 1),
                   font, font_scale, (255, 255, 255), font_thickness)

    return frame

def load_counting_config(config_path):
    """Load counting configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        if 'lines' not in config:
            print(f"Warning: No 'lines' found in counting config {config_path}")
            return []

        counting_lines = []
        for line_config in config['lines']:
            # Validate required fields
            if not all(key in line_config for key in ['name', 'start', 'end']):
                print(f"Warning: Skipping invalid line config: {line_config}")
                continue

            # Get trigger anchors (default to CENTER if not specified)
            trigger_anchors = line_config.get('trigger_anchors', ['CENTER'])

            # Convert string anchors to sv.Position enum
            anchor_mapping = {
                'CENTER': sv.Position.CENTER,
                'BOTTOM': sv.Position.BOTTOM_CENTER,  # Fixed: use BOTTOM_CENTER
                'BOTTOM_CENTER': sv.Position.BOTTOM_CENTER,
                'TOP': sv.Position.TOP_CENTER,  # Fixed: use TOP_CENTER
                'TOP_CENTER': sv.Position.TOP_CENTER,
                'LEFT': sv.Position.CENTER_LEFT,  # Fixed: use CENTER_LEFT
                'CENTER_LEFT': sv.Position.CENTER_LEFT,
                'RIGHT': sv.Position.CENTER_RIGHT,  # Fixed: use CENTER_RIGHT
                'CENTER_RIGHT': sv.Position.CENTER_RIGHT,
                'BOTTOM_LEFT': sv.Position.BOTTOM_LEFT,
                'BOTTOM_RIGHT': sv.Position.BOTTOM_RIGHT,
                'TOP_LEFT': sv.Position.TOP_LEFT,
                'TOP_RIGHT': sv.Position.TOP_RIGHT,
                'CENTER_OF_MASS': sv.Position.CENTER_OF_MASS,
            }

            anchors = []
            for anchor_str in trigger_anchors:
                if anchor_str in anchor_mapping:
                    anchors.append(anchor_mapping[anchor_str])
                else:
                    print(f"Warning: Unknown anchor '{anchor_str}', using CENTER")
                    anchors.append(sv.Position.CENTER)

            counting_lines.append({
                'name': line_config['name'],
                'start': np.array(line_config['start']),
                'end': np.array(line_config['end']),
                'anchors': anchors
            })

        print(f"Loaded {len(counting_lines)} counting line(s) from {config_path}")
        return counting_lines

    except Exception as e:
        print(f"Error loading counting config from {config_path}: {e}")
        return []

def write_summary(summary_file, line_zones, class_names):
    """Write counting summary to file. Can be called multiple times to update."""
    if summary_file is None or not line_zones:
        return

    summary_file.seek(0)
    summary_file.truncate()

    summary_text = "\n=== Counting Summary (OUT counts only) ===\n"

    for line_data in line_zones:
        line_name = line_data['name']
        class_counts = line_data['class_counts']
        total_out = sum(class_counts.values())

        summary_text += f"\n{line_name}:\n"
        summary_text += f"  Total OUT: {total_out}\n"

        if class_counts:
            summary_text += f"  Per-class breakdown:\n"
            for class_id in sorted(class_counts.keys()):
                class_name = class_names.get(class_id, f"class_{class_id}")
                count = class_counts[class_id]
                summary_text += f"    {class_name}: {count}\n"
        else:
            summary_text += f"  No objects counted yet\n"

    summary_text += "\n==========================================\n"
    summary_file.write(summary_text)
    summary_file.flush()

def draw_sidebar_panel(frame, line_zones, class_names, congestion_status=None, congestion_mode=None, sidebar_width=300):
    """Draw combined sidebar panel with LOS (top) and counting (bottom) on the right side."""
    if not line_zones:
        return frame

    height = frame.shape[0]
    width = frame.shape[1]

    # Create sidebar (extend frame on the right)
    sidebar_x = width - sidebar_width

    # Draw solid background for sidebar
    cv2.rectangle(frame, (sidebar_x, 0), (width, height), (40, 40, 40), -1)

    # Draw sidebar border
    cv2.rectangle(frame, (sidebar_x, 0), (width - 1, height - 1), (100, 100, 100), 2)

    current_y = 20
    line_height = 32

    # ============ LOS PANEL (TOP HALF) ============
    if congestion_status is not None:
        # Import required for LOS color
        from tools.traffic_congestion import RealtimeTrafficCongestionEstimator
        los_color = RealtimeTrafficCongestionEstimator.get_los_color(congestion_status['los'])
        los_desc = RealtimeTrafficCongestionEstimator.get_los_description(congestion_status['los'])

        # LOS Title
        cv2.putText(frame, "CONGESTION", (sidebar_x + 10, current_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        current_y += line_height

        # Mode indicator
        mode_text = "REAL-TIME" if congestion_mode == 'realtime' else "TIME-BASED"
        cv2.putText(frame, f"({mode_text})", (sidebar_x + 10, current_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
        current_y += line_height - 8

        # V/C Ratio
        vc_text = f"V/C: {congestion_status['vc_ratio']:.4f}"
        cv2.putText(frame, vc_text, (sidebar_x + 10, current_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        current_y += line_height - 5

        # LOS
        los_text = f"LOS: {congestion_status['los']}"
        cv2.putText(frame, los_text, (sidebar_x + 10, current_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.85, los_color, 3)
        current_y += line_height + 2

        # LOS Description
        cv2.putText(frame, los_desc.split('-')[1].strip(), (sidebar_x + 10, current_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 2)
        current_y += line_height - 8

        # Volume
        vol_text = f"Vol: {congestion_status['volume']:.1f}"
        cv2.putText(frame, vol_text, (sidebar_x + 10, current_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        current_y += line_height - 10

        # Capacity
        cap_text = f"Cap: {congestion_status['capacity']:.1f}"
        cv2.putText(frame, cap_text, (sidebar_x + 10, current_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        current_y += line_height + 2

    # ============ SEPARATOR ============
    cv2.line(frame, (sidebar_x + 5, current_y - 5), (width - 5, current_y - 5), (80, 80, 80), 2)
    current_y += 30  # More space before counting section

    # ============ COUNTING PANEL (BOTTOM HALF) ============
    cv2.putText(frame, "COUNTING", (sidebar_x + 10, current_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    current_y += line_height + 5

    # Draw counts for each line
    for line_data in line_zones:
        class_counts = line_data['class_counts']
        total_out = sum(class_counts.values())

        # Show TOTAL count
        count_text = f"TOTAL: {total_out}"
        cv2.putText(frame, count_text, (sidebar_x + 10, current_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        current_y += line_height

        # Per-class breakdown
        if class_counts:
            for class_id in sorted(class_counts.keys()):
                class_name = class_names.get(class_id, f"class_{class_id}")
                count = class_counts[class_id]
                class_text = f"{class_name}: {count}"
                cv2.putText(frame, class_text, (sidebar_x + 25, current_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
                current_y += line_height - 5

            # Space between lines
            current_y += 4

    return frame

def process_video(args):
    """Process video file frame by frame"""
    cfg = YAMLConfig(args.config, resume=args.resume)
    
    # Load class names and mapping from annotation file
    class_names = {}
    class_id_mapping = {}
    if hasattr(args, 'ann_file') and args.ann_file:
        class_names, class_id_mapping = load_class_names(args.ann_file)
    else:
        print("No annotation file specified, using class IDs")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    
    cfg.model.load_state_dict(state)
    
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    model.eval()
    
    # Calculate and print model statistics
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {n_parameters:,}')
    
    try:
        gflops = calculate_gflops(model)
        if gflops is not None:
            print(f'GFLOPs: {gflops:.2f}')
        else:
            print('GFLOPs: calculation failed')
    except Exception as e:
        print(f'GFLOPs: calculation error - {e}')
    
    transforms = T.Compose([
        T.Resize((640, 640)),  
        T.ToTensor(),
    ])
    
    # Open video (local file or YouTube URL)
    if args.video_file.startswith(('http://', 'https://')) and ('youtube.com' in args.video_file or 'youtu.be' in args.video_file):
        print(f"Loading YouTube video: {args.video_file}")
        cap = cap_from_youtube(args.video_file)
        if cap is None:
            raise ValueError(f"Cannot load YouTube video: {args.video_file}")
    else:
        cap = cv2.VideoCapture(args.video_file)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {args.video_file}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sidebar width for counting display
    sidebar_width = 300
    output_width = width + sidebar_width if (hasattr(args, 'counting_config') and args.counting_config) else width
    output_height = height

    print(f"Video info: {width}x{height} at {fps} FPS, {total_frames} frames")

    # Setup output video writer (optional)
    out = None
    if args.output:
        output_suffix = str(Path(args.output).suffix).lower()
        # Prefer MP4-compatible codec when writing .mp4 files.
        fourcc = cv2.VideoWriter_fourcc(*('mp4v' if output_suffix == '.mp4' else 'XVID'))
        out = cv2.VideoWriter(args.output, fourcc, fps, (output_width, output_height))
        print(f"Will save output to: {args.output}")

    # Initialize tracking if enabled
    tracker = None
    trace_annotator = None
    if hasattr(args, 'tracking') and args.tracking:
        print("ByteTrack tracking enabled")
        tracker = sv.ByteTrack()
        trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)

    # Initialize counting if enabled
    line_zones = []
    line_zone_annotators = []
    counting_data = []
    csv_writer = None
    csv_file = None
    summary_file = None

    if hasattr(args, 'counting_config') and args.counting_config:
        if not tracker:
            print("Warning: Counting requires --tracking to be enabled. Ignoring --counting-config.")
        else:
            counting_lines = load_counting_config(args.counting_config)

            if counting_lines:
                # Initialize LineZone for each counting line
                for line_info in counting_lines:
                    line_zone = sv.LineZone(
                        start=sv.Point(line_info['start'][0], line_info['start'][1]),
                        end=sv.Point(line_info['end'][0], line_info['end'][1]),
                        triggering_anchors=line_info['anchors']
                    )
                    line_zones.append({
                        'name': line_info['name'],
                        'zone': line_zone,
                        'start': line_info['start'],
                        'end': line_info['end'],
                        'in_count': 0,
                        'out_count': 0,
                        'class_counts': {}  # Track counts per class
                    })

                    # Create annotator for this line (empty color disables text and box)
                    line_zone_annotator = sv.LineZoneAnnotator(
                        thickness=4
                    )
                    line_zone_annotators.append(line_zone_annotator)

                # Setup CSV file for counting data
                if args.output:
                    csv_path = str(Path(args.output).with_suffix('')) + '_counts.csv'
                    csv_file = open(csv_path, 'w', newline='')
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(['timestamp', 'frame_number', 'line_name', 'track_id', 'class_id', 'class_name', 'direction', 'total_in', 'total_out'])
                    print(f"Counting data will be saved to: {csv_path}")

                    # Setup TXT file for counting summary
                    summary_path = str(Path(args.output).with_suffix('')) + '_summary.txt'
                    summary_file = open(summary_path, 'w')
                    print(f"Counting summary will be saved to: {summary_path}")

    # Initialize traffic congestion estimator if enabled
    congestion_estimator = None
    congestion_csv_writer = None
    congestion_csv_file = None
    congestion_mode = None  # 'realtime' or 'timebased'
    
    if hasattr(args, 'congestion') and args.congestion:
        if not tracker:
            print("Warning: Congestion estimation requires --tracking to be enabled. Ignoring --congestion.")
        else:
            if hasattr(args, 'road_length') and args.road_length is not None:
                # Real-time mode
                congestion_mode = 'realtime'
                print("Traffic congestion estimation enabled (REAL-TIME MODE):")
                print(f"  Road length: {args.road_length} km")
                print(f"  Lanes: {args.lanes}")
                print(f"  Jam density: {RealtimeTrafficCongestionEstimator.JAM_DENSITY} vehicles/km/lane")
                
                congestion_estimator = RealtimeTrafficCongestionEstimator(
                    road_length_km=args.road_length,
                    num_lanes=args.lanes
                )
                print(f"  Capacity (N_max): {congestion_estimator.capacity:.2f} vehicles")
                
            elif hasattr(args, 'road_width') and args.road_width is not None:
                # Time-based mode
                congestion_mode = 'timebased'
                print(f"Traffic congestion estimation enabled (TIME-BASED MODE):")
                print(f"  Road width: {args.road_width} meters")
                print(f"  Number of lanes: {args.num_lanes}")
                print(f"  Time interval: {args.time_interval} minutes")
                
                congestion_estimator = TrafficCongestionEstimator(
                    road_width=args.road_width,
                    time_interval=args.time_interval,
                    num_lanes=args.num_lanes
                )
                
                print(f"  Hourly capacity: {congestion_estimator.capacity} vehicles/hour")
            
            # Setup CSV file for congestion data with detailed vehicle tracking
            if args.output:
                congestion_csv_path = str(Path(args.output).with_suffix('')) + '_congestion.csv'
                congestion_csv_file = open(congestion_csv_path, 'w', newline='')
                congestion_csv_writer = csv.writer(congestion_csv_file)
                
                if congestion_mode == 'realtime':
                    congestion_csv_writer.writerow([
                        'timestamp', 'frame_number', 'volume', 'capacity', 
                        'vc_ratio', 'los',
                        'los_vehicles_total', 'los_vehicles_breakdown',
                        'vehicles_counted_total', 'vehicles_counted_breakdown'
                    ])
                else:
                    congestion_csv_writer.writerow([
                        'timestamp', 'frame_number', 'volume', 'capacity', 
                        'vc_ratio', 'los',
                        'los_vehicles_total', 'los_vehicles_breakdown',
                        'vehicles_counted_total', 'vehicles_counted_breakdown'
                    ])
                print(f"Congestion data will be saved to: {congestion_csv_path}")

    frame_count = 0
    detections = None
    track_to_class = {}  # Persistent mapping of track_id to class_id across frames
    total_vehicles_counted = {}  # Track ALL vehicles that crossed counting lines (cumulative, never decreases)
    congestion_status = None  # Initialize congestion status

    # Determine batch size
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 1
    
    if batch_size > 1:
        print(f"Batch processing enabled: processing {batch_size} frames simultaneously for faster inference")
        if tracker is not None:
            print(f"Note: Inference is batched, but tracking/counting/congestion are processed frame-by-frame to maintain accuracy")

    with torch.no_grad():
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        # Batch processing loop
        frame_buffer = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Process remaining frames in buffer (same logic as main batch processing)
                if len(frame_buffer) > 0 and batch_size > 1:
                    # Batch preprocess remaining frames
                    batch_tensors = []
                    batch_orig_sizes = []
                    
                    for frame_item in frame_buffer:
                        frame_rgb = cv2.cvtColor(frame_item, cv2.COLOR_BGR2RGB)
                        im_pil = Image.fromarray(frame_rgb)
                        w, h = im_pil.size
                        im_tensor = transforms(im_pil)
                        batch_tensors.append(im_tensor)
                        batch_orig_sizes.append(torch.tensor([w, h]))
                    
                    # Stack and transfer to GPU once
                    batch_frames = torch.stack(batch_tensors).to(args.device)
                    batch_orig_sizes = torch.stack(batch_orig_sizes).to(args.device)
                    
                    # Run batched inference
                    try:
                        with autocast('cuda'):
                            batch_outputs = model(batch_frames, batch_orig_sizes)
                    except TypeError:
                        with autocast():
                            batch_outputs = model(batch_frames, batch_orig_sizes)
                    
                    batch_labels, batch_boxes, batch_scores = batch_outputs
                    
                    # Process each frame in the batch (sequentially for tracking)
                    for i, frame_item in enumerate(frame_buffer):
                        # Increment frame count first
                        frame_count += 1
                        
                        labels = [batch_labels[i]]
                        boxes = [batch_boxes[i]]
                        scores = [batch_scores[i]]
                        
                        # Process detections with tracking if enabled (SAME LOGIC AS MAIN LOOP)
                        if tracker is not None:
                            detections = convert_to_supervision_detections(labels, boxes, scores, args.threshold, class_id_mapping)
                            detections = tracker.update_with_detections(detections)
                            
                            # Process line crossing detection for counting
                            if line_zones:
                                import time
                                current_time = time.time()
                                
                                current_frame_classes = {}
                                if detections.tracker_id is not None:
                                    for j in range(len(detections)):
                                        track_id = detections.tracker_id[j]
                                        class_id = int(detections.class_id[j])
                                        current_frame_classes[track_id] = class_id
                                        track_to_class[track_id] = class_id
                                
                                for idx, line_data in enumerate(line_zones):
                                    line_zone = line_data['zone']
                                    crossed_in_mask, crossed_out_mask = line_zone.trigger(detections)
                                    line_data['in_count'] = line_zone.in_count
                                    line_data['out_count'] = line_zone.out_count
                                    
                                    if detections.tracker_id is not None and crossed_out_mask.any():
                                        crossed_out_ids = detections.tracker_id[crossed_out_mask]
                                        for track_id in crossed_out_ids:
                                            class_id = current_frame_classes.get(track_id, track_to_class.get(track_id))
                                            if class_id is not None:
                                                if class_id not in line_data['class_counts']:
                                                    line_data['class_counts'][class_id] = 0
                                                line_data['class_counts'][class_id] += 1
                                                
                                                class_name = class_names.get(class_id, f"class_{class_id}")
                                                if class_name not in total_vehicles_counted:
                                                    total_vehicles_counted[class_name] = 0
                                                total_vehicles_counted[class_name] += 1
                                                
                                                if summary_file is not None:
                                                    write_summary(summary_file, line_zones, class_names)
                                                
                                                if congestion_estimator is not None and congestion_mode == 'timebased':
                                                    congestion_estimator.add_vehicle(class_name, current_time)
                                    
                                    if csv_writer and detections.tracker_id is not None:
                                        if crossed_in_mask.any():
                                            crossed_in_ids = detections.tracker_id[crossed_in_mask]
                                            for track_id in crossed_in_ids:
                                                class_id = current_frame_classes.get(track_id, track_to_class.get(track_id, -1))
                                                class_name = class_names.get(class_id, f"class_{class_id}") if class_id >= 0 else "unknown"
                                                csv_writer.writerow([
                                                    format_video_timestamp(frame_count, fps),
                                                    frame_count,
                                                    line_data['name'],
                                                    track_id,
                                                    class_id,
                                                    class_name,
                                                    'in',
                                                    line_data['in_count'],
                                                    line_data['out_count']
                                                ])
                                        
                                        if crossed_out_mask.any():
                                            crossed_out_ids = detections.tracker_id[crossed_out_mask]
                                            for track_id in crossed_out_ids:
                                                class_id = current_frame_classes.get(track_id, track_to_class.get(track_id, -1))
                                                class_name = class_names.get(class_id, f"class_{class_id}") if class_id >= 0 else "unknown"
                                                csv_writer.writerow([
                                                    format_video_timestamp(frame_count, fps),
                                                    frame_count,
                                                    line_data['name'],
                                                    track_id,
                                                    class_id,
                                                    class_name,
                                                    'out',
                                                    line_data['in_count'],
                                                    line_data['out_count']
                                                ])
                            
                            if congestion_estimator is not None:
                                import time
                                import json as json_module
                                current_time = time.time()
                                
                                if congestion_mode == 'realtime':
                                    current_vehicle_counts = {}
                                    if detections.tracker_id is not None:
                                        for j in range(len(detections)):
                                            category_id = int(detections.class_id[j])
                                            class_name = class_names.get(category_id, f"class_{category_id}")
                                            if class_name not in current_vehicle_counts:
                                                current_vehicle_counts[class_name] = 0
                                            current_vehicle_counts[class_name] += 1
                                    congestion_status = congestion_estimator.get_congestion_status(current_vehicle_counts)
                                    los_vehicles = current_vehicle_counts  # Vehicles in current frame
                                else:
                                    congestion_status = congestion_estimator.get_congestion_status(current_time)
                                    los_vehicles = congestion_status['vehicle_counts']  # Vehicles in time interval
                                
                                # Write congestion data to CSV
                                if congestion_status:
                                    los_vehicles_total = sum(los_vehicles.values()) if los_vehicles else 0
                                    los_vehicles_breakdown = json_module.dumps(los_vehicles if los_vehicles else {})
                                    vehicles_counted_total = sum(total_vehicles_counted.values()) if total_vehicles_counted else 0
                                    vehicles_counted_breakdown = json_module.dumps(total_vehicles_counted if total_vehicles_counted else {})
                                    
                                    # Log to CSV every 10 frames to reduce file size
                                    if congestion_csv_writer and frame_count % 10 == 0:
                                        congestion_csv_writer.writerow([
                                            format_video_timestamp(frame_count, fps),
                                            frame_count,
                                            f"{congestion_status['volume']:.2f}",
                                            f"{congestion_status['capacity']:.2f}",
                                            f"{congestion_status['vc_ratio']:.4f}",
                                            congestion_status['los'],
                                            los_vehicles_total,
                                            los_vehicles_breakdown,
                                            vehicles_counted_total,
                                            vehicles_counted_breakdown
                                        ])
                            
                            annotated_frame = frame_item.copy()
                            if line_zones:
                                for idx, line_data in enumerate(line_zones):
                                    start_point = tuple(line_data['start'].astype(int))
                                    end_point = tuple(line_data['end'].astype(int))
                                    cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 4)
                            
                            if trace_annotator is not None and len(detections) > 0:
                                annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
                            
                            annotated_frame = draw_tracked_boxes(annotated_frame, detections, class_names, getattr(args, 'border_thickness', 3))
                        else:
                            annotated_frame = draw_cv2(frame_item.copy(), labels, boxes, scores, 
                                                       args.threshold, class_names, class_id_mapping, 
                                                       getattr(args, 'border_thickness', 3))
                        
                        output_frame = annotated_frame
                        if line_zones and congestion_estimator is not None:
                            output_frame = cv2.copyMakeBorder(annotated_frame, 0, 0, 0, sidebar_width, cv2.BORDER_CONSTANT, value=(40, 40, 40))
                            output_frame = draw_sidebar_panel(output_frame, line_zones, class_names, congestion_status, congestion_mode, sidebar_width)
                        
                        if out is not None:
                            out.write(output_frame)
                        
                        if args.display:
                            cv2.imshow('RT-DETR Video Inference - Press Q to quit', output_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("\nStopping inference (user pressed 'q')")
                                break
                        
                        pbar.update(1)
                
                break
            
            if batch_size > 1:
                # Accumulate RAW frames first (defer preprocessing until batch is full)
                frame_buffer.append(frame)
                
                # Process batch when buffer is full
                if len(frame_buffer) >= batch_size:
                    # Batch preprocess all frames at once
                    batch_tensors = []
                    batch_orig_sizes = []
                    
                    for frame_item in frame_buffer:
                        # Convert BGR to RGB for PIL
                        frame_rgb = cv2.cvtColor(frame_item, cv2.COLOR_BGR2RGB)
                        im_pil = Image.fromarray(frame_rgb)
                        w, h = im_pil.size
                        
                        # Transform and prepare tensor (still on CPU for now)
                        im_tensor = transforms(im_pil)
                        batch_tensors.append(im_tensor)
                        batch_orig_sizes.append(torch.tensor([w, h]))
                    
                    # Stack into batch and transfer to GPU ONCE
                    batch_frames = torch.stack(batch_tensors).to(args.device)
                    batch_orig_sizes = torch.stack(batch_orig_sizes).to(args.device)
                    # Already batched above - no need to concatenate
                    
                    # Run batched inference
                    try:
                        with autocast('cuda'):
                            batch_outputs = model(batch_frames, batch_orig_sizes)
                    except TypeError:
                        with autocast():
                            batch_outputs = model(batch_frames, batch_orig_sizes)
                    
                    batch_labels, batch_boxes, batch_scores = batch_outputs
                    
                    # Process each frame in the batch (sequentially for tracking)
                    for i, frame_item in enumerate(frame_buffer):
                        # Increment frame count first so it's correct when used below
                        frame_count += 1
                        
                        labels = [batch_labels[i]]
                        boxes = [batch_boxes[i]]
                        scores = [batch_scores[i]]
                        
                        # Process detections with tracking if enabled
                        if tracker is not None:
                            detections = convert_to_supervision_detections(labels, boxes, scores, args.threshold, class_id_mapping)
                            detections = tracker.update_with_detections(detections)
                            
                            # Process line crossing detection for counting
                            if line_zones:
                                import time
                                current_time = time.time()
                                
                                # Build current frame's track-to-class mapping
                                current_frame_classes = {}
                                if detections.tracker_id is not None:
                                    for j in range(len(detections)):
                                        track_id = detections.tracker_id[j]
                                        class_id = int(detections.class_id[j])
                                        current_frame_classes[track_id] = class_id
                                        track_to_class[track_id] = class_id
                                
                                for idx, line_data in enumerate(line_zones):
                                    line_zone = line_data['zone']
                                    crossed_in_mask, crossed_out_mask = line_zone.trigger(detections)
                                    line_data['in_count'] = line_zone.in_count
                                    line_data['out_count'] = line_zone.out_count
                                    
                                    if detections.tracker_id is not None and crossed_out_mask.any():
                                        crossed_out_ids = detections.tracker_id[crossed_out_mask]
                                        for track_id in crossed_out_ids:
                                            class_id = current_frame_classes.get(track_id, track_to_class.get(track_id))
                                            if class_id is not None:
                                                if class_id not in line_data['class_counts']:
                                                    line_data['class_counts'][class_id] = 0
                                                line_data['class_counts'][class_id] += 1
                                                
                                                class_name = class_names.get(class_id, f"class_{class_id}")
                                                if class_name not in total_vehicles_counted:
                                                    total_vehicles_counted[class_name] = 0
                                                total_vehicles_counted[class_name] += 1
                                                
                                                if summary_file is not None:
                                                    write_summary(summary_file, line_zones, class_names)
                                                
                                                if congestion_estimator is not None and congestion_mode == 'timebased':
                                                    congestion_estimator.add_vehicle(class_name, current_time)
                                    
                                    # Record crossing events to CSV
                                    if csv_writer and detections.tracker_id is not None:
                                        if crossed_in_mask.any():
                                            crossed_in_ids = detections.tracker_id[crossed_in_mask]
                                            for track_id in crossed_in_ids:
                                                class_id = current_frame_classes.get(track_id, track_to_class.get(track_id, -1))
                                                class_name = class_names.get(class_id, f"class_{class_id}") if class_id >= 0 else "unknown"
                                                csv_writer.writerow([
                                                    format_video_timestamp(frame_count, fps),
                                                    frame_count,
                                                    line_data['name'],
                                                    track_id,
                                                    class_id,
                                                    class_name,
                                                    'in',
                                                    line_data['in_count'],
                                                    line_data['out_count']
                                                ])
                                        
                                        if crossed_out_mask.any():
                                            crossed_out_ids = detections.tracker_id[crossed_out_mask]
                                            for track_id in crossed_out_ids:
                                                class_id = current_frame_classes.get(track_id, track_to_class.get(track_id, -1))
                                                class_name = class_names.get(class_id, f"class_{class_id}") if class_id >= 0 else "unknown"
                                                csv_writer.writerow([
                                                    format_video_timestamp(frame_count, fps),
                                                    frame_count,
                                                    line_data['name'],
                                                    track_id,
                                                    class_id,
                                                    class_name,
                                                    'out',
                                                    line_data['in_count'],
                                                    line_data['out_count']
                                                ])
                            
                            # Process congestion estimation
                            if congestion_estimator is not None:
                                import time
                                import json as json_module
                                current_time = time.time()
                                
                                if congestion_mode == 'realtime':
                                    current_vehicle_counts = {}
                                    if detections.tracker_id is not None:
                                        for j in range(len(detections)):
                                            category_id = int(detections.class_id[j])
                                            class_name = class_names.get(category_id, f"class_{category_id}")
                                            if class_name not in current_vehicle_counts:
                                                current_vehicle_counts[class_name] = 0
                                            current_vehicle_counts[class_name] += 1
                                    congestion_status = congestion_estimator.get_congestion_status(current_vehicle_counts)
                                    los_vehicles = current_vehicle_counts  # Vehicles in current frame
                                else:
                                    congestion_status = congestion_estimator.get_congestion_status(current_time)
                                    los_vehicles = congestion_status['vehicle_counts']  # Vehicles in time interval
                                
                                # Write congestion data to CSV
                                if congestion_status:
                                    los_vehicles_total = sum(los_vehicles.values()) if los_vehicles else 0
                                    los_vehicles_breakdown = json_module.dumps(los_vehicles if los_vehicles else {})
                                    vehicles_counted_total = sum(total_vehicles_counted.values()) if total_vehicles_counted else 0
                                    vehicles_counted_breakdown = json_module.dumps(total_vehicles_counted if total_vehicles_counted else {})
                                    
                                    # Log to CSV every 10 frames to reduce file size
                                    if congestion_csv_writer and frame_count % 10 == 0:
                                        congestion_csv_writer.writerow([
                                            format_video_timestamp(frame_count, fps),
                                            frame_count,
                                            f"{congestion_status['volume']:.2f}",
                                            f"{congestion_status['capacity']:.2f}",
                                            f"{congestion_status['vc_ratio']:.4f}",
                                            congestion_status['los'],
                                            los_vehicles_total,
                                            los_vehicles_breakdown,
                                            vehicles_counted_total,
                                            vehicles_counted_breakdown
                                        ])
                            
                            # Draw tracked boxes with trajectories
                            annotated_frame = frame_item.copy()
                            if line_zones:
                                for idx, line_data in enumerate(line_zones):
                                    start_point = tuple(line_data['start'].astype(int))
                                    end_point = tuple(line_data['end'].astype(int))
                                    cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 4)
                            
                            if trace_annotator is not None and len(detections) > 0:
                                annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
                            
                            annotated_frame = draw_tracked_boxes(annotated_frame, detections, class_names, getattr(args, 'border_thickness', 3))
                        else:
                            # No tracking - just draw boxes
                            annotated_frame = draw_cv2(frame_item.copy(), labels, boxes, scores, 
                                                       args.threshold, class_names, class_id_mapping, 
                                                       getattr(args, 'border_thickness', 3))
                        
                        # Add sidebar if counting is enabled
                        output_frame = annotated_frame
                        if line_zones and congestion_estimator is not None:
                            output_frame = cv2.copyMakeBorder(annotated_frame, 0, 0, 0, sidebar_width, cv2.BORDER_CONSTANT, value=(40, 40, 40))
                            output_frame = draw_sidebar_panel(output_frame, line_zones, class_names, congestion_status, congestion_mode, sidebar_width)
                        
                        if out is not None:
                            out.write(output_frame)
                        
                        if args.display:
                            cv2.imshow('RT-DETR Video Inference - Press Q to quit', output_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("\nStopping inference (user pressed 'q')")
                                cap.release()
                                if out is not None:
                                    out.release()
                                pbar.close()
                                cv2.destroyAllWindows()
                                return
                        
                        pbar.update(1)
                    
                    # Clear buffer
                    frame_buffer = []
                    
                    # Clear GPU cache periodically
                    if frame_count % 100 == 0:
                        torch.cuda.empty_cache()
                
                continue
            
            # Single frame processing (batch_size == 1) or fallback
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(frame_rgb)
            w, h = im_pil.size
            
            # Prepare input tensor
            orig_size = torch.tensor([w, h])[None].to(args.device)
            im_data = transforms(im_pil)[None].to(args.device)
            
            # Run inference
            try:
                with autocast('cuda'):
                    output = model(im_data, orig_size)
            except TypeError:
                with autocast():
                    output = model(im_data, orig_size)
            
            labels, boxes, scores = output

            # Process detections with tracking if enabled
            if tracker is not None:
                # Convert to supervision format (with class remapping)
                detections = convert_to_supervision_detections(labels, boxes, scores, args.threshold, class_id_mapping)

                # Update tracker
                detections = tracker.update_with_detections(detections)

                # Process line crossing detection for counting
                if line_zones:
                    import time
                    current_time = time.time()

                    # Build current frame's track-to-class mapping (most accurate)
                    current_frame_classes = {}
                    if detections.tracker_id is not None:
                        for i in range(len(detections)):
                            track_id = detections.tracker_id[i]
                            class_id = int(detections.class_id[i])
                            current_frame_classes[track_id] = class_id
                            # Also update persistent mapping
                            track_to_class[track_id] = class_id

                    for idx, line_data in enumerate(line_zones):
                        line_zone = line_data['zone']

                        # Trigger line zone with detections - returns boolean arrays
                        crossed_in_mask, crossed_out_mask = line_zone.trigger(detections)

                        # Update our tracking counts
                        line_data['in_count'] = line_zone.in_count
                        line_data['out_count'] = line_zone.out_count

                        # Get track IDs for objects that crossed OUT
                        if detections.tracker_id is not None and crossed_out_mask.any():
                            crossed_out_ids = detections.tracker_id[crossed_out_mask]

                            # Update per-class counts for OUT crossings only
                            for track_id in crossed_out_ids:
                                # Prioritize current frame data, fall back to persistent mapping
                                class_id = None
                                if track_id in current_frame_classes:
                                    class_id = current_frame_classes[track_id]
                                elif track_id in track_to_class:
                                    class_id = track_to_class[track_id]

                                if class_id is not None:
                                    if class_id not in line_data['class_counts']:
                                        line_data['class_counts'][class_id] = 0
                                    line_data['class_counts'][class_id] += 1

                                    # Update cumulative vehicle count (for congestion CSV)
                                    class_name = class_names.get(class_id, f"class_{class_id}")
                                    if class_name not in total_vehicles_counted:
                                        total_vehicles_counted[class_name] = 0
                                    total_vehicles_counted[class_name] += 1

                                    # Update summary file in real-time
                                    if summary_file is not None:
                                        write_summary(summary_file, line_zones, class_names)

                                    # Update congestion estimator if enabled (time-based mode only)
                                    if congestion_estimator is not None and congestion_mode == 'timebased':
                                        congestion_estimator.add_vehicle(class_name, current_time)

                        # Record crossing events to CSV
                        if csv_writer and detections.tracker_id is not None:
                            # Log IN crossings
                            if crossed_in_mask.any():
                                crossed_in_ids = detections.tracker_id[crossed_in_mask]
                                for track_id in crossed_in_ids:
                                    # Prioritize current frame data, fall back to persistent mapping
                                    class_id = current_frame_classes.get(track_id, track_to_class.get(track_id, -1))
                                    class_name = class_names.get(class_id, f"class_{class_id}") if class_id >= 0 else "unknown"
                                    csv_writer.writerow([
                                        format_video_timestamp(frame_count, fps),
                                        frame_count,
                                        line_data['name'],
                                        track_id,
                                        class_id,
                                        class_name,
                                        'in',
                                        line_data['in_count'],
                                        line_data['out_count']
                                    ])

                            # Log OUT crossings
                            if crossed_out_mask.any():
                                crossed_out_ids = detections.tracker_id[crossed_out_mask]
                                for track_id in crossed_out_ids:
                                    # Prioritize current frame data, fall back to persistent mapping
                                    class_id = current_frame_classes.get(track_id, track_to_class.get(track_id, -1))
                                    class_name = class_names.get(class_id, f"class_{class_id}") if class_id >= 0 else "unknown"
                                    csv_writer.writerow([
                                        format_video_timestamp(frame_count, fps),
                                        frame_count,
                                        line_data['name'],
                                        track_id,
                                        class_id,
                                        class_name,
                                        'out',
                                        line_data['in_count'],
                                        line_data['out_count']
                                    ])

                # Draw tracked boxes
                annotated_frame = frame.copy()

                # Draw counting lines (without default text)
                if line_zones:
                    for idx, line_data in enumerate(line_zones):
                        # Draw line manually without text annotations or box
                        start_point = tuple(line_data['start'].astype(int))
                        end_point = tuple(line_data['end'].astype(int))
                        cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 4)


                # Display traffic congestion information
                if congestion_estimator is not None:
                    import time
                    current_time = time.time()
                    
                    # Get congestion status based on mode
                    if congestion_mode == 'realtime':
                        # Count current vehicles in frame by class
                        current_vehicle_counts = {}
                        if detections.tracker_id is not None:
                            for i in range(len(detections)):
                                category_id = int(detections.class_id[i])
                                class_name = class_names.get(category_id, f"class_{category_id}")
                                if class_name not in current_vehicle_counts:
                                    current_vehicle_counts[class_name] = 0
                                current_vehicle_counts[class_name] += 1
                        
                        congestion_status = congestion_estimator.get_congestion_status(current_vehicle_counts)
                        los_vehicles = current_vehicle_counts  # Vehicles used for LOS (current frame)
                        
                    else:  # timebased
                        congestion_status = congestion_estimator.get_congestion_status(current_time)
                        los_vehicles = congestion_status['vehicle_counts']  # Vehicles used for LOS (interval)
                    
                    # Get vehicles used for LOS computation
                    los_vehicles_total = sum(los_vehicles.values()) if los_vehicles else 0
                    
                    # Get total vehicles counted (from counting lines)
                    vehicles_counted_total = sum(total_vehicles_counted.values())
                    
                    # Format breakdowns as JSON
                    import json as json_module
                    los_vehicles_breakdown = json_module.dumps(los_vehicles if los_vehicles else {})
                    vehicles_counted_breakdown = json_module.dumps(total_vehicles_counted if total_vehicles_counted else {})
                    
                    # Log to CSV every 10 frames to reduce file size
                    if congestion_csv_writer and frame_count % 10 == 0:
                        congestion_csv_writer.writerow([
                            format_video_timestamp(frame_count, fps),
                            frame_count,
                            f"{congestion_status['volume']:.2f}",
                            f"{congestion_status['capacity']:.2f}",
                            f"{congestion_status['vc_ratio']:.4f}",
                            congestion_status['los'],
                            los_vehicles_total,
                            los_vehicles_breakdown,
                            vehicles_counted_total,
                            vehicles_counted_breakdown
                        ])

                # Draw trajectories (so they appear behind boxes)
                if trace_annotator is not None and len(detections) > 0:
                    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

                # Draw tracked boxes with IDs
                annotated_frame = draw_tracked_boxes(annotated_frame, detections, class_names, getattr(args, 'border_thickness', 3))
            else:
                # Original drawing without tracking (with class remapping)
                annotated_frame = draw_cv2(frame.copy(), labels, boxes, scores, args.threshold, class_names, class_id_mapping, getattr(args, 'border_thickness', 3))
            
            # Add sidebar (LOS + counting) if enabled
            output_frame = annotated_frame
            congestion_status = None
            if line_zones:
                # Get latest congestion status if available
                if congestion_estimator is not None:
                    import time
                    current_time = time.time()
                    if congestion_mode == 'realtime':
                        current_vehicle_counts = {}
                        if detections.tracker_id is not None:
                            for i in range(len(detections)):
                                category_id = int(detections.class_id[i])
                                class_name = class_names.get(category_id, f"class_{category_id}")
                                if class_name not in current_vehicle_counts:
                                    current_vehicle_counts[class_name] = 0
                                current_vehicle_counts[class_name] += 1
                        congestion_status = congestion_estimator.get_congestion_status(current_vehicle_counts)
                    else:  # timebased
                        congestion_status = congestion_estimator.get_congestion_status(current_time)

                # Create output frame with sidebar
                output_frame = cv2.copyMakeBorder(annotated_frame, 0, 0, 0, sidebar_width, cv2.BORDER_CONSTANT, value=(40, 40, 40))
                output_frame = draw_sidebar_panel(output_frame, line_zones, class_names, congestion_status, congestion_mode, sidebar_width)

            # Write frame to output video (if saving)
            if out is not None:
                out.write(output_frame)

            # Display frame in real-time only when explicitly enabled.
            if args.display:
                cv2.imshow('RT-DETR Video Inference - Press Q to quit', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\\nStopping inference (user pressed 'q')")
                    break
            
            frame_count += 1
            pbar.update(1)
            
            # Clear GPU cache periodically
            if frame_count % 100 == 0:
                torch.cuda.empty_cache()
    
    pbar.close()
    cap.release()
    if out is not None:
        out.release()
        print(f"Video processing complete. Output saved to: {args.output}")
    else:
        print("Video processing complete. No output file saved (display only).")

    # Close CSV file and print counting summary
    if csv_file is not None:
        csv_file.close()
        print(f"\nCounting data saved to CSV file")

    if summary_file is not None:
        summary_file.close()
        print(f"Counting summary saved to TXT file")

    if line_zones:
        print("\n=== Counting Summary (OUT counts only) ===")

        for line_data in line_zones:
            line_name = line_data['name']
            class_counts = line_data['class_counts']
            total_out = sum(class_counts.values())

            print(f"\n{line_name}:")
            print(f"  Total OUT: {total_out}")

            if class_counts:
                print(f"  Per-class breakdown:")
                for class_id in sorted(class_counts.keys()):
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    count = class_counts[class_id]
                    print(f"    {class_name}: {count}")
            else:
                print(f"  No objects counted yet")

        print("\n==========================================\n")
    
    # Close congestion CSV file and print congestion summary
    if congestion_csv_file is not None:
        congestion_csv_file.close()
        print(f"\nCongestion data saved to CSV file")
    
    if congestion_estimator is not None:
        import time
        current_time = time.time()
        
        if congestion_mode == 'realtime':
            # For real-time mode, get final status from last frame detections
            final_vehicle_counts = {}
            if detections is not None and detections.tracker_id is not None:
                for i in range(len(detections)):
                    category_id = int(detections.class_id[i])
                    class_name = class_names.get(category_id, f"class_{category_id}")
                    if class_name not in final_vehicle_counts:
                        final_vehicle_counts[class_name] = 0
                    final_vehicle_counts[class_name] += 1
            
            final_status = congestion_estimator.get_congestion_status(final_vehicle_counts)
            
            print("\n=== Traffic Congestion Summary (REAL-TIME MODE) ===")
            print(f"Capacity: {final_status['capacity']:.2f} PCE")
            print(f"Final Frame Volume: {final_status['volume']:.2f} PCE")
            print(f"Final V/C Ratio: {final_status['vc_ratio']:.4f}")
            print(f"Level of Service: {final_status['los']} - {RealtimeTrafficCongestionEstimator.get_los_description(final_status['los'])}")
            
            if final_status['vehicle_counts']:
                print("\nVehicle counts in final frame:")
                for class_name, count in sorted(final_status['vehicle_counts'].items()):
                    multiplier = RealtimeTrafficCongestionEstimator.VEHICLE_MULTIPLIERS.get(class_name, 1.0)
                    pce_contribution = count * multiplier
                    print(f"  {class_name}: {count} vehicles (x{multiplier} = {pce_contribution:.1f} PCE)")
            else:
                print("\nNo vehicles in the final frame")
            print("\n===================================================\n")
            
        else:  # timebased
            final_status = congestion_estimator.get_congestion_status(current_time)
            
            print("\n=== Traffic Congestion Summary (TIME-BASED MODE) ===")
            print(f"Road Width: {final_status['road_width']} meters")
            print(f"Hourly Capacity: {final_status['hourly_capacity']} vehicles/hour")
            print(f"Time Interval: {final_status['time_interval']} minutes")
            print(f"Current Volume: {final_status['volume']:.2f} PCE")
            print(f"Interval Capacity: {final_status['capacity']:.2f} PCE")
            print(f"V/C Ratio: {final_status['vc_ratio']:.4f}")
            print(f"Level of Service: {final_status['los']} - {TrafficCongestionEstimator.get_los_description(final_status['los'])}")
            
            if final_status['vehicle_counts']:
                print("\nVehicle counts in last interval:")
                for class_name, count in sorted(final_status['vehicle_counts'].items()):
                    multiplier = TrafficCongestionEstimator.VEHICLE_MULTIPLIERS.get(class_name, 1.0)
                    pce_contribution = count * multiplier
                    print(f"  {class_name}: {count} vehicles (x{multiplier} = {pce_contribution:.1f} PCE)")
            else:
                print("\nNo vehicles in the last time interval")
            print("\n====================================================\n")

    if args.display:
        cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")

def main(args, ):
    """main
    """
    # Check if tracking is enabled with single image mode
    if hasattr(args, 'tracking') and args.tracking and not args.video_file:
        print("Warning: --tracking flag is only supported for video inference. Ignoring tracking for image inference.")
        args.tracking = False

    if args.video_file:
        process_video(args)
        return
        
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    
    # Calculate and print model statistics
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {n_parameters:,}')
    
    try:
        gflops = calculate_gflops(model)
        if gflops is not None:
            print(f'GFLOPs: {gflops:.2f}')
        else:
            print('GFLOPs: calculation failed')
    except Exception as e:
        print(f'GFLOPs: calculation error - {e}')
    
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)
    
    transforms = T.Compose([
        T.Resize((640, 640)),  
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)
    if args.sliced:
        num_boxes = args.numberofboxes
        
        aspect_ratio = w / h
        num_cols = int(np.sqrt(num_boxes * aspect_ratio)) 
        num_rows = int(num_boxes / num_cols)
        slice_height = h // num_rows
        slice_width = w // num_cols
        overlap_ratio = 0.2
        slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
        predictions = []
        for i, slice_img in enumerate(slices):
            slice_tensor = transforms(slice_img)[None].to(args.device)
            with autocast():  # Use AMP for each slice
                output = model(slice_tensor, torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device))
            torch.cuda.empty_cache() 
            labels, boxes, scores = output
            
            labels = labels.cpu().detach().numpy()
            boxes = boxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            predictions.append((labels, boxes, scores))
        
        merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height)
        labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
    else:
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        
    draw([im_pil], labels, boxes, scores, 0.6)
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, help='Path to checkpoint file')
    parser.add_argument('-f', '--im-file', type=str, help='Path to input image (for image inference)')
    parser.add_argument('-v', '--video-file', type=str, help='Path to input video (for video inference)')
    parser.add_argument('-a', '--ann-file', type=str, help='Path to annotation file for class names')
    parser.add_argument('-o', '--output', type=str, help='Path to output video file')
    parser.add_argument('-s', '--sliced', type=bool, default=False, help='Enable sliced inference')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25, help='Number of boxes for sliced inference')
    parser.add_argument('-t', '--threshold', type=float, default=0.6, help='Detection confidence threshold')
    parser.add_argument('--display', action='store_true', help='Display video frames during processing')
    parser.add_argument('-bt', '--border-thickness', type=int, default=3, help='Thickness of bounding box borders')
    parser.add_argument('--tracking', action='store_true', help='Enable ByteTrack tracking (only for video inference)')
    parser.add_argument('--counting-config', type=str, help='Path to counting configuration JSON file (requires --tracking)')
    
    # Traffic congestion estimation arguments
    parser.add_argument('--congestion', action='store_true', help='Enable traffic congestion estimation (requires --tracking)')
    parser.add_argument('--road-length', type=float, help='Road length in kilometers (required for real-time mode)')
    parser.add_argument('--lanes', type=int, help='Number of lanes (required for real-time mode)')
    parser.add_argument('--road-width', type=float, help='Road width in meters (for time-based mode)')
    parser.add_argument('--num-lanes', type=int, default=1, help='Number of lanes for time-based mode (default: 1). Multiplies hourly capacity based on road width.')
    parser.add_argument('--time-interval', type=float, default=15.0, help='Time interval in minutes for time-based congestion calculation (default: 15.0)')
    
    # Batch processing argument for faster video processing
    parser.add_argument('--batch-size', type=int, default=1, help='Number of frames to process simultaneously (default: 1). Higher values utilize GPU better for faster-than-realtime processing on saved videos. Recommended: 4-8 for most GPUs.')
    
    args = parser.parse_args()
    
    if not args.video_file and not args.im_file:
        parser.error('Either --video-file or --im-file must be specified')
    
    # Validate congestion arguments
    if args.congestion:
        if not args.tracking:
            parser.error('--congestion requires --tracking to be enabled')
        
        # Must specify either --road-length/--lanes (real-time) or --road-width (time-based)
        if (args.road_length is None or args.lanes is None) and args.road_width is None:
            parser.error('--congestion requires either --road-length and --lanes (real-time mode) or --road-width (time-based mode)')
        
        if args.road_width is not None and (args.road_length is not None or args.lanes is not None):
            parser.error('Cannot use both real-time (--road-length/--lanes) and time-based (--road-width) modes')
        
        if args.road_length is not None and args.road_length <= 0:
            parser.error('--road-length must be a positive number')
        
        if args.lanes is not None and args.lanes <= 0:
            parser.error('--lanes must be a positive integer')
        
        if args.road_width is not None and args.road_width <= 0:
            parser.error('--road-width must be a positive number')
        
        if args.num_lanes is not None and args.num_lanes <= 0:
            parser.error('--num-lanes must be a positive integer')
        
        if args.road_width is not None and args.time_interval <= 0:
            parser.error('--time-interval must be a positive number')
    
    main(args)











