"""
Visualization utilities for saving inference results during validation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict
import torchvision.transforms.functional as F


def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor back to [0, 255] range for proper display
    Args:
        image_tensor: Normalized image tensor [C, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
    Returns:
        Denormalized image as numpy array [H, W, C] in uint8 [0, 255] range
    """
    image = image_tensor.clone().cpu().float()
    
    # Denormalize: reverse the normalization process
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    
    # Clamp to [0, 1] range
    image = image.clamp(0, 1)
    
    # Convert to [0, 255] range and uint8 for proper matplotlib display
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    
    return image


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes [x1, y1, x2, y2]
    """
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x2_min < x1_max or y2_min < y1_max:
        return 0.0
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def evaluate_image_quality(pred, target, iou_threshold=0.5, conf_threshold=0.3):
    """
    Evaluate prediction quality for a single image
    Returns dict with metrics indicating if this is a "bad" prediction
    """
    metrics = {
        'is_bad': False,
        'reason': [],
        'score': 1.0,  # 1.0 is perfect, 0.0 is worst
        'num_gt': 0,
        'num_pred': 0,
        'num_matched': 0,
        'precision': 0.0,
        'recall': 0.0
    }
    
    # Get ground truth boxes
    if 'boxes' not in target or len(target['boxes']) == 0:
        metrics['num_gt'] = 0
        # No ground truth - check if we have false positives
        if 'boxes' in pred and len(pred['boxes']) > 0:
            pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']
            high_conf_preds = (pred_scores >= conf_threshold).sum()
            if high_conf_preds > 0:
                metrics['is_bad'] = True
                metrics['reason'].append(f'False positives: {high_conf_preds} predictions but no GT')
                metrics['score'] = 0.0
        return metrics
    
    gt_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
    gt_labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else target['labels']
    metrics['num_gt'] = len(gt_boxes)
    
    # Get predictions with high confidence
    if 'boxes' not in pred or len(pred['boxes']) == 0:
        metrics['num_pred'] = 0
        metrics['is_bad'] = True
        metrics['reason'].append(f'Missed all {metrics["num_gt"]} GT objects')
        metrics['recall'] = 0.0
        metrics['score'] = 0.0
        return metrics
    
    pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
    pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']
    pred_labels = pred['labels'].cpu().numpy() if torch.is_tensor(pred['labels']) else pred['labels']
    
    # Filter by confidence
    high_conf_mask = pred_scores >= conf_threshold
    pred_boxes = pred_boxes[high_conf_mask]
    pred_scores = pred_scores[high_conf_mask]
    pred_labels = pred_labels[high_conf_mask]
    metrics['num_pred'] = len(pred_boxes)
    
    if len(pred_boxes) == 0:
        metrics['is_bad'] = True
        metrics['reason'].append(f'No confident predictions (GT: {metrics["num_gt"]})')
        metrics['recall'] = 0.0
        metrics['score'] = 0.0
        return metrics
    
    # Match predictions to ground truth
    matched_gt = set()
    matched_pred = set()
    
    for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        best_iou = 0.0
        best_pred_idx = -1
        
        for j, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
            if pred_label == gt_label:  # Same class
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j
        
        if best_iou >= iou_threshold and best_pred_idx not in matched_pred:
            matched_gt.add(i)
            matched_pred.add(best_pred_idx)
    
    metrics['num_matched'] = len(matched_gt)
    metrics['precision'] = len(matched_pred) / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    metrics['recall'] = len(matched_gt) / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    
    # Calculate overall score (F1-like)
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    else:
        metrics['score'] = 0.0
    
    # Determine if "bad" based on multiple criteria
    if metrics['recall'] < 0.5:
        metrics['is_bad'] = True
        metrics['reason'].append(f'Low recall: {metrics["recall"]:.2f} ({metrics["num_matched"]}/{metrics["num_gt"]} matched)')
    
    if metrics['precision'] < 0.5 and len(pred_boxes) > 2:
        metrics['is_bad'] = True
        metrics['reason'].append(f'Low precision: {metrics["precision"]:.2f} (many false positives)')
    
    if metrics['num_matched'] == 0 and metrics['num_gt'] > 0:
        metrics['is_bad'] = True
        metrics['reason'].append('Complete miss: No GT objects detected')
    
    false_positives = len(pred_boxes) - len(matched_pred)
    if false_positives >= metrics['num_gt'] and false_positives > 3:
        metrics['is_bad'] = True
        metrics['reason'].append(f'Too many false positives: {false_positives}')
    
    return metrics


def denormalize_image(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    Args:
        image_tensor: Normalized image tensor [C, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
    Returns:
        Denormalized image as numpy array [H, W, C]
    """
    image = image_tensor.clone().cpu()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return image


def draw_boxes_on_image(ax, boxes, labels, scores, color='red', label_prefix='Pred', class_names=None, coco_api=None):
    """
    Draw bounding boxes on matplotlib axis
    Args:
        ax: Matplotlib axis
        boxes: Bounding boxes [N, 4] in format (x1, y1, x2, y2)
        labels: Class labels [N]
        scores: Confidence scores [N]
        color: Box color
        label_prefix: Prefix for label text
        class_names: Optional list of class names
        coco_api: Optional COCO API object for getting category names
    """
    if boxes is None or len(boxes) == 0:
        return
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle patch - increased linewidth for better visibility
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=1, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label - try to get class name from COCO API if available
        class_name = None
        if coco_api and hasattr(coco_api, 'loadCats'):
            try:
                cats = coco_api.loadCats([int(label)])
                if cats:
                    class_name = cats[0]['name']
            except:
                pass
        
        if class_name is None and class_names and int(label) < len(class_names):
            class_name = class_names[int(label)]
        
        if class_name is None:
            class_name = f"Class {int(label)}"
        
        label_text = f"{label_prefix}: {class_name} {score:.2f}"
        ax.text(
            x1, y1 - 5,
            label_text,
            color='white',
            fontsize=12,
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2)
        )


def visualize_predictions(
    images: torch.Tensor,
    predictions: List[Dict],
    targets: List[Dict],
    coco_api,
    output_dir: Path,
    epoch: int,
    metrics: Dict = None,
    max_images: int = 10,
    class_names: List[str] = None,
    save_poor_performers: bool = True
):
    """
    Visualize and save predictions vs ground truth
    Args:
        images: Batch of images [B, C, H, W]
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
        targets: List of target dicts with 'boxes', 'labels', 'image_id'
        coco_api: COCO API object for getting image info
        output_dir: Output directory to save visualizations
        epoch: Current epoch number
        metrics: Optional dict of metrics to display
        max_images: Maximum number of images to visualize
        class_names: Optional list of class names
        save_poor_performers: If True, save poor performing images to separate folder
    """
    # Create visualization directories
    vis_dir = output_dir / f'visualizations_epoch_{epoch:04d}'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    poor_perf_dir = output_dir / f'visualizations_epoch_{epoch:04d}_poor_performance'
    if save_poor_performers:
        poor_perf_dir.mkdir(parents=True, exist_ok=True)
    
    num_images = min(len(images), max_images)
    
    print(f"\n[Visualization] Creating {num_images} visualizations for epoch {epoch} in {vis_dir}")
    if save_poor_performers:
        print(f"[Visualization] Poor performers will be saved to {poor_perf_dir}")
    
    poor_performer_count = 0
    
    for idx in range(num_images):
        image = images[idx]
        pred = predictions[idx]
        target = targets[idx]
        
        # Get image info
        image_id = target['image_id'].item()
        
        # Get image dimensions
        img_h, img_w = image.shape[1], image.shape[2]  # Current image size (e.g., 640x640)
        
        # Get original image size for scaling prediction boxes
        if 'orig_size' in target:
            orig_w, orig_h = target['orig_size'][0].item(), target['orig_size'][1].item()
            scale_w = img_w / orig_w
            scale_h = img_h / orig_h
        else:
            scale_w = 1.0
            scale_h = 1.0
        
        img_info = coco_api.loadImgs(image_id)[0] if coco_api else None
        
        # Denormalize image
        img_np = denormalize_image(image)
        
        # Prepare scaled prediction boxes for quality evaluation
        pred_for_eval = pred.copy() if isinstance(pred, dict) else {}
        if 'boxes' in pred and len(pred['boxes']) > 0:
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            
            # Scale prediction boxes to display coordinates for evaluation
            pred_boxes_scaled = pred_boxes.copy()
            pred_boxes_scaled[:, [0, 2]] *= scale_w
            pred_boxes_scaled[:, [1, 3]] *= scale_h
            
            pred_for_eval = {
                'boxes': pred_boxes_scaled,
                'labels': pred_labels,
                'scores': pred_scores
            }
        
        # Evaluate image quality to determine if it's a poor performer
        quality_metrics = evaluate_image_quality(pred_for_eval, target, iou_threshold=0.5, conf_threshold=0.3)
        is_poor_performer = quality_metrics['is_bad']
        
        # Create figure with two subplots - increased size for better visibility
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))
        
        # Add red border for poor performers
        if is_poor_performer:
            fig.patch.set_edgecolor('red')
            fig.patch.set_linewidth(10)
        
        # Ground Truth subplot
        ax1.imshow(img_np, interpolation='nearest', vmin=0, vmax=255)  # Explicit range for uint8 images
        ax1.set_title(f'Ground Truth (Image ID: {image_id})', fontsize=20, fontweight='bold')
        ax1.axis('off')
        
        # Draw ground truth boxes (already in display coordinates from transforms)
        if 'boxes' in target and len(target['boxes']) > 0:
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            gt_scores = np.ones(len(gt_labels))  # Ground truth has score 1.0
            
            draw_boxes_on_image(
                ax1, gt_boxes, gt_labels, gt_scores,
                color='green', label_prefix='GT', class_names=class_names, coco_api=coco_api
            )
        
        # Prediction subplot
        pred_title = f'Predictions (conf ≥ 0.3)'
        if is_poor_performer:
            pred_title += ' - ⚠️ POOR PERFORMANCE'
        ax2.imshow(img_np, interpolation='nearest', vmin=0, vmax=255)  # Explicit range for uint8 images
        ax2.set_title(pred_title, fontsize=20, fontweight='bold', 
                     color='red' if is_poor_performer else 'black')
        ax2.axis('off')
        
        # Draw prediction boxes (need to scale FROM original size TO display size)
        if 'boxes' in pred and len(pred['boxes']) > 0:
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            
            # Filter predictions by confidence threshold
            confidence_threshold = 0.3  # Show predictions with >30% confidence
            high_conf_mask = pred_scores >= confidence_threshold
            
            if high_conf_mask.sum() > 0:
                # Filter boxes, labels, and scores
                pred_boxes = pred_boxes[high_conf_mask]
                pred_labels = pred_labels[high_conf_mask]
                pred_scores = pred_scores[high_conf_mask]
                
                # Predictions are in original image coordinates, need to scale to display size
                pred_boxes_scaled = pred_boxes.copy()
                pred_boxes_scaled[:, [0, 2]] *= scale_w  # Scale x coordinates  
                pred_boxes_scaled[:, [1, 3]] *= scale_h  # Scale y coordinates
                
                draw_boxes_on_image(
                    ax2, pred_boxes_scaled, pred_labels, pred_scores,
                    color='red', label_prefix='Pred', class_names=class_names, coco_api=coco_api
                )
        
        # Build metrics text
        metrics_text = f"Epoch {epoch}\n"
        
        # Add overall metrics if provided
        if metrics:
            metrics_text += f"Overall mAP@0.5:0.95: {metrics.get('mAP@0.5:0.95', 0):.4f}\n"
            metrics_text += f"Overall mAP@0.5: {metrics.get('mAP@0.5', 0):.4f}\n"
        
        # Add per-image quality metrics
        metrics_text += f"\nPer-Image Quality:\n"
        metrics_text += f"GT Objects: {quality_metrics['num_gt']}\n"
        metrics_text += f"Predictions (conf≥0.3): {quality_metrics['num_pred']}\n"
        metrics_text += f"Matched: {quality_metrics['num_matched']}\n"
        metrics_text += f"Precision: {quality_metrics['precision']:.3f}\n"
        metrics_text += f"Recall: {quality_metrics['recall']:.3f}\n"
        metrics_text += f"Quality Score: {quality_metrics['score']:.3f}"
        
        if is_poor_performer:
            metrics_text += f"\n\n⚠️ ISSUES:\n" + "\n".join(quality_metrics['reason'])
        
        # Add text box with metrics - increased font size
        props = dict(boxstyle='round', 
                    facecolor='lightcoral' if is_poor_performer else 'wheat', 
                    alpha=0.9)
        fig.text(
            0.5, 0.02, metrics_text,
            fontsize=14, ha='center',
            bbox=props
        )
        
        # Add filename if available
        title_text = f"Image: {img_info['file_name']}" if (img_info and 'file_name' in img_info) else f"Image ID: {image_id}"
        fig.suptitle(title_text, fontsize=18, y=0.98)
        
        plt.tight_layout(rect=[0, 0.15, 1, 0.96])
        
        # Save to regular directory with high quality settings
        save_path = vis_dir / f'image_{image_id:012d}.png'
        plt.savefig(save_path, 
                   dpi=300,                    # Higher DPI for better quality (was 150)
                   bbox_inches='tight',
                   format='png',
                   pil_kwargs={'compress_level': 1})  # Less compression (1-9, lower=better quality)
        
        # Also save to poor performers directory if applicable
        if is_poor_performer and save_poor_performers:
            poor_save_path = poor_perf_dir / f'image_{image_id:012d}_poor.png'
            plt.savefig(poor_save_path, 
                       dpi=300,                    # Higher DPI for better quality
                       bbox_inches='tight',
                       format='png',
                       pil_kwargs={'compress_level': 1})  # Less compression
            poor_performer_count += 1
        
        plt.close(fig)
    
    print(f"[Visualization] Saved {num_images} visualization images to {vis_dir}")
    if save_poor_performers:
        print(f"[Visualization] Saved {poor_performer_count} poor performers to {poor_perf_dir}")


def visualize_batch(
    images: torch.Tensor,
    predictions: List[Dict],
    targets: List[Dict],
    coco_api,
    output_dir: Path,
    epoch: int,
    batch_idx: int,
    metrics: Dict = None,
    max_images: int = 4,
    class_names: List[str] = None
):
    """
    Visualize a batch of predictions (lighter version for frequent visualization)
    Args:
        images: Batch of images [B, C, H, W]
        predictions: List of prediction dicts
        targets: List of target dicts
        coco_api: COCO API object
        output_dir: Output directory
        epoch: Current epoch
        batch_idx: Batch index
        metrics: Optional metrics dict
        max_images: Max images to visualize from batch
        class_names: Optional list of class names
    """
    vis_dir = output_dir / f'visualizations_epoch_{epoch:04d}'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    num_images = min(len(images), max_images)
    
    # Create a grid figure - increased size for better visibility
    fig, axes = plt.subplots(2, num_images, figsize=(8*num_images, 16))
    if num_images == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(num_images):
        image = images[idx]
        pred = predictions[idx]
        target = targets[idx]
        
        # Get image info
        image_id = target['image_id'].item()
        
        # Denormalize image
        img_np = denormalize_image(image)
        
        # Ground Truth
        axes[0, idx].imshow(img_np, interpolation='nearest', vmin=0, vmax=255)  # Explicit range for uint8 images
        axes[0, idx].set_title(f'GT (ID: {image_id})', fontsize=14, fontweight='bold')
        axes[0, idx].axis('off')
        
        if 'boxes' in target and len(target['boxes']) > 0:
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            gt_scores = np.ones(len(gt_labels))
            draw_boxes_on_image(
                axes[0, idx], gt_boxes, gt_labels, gt_scores,
                color='green', label_prefix='', class_names=class_names, coco_api=coco_api
            )
        
        # Predictions
        axes[1, idx].imshow(img_np, interpolation='nearest', vmin=0, vmax=255)  # Explicit range for uint8 images
        axes[1, idx].set_title(f'Pred', fontsize=14, fontweight='bold')
        axes[1, idx].axis('off')
        
        if 'boxes' in pred and len(pred['boxes']) > 0:
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            draw_boxes_on_image(
                axes[1, idx], pred_boxes, pred_labels, pred_scores,
                color='red', label_prefix='', class_names=class_names, coco_api=coco_api
            )
    
    plt.suptitle(f'Epoch {epoch} - Batch {batch_idx}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    save_path = vis_dir / f'batch_{batch_idx:04d}.png'
    plt.savefig(save_path, 
               dpi=300,                    # Higher DPI for better quality
               bbox_inches='tight',
               format='png',
               pil_kwargs={'compress_level': 1})  # Less compression
    plt.close(fig)
