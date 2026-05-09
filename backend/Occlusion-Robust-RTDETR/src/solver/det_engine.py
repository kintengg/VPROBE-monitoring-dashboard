"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp
import numpy as np

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
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


def compute_confusion_matrix(predictions, targets, num_classes, iou_threshold=0.5, score_threshold=0.3):
    """
    Compute confusion matrix for object detection
    predictions: list of dicts with 'boxes', 'scores', 'labels'
    targets: list of dicts with 'boxes', 'labels'
    num_classes: number of classes (used for matrix dimensions)
    """
    import numpy as np
    
    # Find the maximum category ID to properly size the confusion matrix
    # COCO categories are 1-indexed (e.g., 1, 2, 3), not 0-indexed
    max_cat_id = 0
    for pred in predictions:
        if len(pred['labels']) > 0:
            max_cat_id = max(max_cat_id, int(pred['labels'].max().item()))
    for target in targets:
        if len(target['labels']) > 0:
            max_cat_id = max(max_cat_id, int(target['labels'].max().item()))
    
    # Size confusion matrix to accommodate maximum category ID
    # For 1-indexed labels (1-6), we need size 7 (indices 0-6), so add 1
    # num_classes represents the count of classes, not the max index
    matrix_size = max(num_classes + 1, max_cat_id + 1)
    
    # Initialize confusion matrix (rows=predicted, cols=ground truth)
    confusion = np.zeros((matrix_size, matrix_size), dtype=np.float32)
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].numpy()
        pred_scores = pred['scores'].numpy()
        pred_labels = pred['labels'].numpy()
        
        gt_boxes = target['boxes'].numpy()
        gt_labels = target['labels'].numpy()
        
        # Filter predictions by score threshold
        valid_mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[valid_mask]
        pred_labels = pred_labels[valid_mask]
        pred_scores = pred_scores[valid_mask]
        
        # Track matched ground truths
        matched_gts = set()
        
        # Sort predictions by score (highest first)
        sorted_indices = np.argsort(-pred_scores)
        
        for pred_idx in sorted_indices:
            pred_box = pred_boxes[pred_idx]
            pred_label = pred_labels[pred_idx]
            
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            best_gt_label = -1
            
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_idx in matched_gts:
                    continue
                    
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt_label = gt_label
            
            # If IoU is above threshold, it's a match
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gts.add(best_gt_idx)
                # Add to confusion matrix
                confusion[int(pred_label), int(best_gt_label)] += 1
            else:
                # False positive - no matching ground truth
                # Count as misclassification to background
                pass
    
    return confusion


def compute_ap_for_subset(predictions, targets, coco_api):
    """
    Compute mAP for a subset of predictions/targets
    """
    import numpy as np
    from collections import defaultdict
    
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0
    
    # Group by image_id
    pred_by_image = defaultdict(lambda: {'boxes': [], 'scores': [], 'labels': []})
    gt_by_image = defaultdict(lambda: {'boxes': [], 'labels': []})
    
    for pred in predictions:
        img_id = pred['image_id'].item()
        pred_by_image[img_id]['boxes'].append(pred['boxes'].numpy())
        pred_by_image[img_id]['scores'].append(pred['scores'].numpy())
        pred_by_image[img_id]['labels'].append(pred['labels'].numpy())
    
    for target in targets:
        img_id = target['image_id'].item()
        gt_by_image[img_id]['boxes'].append(target['boxes'].numpy())
        gt_by_image[img_id]['labels'].append(target['labels'].numpy())
    
    # Simple AP calculation (simplified for speed)
    # For each IoU threshold, compute precision-recall
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    aps = []
    
    for iou_thresh in iou_thresholds:
        all_scores = []
        all_matches = []
        total_gt = 0
        
        for img_id in pred_by_image.keys():
            if img_id not in gt_by_image:
                continue
                
            pred_boxes = np.concatenate(pred_by_image[img_id]['boxes']) if pred_by_image[img_id]['boxes'] else np.array([])
            pred_scores = np.concatenate(pred_by_image[img_id]['scores']) if pred_by_image[img_id]['scores'] else np.array([])
            pred_labels = np.concatenate(pred_by_image[img_id]['labels']) if pred_by_image[img_id]['labels'] else np.array([])
            
            gt_boxes = np.concatenate(gt_by_image[img_id]['boxes']) if gt_by_image[img_id]['boxes'] else np.array([])
            gt_labels = np.concatenate(gt_by_image[img_id]['labels']) if gt_by_image[img_id]['labels'] else np.array([])
            
            total_gt += len(gt_boxes)
            
            matched_gts = set()
            for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
                best_iou = 0
                best_gt = -1
                for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if j in matched_gts or label != gt_label:
                        continue
                    iou = compute_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j
                
                all_scores.append(score)
                if best_iou >= iou_thresh and best_gt >= 0:
                    matched_gts.add(best_gt)
                    all_matches.append(1)
                else:
                    all_matches.append(0)
        
        if len(all_scores) == 0:
            aps.append(0.0)
            continue
        
        # Sort by score
        sorted_indices = np.argsort(-np.array(all_scores))
        matches = np.array(all_matches)[sorted_indices]
        
        # Compute precision-recall
        tp = np.cumsum(matches)
        fp = np.cumsum(1 - matches)
        recall = tp / total_gt if total_gt > 0 else tp * 0
        precision = tp / (tp + fp)
        
        # Compute AP (area under PR curve)
        ap = 0
        for t in np.linspace(0, 1, 101):
            p = precision[recall >= t]
            ap += (p.max() if len(p) > 0 else 0) / 101
        
        aps.append(ap)
    
    return np.mean(aps) if len(aps) > 0 else 0.0


def compute_per_object_occlusion_metrics(all_predictions, all_targets, score_threshold=0.3, iou_threshold=0.5):
    """
    Compute metrics by matching predictions to GTs, then grouping by GT occlusion level.
    This properly handles per-object occlusion by only counting each prediction once,
    assigning it to the occlusion level of the GT it matches (if any).
    """
    import numpy as np
    from collections import defaultdict
    
    # Initialize metrics storage for each occlusion level
    metrics_by_occ = {
        'none': {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0},
        'partial': {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0},
        'heavy': {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0}
    }
    
    # Group by image_id
    pred_by_image = defaultdict(lambda: {'boxes': [], 'scores': [], 'labels': []})
    gt_by_image = defaultdict(lambda: {'boxes': [], 'labels': [], 'occlusions': []})
    
    # Group predictions
    for pred in all_predictions:
        img_id = pred['image_id'].item() if torch.is_tensor(pred['image_id']) else pred['image_id']
        pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
        pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']
        pred_labels = pred['labels'].cpu().numpy() if torch.is_tensor(pred['labels']) else pred['labels']
        
        if len(pred_by_image[img_id]['boxes']) == 0:
            pred_by_image[img_id]['boxes'] = pred_boxes
            pred_by_image[img_id]['scores'] = pred_scores
            pred_by_image[img_id]['labels'] = pred_labels
        else:
            pred_by_image[img_id]['boxes'] = np.concatenate([pred_by_image[img_id]['boxes'], pred_boxes])
            pred_by_image[img_id]['scores'] = np.concatenate([pred_by_image[img_id]['scores'], pred_scores])
            pred_by_image[img_id]['labels'] = np.concatenate([pred_by_image[img_id]['labels'], pred_labels])
    
    # Group targets with occlusion info
    for target in all_targets:
        img_id = target['image_id'].item() if torch.is_tensor(target['image_id']) else target['image_id']
        gt_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
        gt_labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else target['labels']
        
        # Get occlusion levels - should be per-object
        if 'occlusions' in target:
            gt_occlusions = target['occlusions'].cpu().numpy() if torch.is_tensor(target['occlusions']) else target['occlusions']
        elif 'occlusion' in target:
            gt_occlusions = target['occlusion'].cpu().numpy() if torch.is_tensor(target['occlusion']) else target['occlusion']
        else:
            gt_occlusions = np.zeros(len(gt_boxes), dtype=np.int32)
        
        if len(gt_by_image[img_id]['boxes']) == 0:
            gt_by_image[img_id]['boxes'] = gt_boxes
            gt_by_image[img_id]['labels'] = gt_labels
            gt_by_image[img_id]['occlusions'] = gt_occlusions
        else:
            gt_by_image[img_id]['boxes'] = np.concatenate([gt_by_image[img_id]['boxes'], gt_boxes])
            gt_by_image[img_id]['labels'] = np.concatenate([gt_by_image[img_id]['labels'], gt_labels])
            gt_by_image[img_id]['occlusions'] = np.concatenate([gt_by_image[img_id]['occlusions'], gt_occlusions])
    
    # Now match predictions to GTs and assign to occlusion levels
    for img_id in gt_by_image.keys():
        if img_id not in pred_by_image:
            # No predictions for this image - all GTs are false negatives
            gt_occlusions = gt_by_image[img_id]['occlusions']
            for occ in gt_occlusions:
                occ_name = 'heavy' if occ == 2 else ('partial' if occ == 1 else 'none')
                metrics_by_occ[occ_name]['fn'] += 1
                metrics_by_occ[occ_name]['total_gt'] += 1
            continue
        
        pred_boxes = pred_by_image[img_id]['boxes']
        pred_scores = pred_by_image[img_id]['scores']
        pred_labels = pred_by_image[img_id]['labels']
        
        gt_boxes = gt_by_image[img_id]['boxes']
        gt_labels = gt_by_image[img_id]['labels']
        gt_occlusions = gt_by_image[img_id]['occlusions']
        
        # Filter by score threshold
        valid_mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[valid_mask]
        pred_scores = pred_scores[valid_mask]
        pred_labels = pred_labels[valid_mask]
        
        # Track matched GTs and predictions
        matched_gts = set()
        matched_preds = set()
        
        # Sort predictions by score (highest first)
        sorted_indices = np.argsort(-pred_scores)
        
        # For each prediction, try to match to a GT
        for pred_idx in sorted_indices:
            pred_box = pred_boxes[pred_idx]
            pred_label = pred_labels[pred_idx]
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching GT
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gts:
                    continue
                if gt_labels[gt_idx] != pred_label:
                    continue
                
                iou = compute_iou(pred_box, gt_boxes[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # If match found
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gts.add(best_gt_idx)
                matched_preds.add(pred_idx)
                # Assign this TP to the GT's occlusion level
                gt_occ = gt_occlusions[best_gt_idx]
                occ_name = 'heavy' if gt_occ == 2 else ('partial' if gt_occ == 1 else 'none')
                metrics_by_occ[occ_name]['tp'] += 1
            else:
                # False positive - doesn't match any GT well enough
                # We could assign it to an occlusion level, but for now count it separately
                pass
        
        # Count false negatives (unmatched GTs) by occlusion level
        for gt_idx in range(len(gt_boxes)):
            gt_occ = gt_occlusions[gt_idx]
            occ_name = 'heavy' if gt_occ == 2 else ('partial' if gt_occ == 1 else 'none')
            metrics_by_occ[occ_name]['total_gt'] += 1
            
            if gt_idx not in matched_gts:
                metrics_by_occ[occ_name]['fn'] += 1
        
        # Count unmatched predictions as FPs
        # For per-object metrics, we assign FPs to the occlusion level of the closest GT
        # or distribute them if no GTs exist
        for pred_idx in range(len(pred_boxes)):
            if pred_idx in matched_preds:
                continue
            
            # This is a False Positive
            pred_box = pred_boxes[pred_idx]
            pred_label = pred_labels[pred_idx]
            
            # Find closest GT (even if below IoU threshold) to assign occlusion level
            best_iou = 0
            best_gt_idx = -1
            for gt_idx in range(len(gt_boxes)):
                if gt_labels[gt_idx] != pred_label:
                    continue
                iou = compute_iou(pred_box, gt_boxes[gt_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Assign FP to the occlusion level of the closest GT
            if best_gt_idx >= 0:
                gt_occ = gt_occlusions[best_gt_idx]
                occ_name = 'heavy' if gt_occ == 2 else ('partial' if gt_occ == 1 else 'none')
                metrics_by_occ[occ_name]['fp'] += 1
            else:
                # No matching GT of same class, distribute proportionally or assign to dominant occlusion level
                # For simplicity, count as FP in the most common occlusion level in this image
                if len(gt_occlusions) > 0:
                    dominant_occ = np.bincount(gt_occlusions).argmax()
                    occ_name = 'heavy' if dominant_occ == 2 else ('partial' if dominant_occ == 1 else 'none')
                    metrics_by_occ[occ_name]['fp'] += 1
    
    # Compute final metrics for each occlusion level
    results = {}
    for occ_name in ['none', 'partial', 'heavy']:
        tp = metrics_by_occ[occ_name]['tp']
        fp = metrics_by_occ[occ_name]['fp']
        fn = metrics_by_occ[occ_name]['fn']
        total_gt = metrics_by_occ[occ_name]['total_gt']
        
        # Correct precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[occ_name] = {
            'total_objects': total_gt,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    return results


def compute_per_class_metrics(predictions, targets, score_threshold=0.3):
    """
    Compute per-class metrics (AP, precision, recall, F1) for a subset of predictions/targets
    
    Returns:
        dict: Dictionary mapping class_id to metrics dict containing:
            - ap_0.5_0.95: Average Precision @ IoU 0.5:0.95
            - ap_0.5: Average Precision @ IoU 0.5
            - precision: Precision @ IoU 0.5
            - recall: Recall @ IoU 0.5
            - f1_score: F1 Score @ IoU 0.5
            - gt_count: Number of ground truth objects for this class
    """
    import numpy as np
    from collections import defaultdict
    
    if len(predictions) == 0 or len(targets) == 0:
        return {}
    
    # Group by image_id
    pred_by_image = defaultdict(lambda: {'boxes': [], 'scores': [], 'labels': []})
    gt_by_image = defaultdict(lambda: {'boxes': [], 'labels': []})
    
    # Group predictions by image_id
    for pred in predictions:
        img_id = pred['image_id'].item() if torch.is_tensor(pred['image_id']) else pred['image_id']
        
        pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
        pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']
        pred_labels = pred['labels'].cpu().numpy() if torch.is_tensor(pred['labels']) else pred['labels']
        
        if img_id not in pred_by_image or len(pred_by_image[img_id]['boxes']) == 0:
            pred_by_image[img_id]['boxes'] = pred_boxes
            pred_by_image[img_id]['scores'] = pred_scores
            pred_by_image[img_id]['labels'] = pred_labels
        else:
            pred_by_image[img_id]['boxes'] = np.concatenate([pred_by_image[img_id]['boxes'], pred_boxes])
            pred_by_image[img_id]['scores'] = np.concatenate([pred_by_image[img_id]['scores'], pred_scores])
            pred_by_image[img_id]['labels'] = np.concatenate([pred_by_image[img_id]['labels'], pred_labels])
    
    # Group targets by image_id
    for target in targets:
        img_id = target['image_id'].item() if torch.is_tensor(target['image_id']) else target['image_id']
        
        gt_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
        gt_labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else target['labels']
        
        if img_id not in gt_by_image or len(gt_by_image[img_id]['boxes']) == 0:
            gt_by_image[img_id]['boxes'] = gt_boxes
            gt_by_image[img_id]['labels'] = gt_labels
        else:
            gt_by_image[img_id]['boxes'] = np.concatenate([gt_by_image[img_id]['boxes'], gt_boxes])
            gt_by_image[img_id]['labels'] = np.concatenate([gt_by_image[img_id]['labels'], gt_labels])
    
    # Get all unique classes present in ground truth
    all_classes = set()
    for img_id in gt_by_image:
        all_classes.update(gt_by_image[img_id]['labels'].tolist())
    all_classes = sorted(list(all_classes))
    
    # IoU thresholds
    iou_thresholds_all = np.linspace(0.5, 0.95, 10)
    
    # Store results per class
    per_class_results = {}
    
    for class_id in all_classes:
        # Compute AP for this class across all IoU thresholds
        class_aps = []
        class_tp_50 = 0
        class_fp_50 = 0
        class_gt_count = 0
        
        for iou_thresh in iou_thresholds_all:
            class_scores = []
            class_matches = []
            local_gt_count = 0
            
            # Collect predictions and ground truth for this class
            for img_id in pred_by_image.keys():
                if img_id not in gt_by_image:
                    continue
                
                pred_boxes = pred_by_image[img_id]['boxes']
                pred_scores = pred_by_image[img_id]['scores']
                pred_labels = pred_by_image[img_id]['labels']
                
                # Filter by score threshold and class
                if len(pred_scores) > 0:
                    valid_mask = (pred_scores >= score_threshold) & (pred_labels == class_id)
                    pred_boxes_cls = pred_boxes[valid_mask]
                    pred_scores_cls = pred_scores[valid_mask]
                else:
                    pred_boxes_cls = np.array([])
                    pred_scores_cls = np.array([])
                
                gt_boxes = gt_by_image[img_id]['boxes']
                gt_labels = gt_by_image[img_id]['labels']
                
                # Filter ground truth by class
                gt_mask = gt_labels == class_id
                gt_boxes_cls = gt_boxes[gt_mask]
                local_gt_count += len(gt_boxes_cls)
                
                # Match predictions to ground truth
                matched_gts = set()
                for box, score in zip(pred_boxes_cls, pred_scores_cls):
                    best_iou = 0
                    best_gt = -1
                    for j, gt_box in enumerate(gt_boxes_cls):
                        if j in matched_gts:
                            continue
                        iou = compute_iou(box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = j
                    
                    class_scores.append(score)
                    if best_iou >= iou_thresh and best_gt >= 0:
                        matched_gts.add(best_gt)
                        class_matches.append(1)
                    else:
                        class_matches.append(0)
            
            # Store GT count (same for all IoU thresholds)
            if iou_thresh == 0.5:
                class_gt_count = local_gt_count
            
            # Compute AP for this class at this IoU threshold
            if len(class_scores) == 0 or local_gt_count == 0:
                class_aps.append(0.0)
            else:
                # Sort by score
                sorted_indices = np.argsort(-np.array(class_scores))
                matches = np.array(class_matches)[sorted_indices]
                
                # Compute precision-recall
                tp = np.cumsum(matches)
                fp = np.cumsum(1 - matches)
                recall_curve = tp / local_gt_count
                precision_curve = tp / (tp + fp)
                
                # Compute AP (area under PR curve using 101-point interpolation)
                ap = 0
                for t in np.linspace(0, 1, 101):
                    p = precision_curve[recall_curve >= t]
                    ap += (p.max() if len(p) > 0 else 0) / 101
                
                class_aps.append(ap)
                
                # For IoU=0.5, store TP/FP for precision/recall
                if iou_thresh == 0.5:
                    class_tp_50 = tp[-1] if len(tp) > 0 else 0
                    class_fp_50 = fp[-1] if len(fp) > 0 else 0
        
        # Compute metrics for this class
        ap_0_5_0_95 = np.mean(class_aps) if len(class_aps) > 0 else 0.0
        ap_0_5 = class_aps[0] if len(class_aps) > 0 else 0.0
        
        precision = class_tp_50 / (class_tp_50 + class_fp_50) if (class_tp_50 + class_fp_50) > 0 else 0.0
        recall = class_tp_50 / class_gt_count if class_gt_count > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_results[class_id] = {
            'ap_0.5_0.95': ap_0_5_0_95,
            'ap_0.5': ap_0_5,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'gt_count': class_gt_count
        }
    
    return per_class_results


def compute_metrics_for_subset(predictions, targets, coco_api, score_threshold=0.3):
    """
    Compute comprehensive metrics (mAP, precision, recall, F1, FPR) for a subset of predictions/targets
    """
    import numpy as np
    from collections import defaultdict
    
    if len(predictions) == 0 or len(targets) == 0:
        return {
            'mAP@0.5:0.95': 0.0,
            'mAP@0.5': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'fpr': 0.0
        }
    
    # Group by image_id
    pred_by_image = defaultdict(lambda: {'boxes': [], 'scores': [], 'labels': []})
    gt_by_image = defaultdict(lambda: {'boxes': [], 'labels': []})
    
    # Group predictions by image_id
    for pred in predictions:
        img_id = pred['image_id'].item() if torch.is_tensor(pred['image_id']) else pred['image_id']
        
        # Convert tensors to numpy if needed
        pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
        pred_scores = pred['scores'].cpu().numpy() if torch.is_tensor(pred['scores']) else pred['scores']
        pred_labels = pred['labels'].cpu().numpy() if torch.is_tensor(pred['labels']) else pred['labels']
        
        # Don't append - directly assign since each pred is per-image already
        if img_id not in pred_by_image or len(pred_by_image[img_id]['boxes']) == 0:
            pred_by_image[img_id]['boxes'] = pred_boxes
            pred_by_image[img_id]['scores'] = pred_scores
            pred_by_image[img_id]['labels'] = pred_labels
        else:
            # If same image appears multiple times, concatenate
            pred_by_image[img_id]['boxes'] = np.concatenate([pred_by_image[img_id]['boxes'], pred_boxes])
            pred_by_image[img_id]['scores'] = np.concatenate([pred_by_image[img_id]['scores'], pred_scores])
            pred_by_image[img_id]['labels'] = np.concatenate([pred_by_image[img_id]['labels'], pred_labels])
    
    # Group targets by image_id
    for target in targets:
        img_id = target['image_id'].item() if torch.is_tensor(target['image_id']) else target['image_id']
        
        gt_boxes = target['boxes'].cpu().numpy() if torch.is_tensor(target['boxes']) else target['boxes']
        gt_labels = target['labels'].cpu().numpy() if torch.is_tensor(target['labels']) else target['labels']
        
        if img_id not in gt_by_image or len(gt_by_image[img_id]['boxes']) == 0:
            gt_by_image[img_id]['boxes'] = gt_boxes
            gt_by_image[img_id]['labels'] = gt_labels
        else:
            gt_by_image[img_id]['boxes'] = np.concatenate([gt_by_image[img_id]['boxes'], gt_boxes])
            gt_by_image[img_id]['labels'] = np.concatenate([gt_by_image[img_id]['labels'], gt_labels])
    
    # Compute mAP @ 0.5:0.95 and @ 0.5
    iou_thresholds_all = np.linspace(0.5, 0.95, 10)
    iou_thresholds_50 = [0.5]
    
    # Debug: Check how many predictions pass score threshold
    total_preds_before = sum(len(pred_by_image[img_id]['scores']) for img_id in pred_by_image)
    total_preds_after = sum((pred_by_image[img_id]['scores'] >= score_threshold).sum() for img_id in pred_by_image)
    total_gt = sum(len(gt_by_image[img_id]['boxes']) for img_id in gt_by_image)
    
    # Get all unique classes present in ground truth
    all_classes = set()
    for img_id in gt_by_image:
        all_classes.update(gt_by_image[img_id]['labels'].tolist())
    all_classes = sorted(list(all_classes))
    
    # Compute per-class AP (like COCO does)
    per_class_aps_0595 = []  # AP@0.5:0.95 for each class
    per_class_aps_50 = []    # AP@0.5 for each class
    
    all_tp_total, all_fp_total, all_gt_total = 0, 0, 0
    
    for class_id in all_classes:
        # Compute AP for this class across all IoU thresholds
        class_aps = []
        
        for iou_thresh in iou_thresholds_all:
            class_scores = []
            class_matches = []
            class_gt_count = 0
            
            # Collect predictions and ground truth for this class
            for img_id in pred_by_image.keys():
                if img_id not in gt_by_image:
                    continue
                
                pred_boxes = pred_by_image[img_id]['boxes']
                pred_scores = pred_by_image[img_id]['scores']
                pred_labels = pred_by_image[img_id]['labels']
                
                # Filter by score threshold and class
                if len(pred_scores) > 0:
                    valid_mask = (pred_scores >= score_threshold) & (pred_labels == class_id)
                    pred_boxes_cls = pred_boxes[valid_mask]
                    pred_scores_cls = pred_scores[valid_mask]
                else:
                    pred_boxes_cls = np.array([])
                    pred_scores_cls = np.array([])
                
                gt_boxes = gt_by_image[img_id]['boxes']
                gt_labels = gt_by_image[img_id]['labels']
                
                # Filter ground truth by class
                gt_mask = gt_labels == class_id
                gt_boxes_cls = gt_boxes[gt_mask]
                class_gt_count += len(gt_boxes_cls)
                
                # Match predictions to ground truth
                matched_gts = set()
                for box, score in zip(pred_boxes_cls, pred_scores_cls):
                    best_iou = 0
                    best_gt = -1
                    for j, gt_box in enumerate(gt_boxes_cls):
                        if j in matched_gts:
                            continue
                        iou = compute_iou(box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = j
                    
                    class_scores.append(score)
                    if best_iou >= iou_thresh and best_gt >= 0:
                        matched_gts.add(best_gt)
                        class_matches.append(1)
                    else:
                        class_matches.append(0)
            
            # Compute AP for this class at this IoU threshold
            if len(class_scores) == 0 or class_gt_count == 0:
                class_aps.append(0.0)
            else:
                # Sort by score
                sorted_indices = np.argsort(-np.array(class_scores))
                matches = np.array(class_matches)[sorted_indices]
                
                # Compute precision-recall
                tp = np.cumsum(matches)
                fp = np.cumsum(1 - matches)
                recall_curve = tp / class_gt_count
                precision_curve = tp / (tp + fp)
                
                # Compute AP (area under PR curve using 101-point interpolation)
                ap = 0
                for t in np.linspace(0, 1, 101):
                    p = precision_curve[recall_curve >= t]
                    ap += (p.max() if len(p) > 0 else 0) / 101
                
                class_aps.append(ap)
                
                # For IoU=0.5, accumulate TP/FP/FN for overall metrics
                if iou_thresh == 0.5:
                    all_tp_total += tp[-1] if len(tp) > 0 else 0
                    all_fp_total += fp[-1] if len(fp) > 0 else 0
                    all_gt_total += class_gt_count
        
        # Store per-class AP@0.5:0.95 and AP@0.5
        if len(class_aps) > 0:
            per_class_aps_0595.append(np.mean(class_aps))  # Average across IoU thresholds
            per_class_aps_50.append(class_aps[0])  # First threshold is 0.5
        else:
            per_class_aps_0595.append(0.0)
            per_class_aps_50.append(0.0)
    
    # Compute mAP as average of per-class APs (like COCO)
    mAP_all = np.mean(per_class_aps_0595) if len(per_class_aps_0595) > 0 else 0.0
    mAP_50 = np.mean(per_class_aps_50) if len(per_class_aps_50) > 0 else 0.0
    
    # Compute precision, recall, F1 @ IoU=0.5
    precision = all_tp_total / (all_tp_total + all_fp_total) if (all_tp_total + all_fp_total) > 0 else 0.0
    recall = all_tp_total / all_gt_total if all_gt_total > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # False Positive Rate
    fpr = all_fp_total / (all_fp_total + all_tp_total) if (all_fp_total + all_tp_total) > 0 else 0.0
    
    return {
        'mAP@0.5:0.95': mAP_all,
        'mAP@0.5': mAP_50,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'fpr': fpr
    }


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def comprehensive_occlusion_diagnostics(predictions, targets, coco_api, output_dir, 
                                         iou_threshold=0.5, score_threshold=0.3):
    """
    Comprehensive diagnostics aggregated by occlusion level across ALL classes.
    
    Args:
        predictions: list of dicts with 'boxes', 'scores', 'labels', 'image_id'
        targets: list of dicts with 'boxes', 'labels', 'image_id', 'occlusion'
        coco_api: COCO API object
        output_dir: Directory to save diagnostic outputs
        iou_threshold: IoU threshold for matching
        score_threshold: Score threshold for valid predictions
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PER-OCCLUSION LEVEL DIAGNOSTICS - OBJECT-LEVEL AGGREGATION")
    print(f"{'='*80}")
    print(f"NOTE: This section uses individual object occlusion annotations.")
    print(f"      Each object is counted in its own occlusion level (NONE/PARTIAL/HEAVY).")
    print(f"      This differs from 'Per-Occlusion Level Metrics' which groups by image.")
    print(f"{'='*80}")
    
    # Create diagnostic directory
    diag_dir = os.path.join(output_dir, 'diagnostics_occlusion_levels')
    os.makedirs(diag_dir, exist_ok=True)
    
    # Aggregate statistics by occlusion level across ALL classes
    occlusion_stats = {
        0: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'ious': [], 'class_breakdown': {}},
        1: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'ious': [], 'class_breakdown': {}},
        2: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'ious': [], 'class_breakdown': {}}
    }
    
    # Get class names with robust mapping
    # Build mapping from model labels (0-indexed) to class names
    class_names_dict = {}  # Maps 0-indexed label -> class name
    cat_id_to_label = {}  # Maps COCO cat_id -> 0-indexed label
    
    if hasattr(coco_api, 'loadCats'):
        try:
            cat_ids = sorted(coco_api.getCatIds())
            cats = coco_api.loadCats(cat_ids)
            
            # Detect if dataset is 0-indexed or 1-indexed
            min_cat_id = min(cat_ids) if cat_ids else 1
            is_zero_indexed = (min_cat_id == 0)
            
            # Build mapping based on detected indexing
            for cat in cats:
                coco_cat_id = cat['id']
                class_name = cat['name']
                
                if is_zero_indexed:
                    # 0-indexed dataset: category_id IS the label
                    zero_indexed_label = coco_cat_id
                else:
                    # 1-indexed dataset: convert category_id to 0-indexed label
                    zero_indexed_label = coco_cat_id - 1
                
                class_names_dict[zero_indexed_label] = class_name
                cat_id_to_label[coco_cat_id] = zero_indexed_label
                
            print(f"\n[Diagnostics] Detected {'0-indexed' if is_zero_indexed else '1-indexed'} dataset")
            print(f"[Diagnostics] Class mapping: {class_names_dict}")
        except Exception as e:
            print(f"[Diagnostics] Warning: Could not build class mapping: {e}")
            pass
    
    # Process all predictions and targets
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].numpy() if len(pred['boxes']) > 0 else np.array([])
        pred_scores = pred['scores'].numpy() if len(pred['scores']) > 0 else np.array([])
        pred_labels = pred['labels'].numpy() if len(pred['labels']) > 0 else np.array([])
        
        gt_boxes = target['boxes'].numpy() if len(target['boxes']) > 0 else np.array([])
        gt_labels = target['labels'].numpy() if len(target['labels']) > 0 else np.array([])
        gt_occlusions = target['occlusion'].numpy() if 'occlusion' in target and len(target['occlusion']) > 0 else np.array([])
        
        # Update GT counts by occlusion level
        for gt_occ, gt_label in zip(gt_occlusions, gt_labels):
            occ_level = int(gt_occ)
            occlusion_stats[occ_level]['gt'] += 1
            
            # Track class breakdown
            class_name = class_names_dict.get(int(gt_label), f"class_{int(gt_label)}")
            if class_name not in occlusion_stats[occ_level]['class_breakdown']:
                occlusion_stats[occ_level]['class_breakdown'][class_name] = {'gt': 0, 'tp': 0, 'fn': 0, 'fp': 0}
            occlusion_stats[occ_level]['class_breakdown'][class_name]['gt'] += 1
        
        # Filter predictions by score
        valid_mask = pred_scores >= score_threshold
        pred_boxes_filtered = pred_boxes[valid_mask]
        pred_scores_filtered = pred_scores[valid_mask]
        pred_labels_filtered = pred_labels[valid_mask]
        
        matched_gts = set()
        
        # Sort by score descending
        sorted_indices = np.argsort(-pred_scores_filtered)
        
        for pred_idx in sorted_indices:
            pred_box = pred_boxes_filtered[pred_idx]
            pred_score = pred_scores_filtered[pred_idx]
            pred_label = pred_labels_filtered[pred_idx]
            
            best_iou = 0
            best_gt_idx = -1
            best_gt_occ = -1
            best_gt_label = -1
            
            for gt_idx, (gt_box, gt_label, gt_occ) in enumerate(zip(gt_boxes, gt_labels, gt_occlusions)):
                if gt_idx in matched_gts:
                    continue
                
                # Must match class (both pred_label and gt_label are 0-indexed)
                if int(pred_label) != int(gt_label):
                    continue
                
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt_occ = int(gt_occ)
                    best_gt_label = int(gt_label)
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True Positive
                matched_gts.add(best_gt_idx)
                occlusion_stats[best_gt_occ]['tp'] += 1
                occlusion_stats[best_gt_occ]['ious'].append(best_iou)
                
                class_name = class_names_dict.get(best_gt_label, f"class_{best_gt_label}")
                if class_name in occlusion_stats[best_gt_occ]['class_breakdown']:
                    occlusion_stats[best_gt_occ]['class_breakdown'][class_name]['tp'] += 1
            else:
                # False Positive - assign to occlusion level of closest GT of same class
                if best_gt_idx >= 0:
                    occlusion_stats[best_gt_occ]['fp'] += 1
                    class_name = class_names_dict.get(best_gt_label, f"class_{best_gt_label}")
                    if class_name in occlusion_stats[best_gt_occ]['class_breakdown']:
                        occlusion_stats[best_gt_occ]['class_breakdown'][class_name]['fp'] += 1
                elif len(gt_occlusions) > 0:
                    # Assign to dominant occlusion level in image
                    dominant_occ = int(np.bincount(gt_occlusions.astype(int)).argmax())
                    occlusion_stats[dominant_occ]['fp'] += 1
        
        # Count False Negatives
        for gt_idx, (gt_occ, gt_label) in enumerate(zip(gt_occlusions, gt_labels)):
            if gt_idx not in matched_gts:
                occ_level = int(gt_occ)
                occlusion_stats[occ_level]['fn'] += 1
                
                class_name = class_names_dict.get(int(gt_label), f"class_{int(gt_label)}")
                if class_name in occlusion_stats[occ_level]['class_breakdown']:
                    occlusion_stats[occ_level]['class_breakdown'][class_name]['fn'] += 1
    
    # Print comprehensive diagnostics for each occlusion level
    for occ_level in [0, 1, 2]:
        occ_name = ['NONE (No Occlusion)', 'PARTIAL (Partial Occlusion)', 'HEAVY (Heavy Occlusion)'][occ_level]
        stats = occlusion_stats[occ_level]
        
        print(f"\n{'-'*80}")
        print(f"OCCLUSION LEVEL {occ_level}: {occ_name}")
        print(f"{'-'*80}")
        
        # Basic counts
        print(f"\n  [Detection Counts - ALL CLASSES COMBINED]")
        print(f"    Ground Truth Objects: {stats['gt']}")
        print(f"    True Positives (Correctly Detected): {stats['tp']}")
        print(f"    False Positives (Incorrect Detections): {stats['fp']}")
        print(f"    False Negatives (Missed Objects): {stats['fn']}")
        print(f"    Total Predictions at this level: {stats['tp'] + stats['fp']}")
        
        # Performance metrics
        occ_precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        occ_recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        occ_f1 = 2 * occ_precision * occ_recall / (occ_precision + occ_recall) if (occ_precision + occ_recall) > 0 else 0
        
        print(f"\n  [Performance Metrics @ IoU≥{iou_threshold}, Score≥{score_threshold}]")
        print(f"    Precision@IoU={iou_threshold}: {occ_precision:.4f} ({stats['tp']}/{stats['tp'] + stats['fp']} correct out of all predictions)")
        print(f"    Recall@IoU={iou_threshold}: {occ_recall:.4f} ({stats['tp']}/{stats['gt']} detected out of all GT objects)")
        print(f"    F1-Score@IoU={iou_threshold}: {occ_f1:.4f} (harmonic mean of precision and recall)")
        
        # Additional metrics at different IoU thresholds for diagnosis
        print(f"\n  [Multi-Threshold Performance Analysis (Score≥{score_threshold})]")
        for test_iou in [0.5, 0.75, 0.9]:
            if len(stats['ious']) > 0:
                # Count TPs at this IoU threshold
                tp_at_iou = sum(1 for iou in stats['ious'] if iou >= test_iou)
                # FPs are predictions that didn't match at this threshold
                # Since we only track IoUs of TPs, FP = (total predictions) - (TPs at this threshold)
                fp_at_iou = (stats['tp'] + stats['fp']) - tp_at_iou
                # FN = GT objects not detected at this threshold
                fn_at_iou = stats['gt'] - tp_at_iou
                
                # Calculate metrics
                precision_at_iou = tp_at_iou / (tp_at_iou + fp_at_iou) if (tp_at_iou + fp_at_iou) > 0 else 0
                recall_at_iou = tp_at_iou / stats['gt'] if stats['gt'] > 0 else 0
                f1_at_iou = 2 * precision_at_iou * recall_at_iou / (precision_at_iou + recall_at_iou) if (precision_at_iou + recall_at_iou) > 0 else 0
                
                print(f"    @ IoU≥{test_iou}:")
                print(f"      Precision: {precision_at_iou:.4f}, Recall: {recall_at_iou:.4f}, F1: {f1_at_iou:.4f}")
                print(f"      TP={tp_at_iou}, FP={fp_at_iou}, FN={fn_at_iou} (out of {stats['gt']} GT objects)")
            else:
                print(f"    @ IoU≥{test_iou}:")
                print(f"      Precision: 0.0000, Recall: 0.0000, F1: 0.0000 (no TPs detected)")
        
        # Detection rate analysis
        detection_rate = stats['tp'] / stats['gt'] if stats['gt'] > 0 else 0
        miss_rate = stats['fn'] / stats['gt'] if stats['gt'] > 0 else 0
        
        print(f"\n  [Detection Quality @ IoU≥{iou_threshold}]")
        print(f"    Detection Rate: {detection_rate:.2%} (TP/{stats['gt']}: {stats['tp']} detected out of {stats['gt']} GT objects)")
        print(f"    Miss Rate: {miss_rate:.2%} (FN/{stats['gt']}: {stats['fn']} missed out of {stats['gt']} GT objects)")
        
        if stats['fp'] > 0:
            fp_rate = stats['fp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            print(f"    False Positive Rate: {fp_rate:.2%} (FP/(TP+FP): {stats['fp']}/{stats['tp'] + stats['fp']} predictions are incorrect)")
        
        # Score threshold analysis
        print(f"\n  [Filtering Impact]")
        print(f"    Score Threshold Used: {score_threshold}")
        print(f"    Predictions Above Threshold: {stats['tp'] + stats['fp']}")
        print(f"    Note: Lower thresholds may increase recall but decrease precision")
        
        # IoU analysis for TPs
        if len(stats['ious']) > 0:
            avg_iou = np.mean(stats['ious'])
            std_iou = np.std(stats['ious'])
            min_iou = np.min(stats['ious'])
            max_iou = np.max(stats['ious'])
            median_iou = np.median(stats['ious'])
            
            print(f"\n  [IoU Statistics for True Positives (matched at IoU≥{iou_threshold})]")
            print(f"    Average IoU: {avg_iou:.4f}")
            print(f"    Std Dev IoU: {std_iou:.4f}")
            print(f"    Min IoU: {min_iou:.4f}")
            print(f"    Max IoU: {max_iou:.4f}")
            print(f"    Median IoU: {median_iou:.4f}")
            print(f"    Note: These are the actual IoU values of detections matched at IoU≥{iou_threshold}")
            
            # Count how many TPs have high/medium/low IoU
            high_iou_count = sum(1 for iou in stats['ious'] if iou >= 0.75)
            med_iou_count = sum(1 for iou in stats['ious'] if 0.5 <= iou < 0.75)
            low_iou_count = sum(1 for iou in stats['ious'] if iou < 0.5)
            
            print(f"\n  [IoU Quality Distribution]")
            print(f"    High IoU (≥0.75): {high_iou_count} ({high_iou_count/len(stats['ious']):.1%})")
            print(f"    Medium IoU (0.5-0.75): {med_iou_count} ({med_iou_count/len(stats['ious']):.1%})")
            print(f"    Low IoU (<0.5): {low_iou_count} ({low_iou_count/len(stats['ious']):.1%})")
        else:
            print(f"\n  [IoU Statistics]")
            print(f"    No true positives detected at this occlusion level")
        
        # Per-class breakdown
        print(f"\n  [Per-Class Breakdown @ IoU≥{iou_threshold}, Score≥{score_threshold}]")
        for class_name in sorted(stats['class_breakdown'].keys()):
            cls_stats = stats['class_breakdown'][class_name]
            cls_recall = cls_stats['tp'] / cls_stats['gt'] if cls_stats['gt'] > 0 else 0
            cls_precision = cls_stats['tp'] / (cls_stats['tp'] + cls_stats['fp']) if (cls_stats['tp'] + cls_stats['fp']) > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
            cls_miss_rate = cls_stats['fn'] / cls_stats['gt'] if cls_stats['gt'] > 0 else 0
            
            print(f"    {class_name}:")
            print(f"      GT: {cls_stats['gt']}, TP: {cls_stats['tp']}, FP: {cls_stats['fp']}, FN: {cls_stats['fn']}")
            print(f"      Precision@IoU={iou_threshold}: {cls_precision:.4f}, Recall@IoU={iou_threshold}: {cls_recall:.4f}, F1@IoU={iou_threshold}: {cls_f1:.4f}")
            print(f"      Miss Rate: {cls_miss_rate:.2%} ({cls_stats['fn']} of {cls_stats['gt']} objects missed)")
            
            # Identify problematic classes
            if cls_stats['gt'] > 0:
                if cls_recall < 0.5:
                    print(f"      ⚠ ALERT: Very low recall - most objects are being missed!")
                elif cls_stats['fn'] > cls_stats['tp']:
                    print(f"      ⚠ WARNING: More objects missed than detected")
                if cls_stats['fp'] > cls_stats['tp'] and cls_stats['tp'] > 0:
                    print(f"      ⚠ WARNING: More false positives than true positives")
        
        # Performance interpretation
        print(f"\n  [Performance Interpretation]")
        if stats['gt'] == 0:
            print(f"    No ground truth objects at this occlusion level in the test set")
        elif occ_recall >= 0.9:
            print(f"    ✓ EXCELLENT: Model detects {occ_recall:.1%} of objects at this occlusion level")
        elif occ_recall >= 0.7:
            print(f"    ✓ GOOD: Model detects {occ_recall:.1%} of objects at this occlusion level")
        elif occ_recall >= 0.5:
            print(f"    ⚠ MODERATE: Model detects only {occ_recall:.1%} of objects - room for improvement")
        else:
            print(f"    ✗ POOR: Model detects only {occ_recall:.1%} of objects - significant issues")
        
        if stats['fp'] > stats['tp'] and stats['tp'] > 0:
            print(f"    ⚠ WARNING: More false positives than true positives - model is over-predicting")
        elif stats['fp'] > 0 and occ_precision < 0.7:
            print(f"    ⚠ WARNING: Precision is {occ_precision:.1%} - many incorrect detections")
        
        if len(stats['ious']) > 0 and avg_iou < 0.7:
            print(f"    ⚠ WARNING: Average IoU is {avg_iou:.4f} - bounding boxes could be more accurate")
        elif len(stats['ious']) > 0 and avg_iou >= 0.8:
            print(f"    ✓ GOOD: Average IoU is {avg_iou:.4f} - bounding boxes are well-localized")
        
        # Diagnostic hints for low performance
        print(f"\n  [Diagnostic Hints for Low Performance]")
        if stats['fn'] > 0:
            print(f"    → {stats['fn']} objects were NOT detected (False Negatives)")
            print(f"      Possible causes:")
            print(f"        - Score threshold {score_threshold} is too high (detections filtered out)")
            print(f"        - Model confidence is low for these objects")
            print(f"        - Objects are in difficult poses/lighting/backgrounds")
            if occ_level == 0:
                print(f"        - Non-occluded objects should be easiest - check for annotation errors")
                print(f"        - Model may be overfitting to occluded training data")
        
        if stats['fp'] > 0:
            print(f"    → {stats['fp']} false detections (False Positives)")
            print(f"      Possible causes:")
            print(f"        - IoU threshold {iou_threshold} requires tighter bounding boxes")
            print(f"        - Model is detecting background as objects")
            print(f"        - Duplicate detections (NMS threshold may be too high)")
            if occ_level == 0:
                print(f"        - Check if model confuses non-occluded objects with other classes")
        
        if len(stats['ious']) > 0:
            low_iou_ratio = sum(1 for iou in stats['ious'] if iou < 0.7) / len(stats['ious'])
            if low_iou_ratio > 0.3:
                print(f"    → {low_iou_ratio:.1%} of TPs have IoU < 0.7 (poor localization)")
                print(f"      Possible causes:")
                print(f"        - Bounding box regression needs improvement")
                print(f"        - Model struggles with object boundaries at this occlusion level")
                if occ_level == 0:
                    print(f"        - Non-occluded objects should have higher IoU - check bbox quality")
    
    # Cross-occlusion comparison
    print(f"\n{'='*80}")
    print(f"CROSS-OCCLUSION COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Show metrics at multiple IoU thresholds
    for test_iou in [0.5, 0.75, 0.9]:
        print(f"\n{'─'*80}")
        print(f"Metrics @ IoU≥{test_iou}, Score≥{score_threshold}")
        print(f"{'─'*80}")
        
        print(f"\n  [Recall@IoU={test_iou}]")
        for occ_level in [0, 1, 2]:
            occ_name = ['NONE', 'PARTIAL', 'HEAVY'][occ_level]
            stats = occlusion_stats[occ_level]
            
            if len(stats['ious']) > 0:
                tp_at_iou = sum(1 for iou in stats['ious'] if iou >= test_iou)
                recall = tp_at_iou / stats['gt'] if stats['gt'] > 0 else 0
                print(f"    {occ_name:8s}: {recall:.4f} ({tp_at_iou}/{stats['gt']} detected)")
            else:
                print(f"    {occ_name:8s}: 0.0000 (0/{stats['gt']} detected)")
        
        print(f"\n  [Precision@IoU={test_iou}]")
        for occ_level in [0, 1, 2]:
            occ_name = ['NONE', 'PARTIAL', 'HEAVY'][occ_level]
            stats = occlusion_stats[occ_level]
            
            if len(stats['ious']) > 0:
                tp_at_iou = sum(1 for iou in stats['ious'] if iou >= test_iou)
                fp_at_iou = (stats['tp'] + stats['fp']) - tp_at_iou
                precision = tp_at_iou / (tp_at_iou + fp_at_iou) if (tp_at_iou + fp_at_iou) > 0 else 0
                print(f"    {occ_name:8s}: {precision:.4f} ({tp_at_iou}/{tp_at_iou + fp_at_iou} correct)")
            else:
                print(f"    {occ_name:8s}: 0.0000 (0/{stats['tp'] + stats['fp']} correct)")
        
        print(f"\n  [F1-Score@IoU={test_iou}]")
        for occ_level in [0, 1, 2]:
            occ_name = ['NONE', 'PARTIAL', 'HEAVY'][occ_level]
            stats = occlusion_stats[occ_level]
            
            if len(stats['ious']) > 0:
                tp_at_iou = sum(1 for iou in stats['ious'] if iou >= test_iou)
                fp_at_iou = (stats['tp'] + stats['fp']) - tp_at_iou
                precision = tp_at_iou / (tp_at_iou + fp_at_iou) if (tp_at_iou + fp_at_iou) > 0 else 0
                recall = tp_at_iou / stats['gt'] if stats['gt'] > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                print(f"    {occ_name:8s}: {f1:.4f}")
            else:
                print(f"    {occ_name:8s}: 0.0000")
    
    # Identify anomalies at IoU=0.5
    print(f"\n{'─'*80}")
    print(f"Anomaly Detection")
    print(f"{'─'*80}")
    
    recalls_05 = []
    for occ_level in [0, 1, 2]:
        stats = occlusion_stats[occ_level]
        if len(stats['ious']) > 0:
            tp_at_05 = sum(1 for iou in stats['ious'] if iou >= 0.5)
            recall = tp_at_05 / stats['gt'] if stats['gt'] > 0 else 0
        else:
            recall = 0
        recalls_05.append(recall)
    
    if recalls_05[0] < recalls_05[1] or recalls_05[0] < recalls_05[2]:
        print(f"\n⚠ ANOMALY DETECTED @ IoU=0.5:")
        print(f"  Non-occluded (NONE) recall is LOWER than occluded objects!")
        print(f"  Recall comparison: NONE={recalls_05[0]:.4f}, PARTIAL={recalls_05[1]:.4f}, HEAVY={recalls_05[2]:.4f}")
        print(f"  This is unusual - non-occluded objects should be easiest to detect.")
        print(f"  Potential issues:")
        print(f"    1. Dataset annotation errors (objects marked as non-occluded are actually occluded)")
        print(f"    2. Model overfitting to occluded training samples")
        print(f"    3. Non-occluded objects may be in more challenging contexts (small, far, unusual angles)")
        print(f"    4. Class imbalance - check per-class breakdown above")
    else:
        print(f"\nNo anomalies detected - performance degrades as expected with occlusion.")
    
    print(f"\n{'─'*80}")
    print(f"Average IoU of True Positives (matched at IoU≥{iou_threshold})")
    print(f"{'─'*80}")
    for occ_level in [0, 1, 2]:
        occ_name = ['NONE', 'PARTIAL', 'HEAVY'][occ_level]
        stats = occlusion_stats[occ_level]
        avg_iou = np.mean(stats['ious']) if len(stats['ious']) > 0 else 0
        print(f"  {occ_name:8s}: {avg_iou:.4f}")
    
    # Create visualization comparing occlusion levels
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    occ_names = ['None', 'Partial', 'Heavy']
    
    # Plot 1: Precision, Recall, F1 by occlusion
    ax = axes[0, 0]
    precisions = [occlusion_stats[i]['tp'] / (occlusion_stats[i]['tp'] + occlusion_stats[i]['fp']) 
                  if (occlusion_stats[i]['tp'] + occlusion_stats[i]['fp']) > 0 else 0 for i in range(3)]
    recalls = [occlusion_stats[i]['tp'] / (occlusion_stats[i]['tp'] + occlusion_stats[i]['fn']) 
               if (occlusion_stats[i]['tp'] + occlusion_stats[i]['fn']) > 0 else 0 for i in range(3)]
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
    
    x = np.arange(len(occ_names))
    width = 0.25
    ax.bar(x - width, precisions, width, label='Precision', color='blue', alpha=0.7)
    ax.bar(x, recalls, width, label='Recall', color='green', alpha=0.7)
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='orange', alpha=0.7)
    ax.set_xlabel('Occlusion Level')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Occlusion Level (All Classes)')
    ax.set_xticks(x)
    ax.set_xticklabels(occ_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Plot 2: GT distribution and detection
    ax = axes[0, 1]
    gt_counts = [occlusion_stats[i]['gt'] for i in range(3)]
    tp_counts = [occlusion_stats[i]['tp'] for i in range(3)]
    fn_counts = [occlusion_stats[i]['fn'] for i in range(3)]
    
    ax.bar(x, gt_counts, label='Total GT', color='blue', alpha=0.5)
    ax.bar(x, tp_counts, label='Detected (TP)', color='green', alpha=0.7)
    ax.bar(x, fn_counts, bottom=tp_counts, label='Missed (FN)', color='red', alpha=0.7)
    ax.set_xlabel('Occlusion Level')
    ax.set_ylabel('Count')
    ax.set_title('Detection vs Missed by Occlusion Level')
    ax.set_xticks(x)
    ax.set_xticklabels(occ_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Average IoU by occlusion
    ax = axes[1, 0]
    avg_ious = [np.mean(occlusion_stats[i]['ious']) if len(occlusion_stats[i]['ious']) > 0 else 0 for i in range(3)]
    ax.bar(x, avg_ious, color='purple', alpha=0.7)
    ax.set_xlabel('Occlusion Level')
    ax.set_ylabel('Average IoU')
    ax.set_title('Average IoU for True Positives by Occlusion Level')
    ax.set_xticks(x)
    ax.set_xticklabels(occ_names)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Plot 4: TP/FP/FN stacked bars
    ax = axes[1, 1]
    tp_counts = [occlusion_stats[i]['tp'] for i in range(3)]
    fp_counts = [occlusion_stats[i]['fp'] for i in range(3)]
    fn_counts = [occlusion_stats[i]['fn'] for i in range(3)]
    
    ax.bar(x, tp_counts, label='True Positives', color='green', alpha=0.7)
    ax.bar(x, fp_counts, bottom=tp_counts, label='False Positives', color='red', alpha=0.7)
    bottom = [tp + fp for tp, fp in zip(tp_counts, fp_counts)]
    ax.bar(x, fn_counts, bottom=bottom, label='False Negatives', color='orange', alpha=0.7)
    ax.set_xlabel('Occlusion Level')
    ax.set_ylabel('Count')
    ax.set_title('TP/FP/FN Distribution by Occlusion Level')
    ax.set_xticks(x)
    ax.set_xticklabels(occ_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(diag_dir, 'occlusion_level_diagnostics.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*80}")
    print(f"Saved occlusion level diagnostic plots to: {fig_path}")
    print(f"{'='*80}\n")


def visualize_and_diagnose_class(predictions, targets, class_id, class_name, coco_api, output_dir,
                                  iou_threshold=0.5, score_threshold=0.3,
                                  confusion_classes=None):
    """
    Comprehensive diagnostics for a specific class to understand performance issues.

    Args:
        predictions: list of dicts with 'boxes', 'scores', 'labels', 'image_id'
        targets: list of dicts with 'boxes', 'labels', 'image_id', 'occlusion'
        class_id: The class ID to diagnose (0-indexed in tensors, but should match label values)
        class_name: Name of the class
        coco_api: COCO API object
        output_dir: Directory to save diagnostic outputs
        iou_threshold: IoU threshold for matching
        score_threshold: Score threshold for valid predictions
        confusion_classes: Optional list of (class_id, class_name) tuples to track misclassifications to
                          (e.g., [(car_id, 'car'), (truck_id, 'truck')] when diagnosing motorcycle)
    """
    # Normalize confusion_classes to always be a list (empty if None)
    if confusion_classes is None:
        confusion_classes = []
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import cv2
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC ANALYSIS FOR CLASS: {class_name} (ID={class_id})")
    print(f"{'='*80}\n")
    
    # Create diagnostic directory
    diag_dir = os.path.join(output_dir, f'diagnostics_{class_name.lower()}')
    os.makedirs(diag_dir, exist_ok=True)
    
    # 1. Collect all instances of this class
    class_predictions = []
    class_targets = []
    
    for pred, target in zip(predictions, targets):
        # Filter for this class
        pred_mask = pred['labels'] == class_id
        target_mask = target['labels'] == class_id
        
        if pred_mask.any():
            class_predictions.append({
                'boxes': pred['boxes'][pred_mask],
                'scores': pred['scores'][pred_mask],
                'image_id': pred['image_id']
            })
        else:
            class_predictions.append({
                'boxes': torch.tensor([]),
                'scores': torch.tensor([]),
                'image_id': pred['image_id']
            })
            
        if target_mask.any():
            class_targets.append({
                'boxes': target['boxes'][target_mask],
                'image_id': target['image_id'],
                'occlusion': target['occlusion'][target_mask] if 'occlusion' in target else torch.zeros(target_mask.sum(), dtype=torch.int64)
            })
        else:
            class_targets.append({
                'boxes': torch.tensor([]),
                'image_id': target['image_id'],
                'occlusion': torch.tensor([])
            })
    
    # 1b. Collect confusion class predictions (e.g., car/truck/bus predictions for motorcycle GT)
    # Dictionary: confusion_class_id -> list of predictions per image
    confusion_predictions_by_class = {}
    for conf_class_id, conf_class_name in confusion_classes:
        conf_preds = []
        for pred in predictions:
            confusion_mask = pred['labels'] == conf_class_id
            if confusion_mask.any():
                conf_preds.append({
                    'boxes': pred['boxes'][confusion_mask],
                    'scores': pred['scores'][confusion_mask],
                    'image_id': pred['image_id']
                })
            else:
                conf_preds.append({
                    'boxes': torch.tensor([]),
                    'scores': torch.tensor([]),
                    'image_id': pred['image_id']
                })
        confusion_predictions_by_class[conf_class_id] = conf_preds

    # 2. Analyze detection statistics
    total_gt = sum(len(t['boxes']) for t in class_targets)
    total_pred = sum(len(p['boxes']) for p in class_predictions)
    total_pred_filtered = sum((p['scores'] >= score_threshold).sum().item() for p in class_predictions if len(p['scores']) > 0)

    print(f"[Detection Statistics]")
    print(f"  Total GT instances: {total_gt}")
    print(f"  Total predictions (all scores): {total_pred}")
    print(f"  Total predictions (score>={score_threshold}): {total_pred_filtered}")

    # 3. Analyze score distribution
    all_scores = []
    all_scores_matched = []  # Scores of TPs
    all_scores_unmatched = []  # Scores of FPs

    tp_count = 0
    fp_count = 0
    fn_count = 0

    # Track misclassifications (GT of this class predicted as any confusion class)
    # Each entry: {image_id, gt_box, pred_box, pred_score, iou, gt_occlusion, confusion_class_id, confusion_class_name}
    misclassifications = []

    # Track metrics by occlusion level
    occlusion_stats = {0: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'ious': []},
                       1: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'ious': []},
                       2: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'ious': []}}
    
    for pred, target in zip(class_predictions, class_targets):
        pred_boxes = pred['boxes'].numpy() if len(pred['boxes']) > 0 else np.array([])
        pred_scores = pred['scores'].numpy() if len(pred['scores']) > 0 else np.array([])
        
        gt_boxes = target['boxes'].numpy() if len(target['boxes']) > 0 else np.array([])
        gt_occlusions = target['occlusion'].numpy() if len(target['occlusion']) > 0 else np.array([])
        
        # Update occlusion GT counts
        for occ in gt_occlusions:
            occlusion_stats[int(occ)]['gt'] += 1
        
        # Filter by score
        valid_mask = pred_scores >= score_threshold
        pred_boxes_filtered = pred_boxes[valid_mask]
        pred_scores_filtered = pred_scores[valid_mask]
        
        all_scores.extend(pred_scores.tolist())
        
        matched_gts = set()
        
        # Sort by score descending
        sorted_indices = np.argsort(-pred_scores_filtered)
        
        for pred_idx in sorted_indices:
            pred_box = pred_boxes_filtered[pred_idx]
            pred_score = pred_scores_filtered[pred_idx]
            
            best_iou = 0
            best_gt_idx = -1
            best_gt_occ = -1
            
            for gt_idx, (gt_box, gt_occ) in enumerate(zip(gt_boxes, gt_occlusions)):
                if gt_idx in matched_gts:
                    continue
                
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt_occ = int(gt_occ)
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True Positive
                matched_gts.add(best_gt_idx)
                tp_count += 1
                all_scores_matched.append(pred_score)
                occlusion_stats[best_gt_occ]['tp'] += 1
                occlusion_stats[best_gt_occ]['ious'].append(best_iou)
            else:
                # False Positive - assign to occlusion level of closest GT
                fp_count += 1
                all_scores_unmatched.append(pred_score)
                
                # Assign FP to occlusion level by finding closest GT (even if IoU < threshold)
                # This matches the logic in compute_per_object_occlusion_metrics
                if best_gt_idx >= 0:
                    # Found a GT of same class, assign to its occlusion level
                    occlusion_stats[best_gt_occ]['fp'] += 1
                elif len(gt_occlusions) > 0:
                    # No GT of same class, assign to dominant occlusion level in this image
                    dominant_occ = int(np.bincount(gt_occlusions.astype(int)).argmax())
                    occlusion_stats[dominant_occ]['fp'] += 1
                # else: no GTs in image at all - don't assign to any occlusion level
        
        # Count False Negatives (unmatched GTs)
        for gt_idx, gt_occ in enumerate(gt_occlusions):
            if gt_idx not in matched_gts:
                fn_count += 1
                occlusion_stats[int(gt_occ)]['fn'] += 1

        # Check for misclassifications: GT of this class matched by any confusion class predictions
        if len(confusion_classes) > 0:
            img_id = target['image_id'].item() if torch.is_tensor(target['image_id']) else target['image_id']

            # Check each confusion class
            for conf_class_id, conf_class_name in confusion_classes:
                confusion_predictions = confusion_predictions_by_class.get(conf_class_id, [])
                if len(confusion_predictions) == 0:
                    continue

                # Find confusion predictions for same image
                conf_pred = None
                for cp in confusion_predictions:
                    cp_img_id = cp['image_id'].item() if torch.is_tensor(cp['image_id']) else cp['image_id']
                    if cp_img_id == img_id:
                        conf_pred = cp
                        break

                if conf_pred is not None and len(conf_pred['boxes']) > 0:
                    conf_boxes = conf_pred['boxes'].numpy() if torch.is_tensor(conf_pred['boxes']) else conf_pred['boxes']
                    conf_scores = conf_pred['scores'].numpy() if torch.is_tensor(conf_pred['scores']) else conf_pred['scores']

                    # Filter by score threshold
                    conf_valid_mask = conf_scores >= score_threshold
                    conf_boxes_filtered = conf_boxes[conf_valid_mask]
                    conf_scores_filtered = conf_scores[conf_valid_mask]

                    # Check each GT of this class for overlap with this confusion class predictions
                    for gt_idx, (gt_box, gt_occ) in enumerate(zip(gt_boxes, gt_occlusions)):
                        for conf_idx, (conf_box, conf_score) in enumerate(zip(conf_boxes_filtered, conf_scores_filtered)):
                            iou = compute_iou(gt_box, conf_box)
                            if iou >= iou_threshold:
                                # This GT was misclassified as this confusion class
                                misclassifications.append({
                                    'image_id': img_id,
                                    'gt_box': gt_box.tolist(),
                                    'pred_box': conf_box.tolist(),
                                    'pred_score': float(conf_score),
                                    'iou': float(iou),
                                    'gt_occlusion': int(gt_occ),
                                    'confusion_class_id': conf_class_id,
                                    'confusion_class_name': conf_class_name
                                })

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n[Matching Statistics @ IoU>={iou_threshold}, Score>={score_threshold}]")
    print(f"  True Positives: {tp_count}")
    print(f"  False Positives: {fp_count}")
    print(f"  False Negatives: {fn_count}")
    print(f"  Precision@IoU={iou_threshold}: {precision:.4f}")
    print(f"  Recall@IoU={iou_threshold}: {recall:.4f}")
    print(f"  F1-Score@IoU={iou_threshold}: {f1:.4f}")
    
    # 4. Occlusion analysis
    print(f"\n[Per-Occlusion Analysis @ IoU>={iou_threshold}, Score>={score_threshold}]")
    for occ_level in [0, 1, 2]:
        occ_name = ['NONE', 'PARTIAL', 'HEAVY'][occ_level]
        stats = occlusion_stats[occ_level]
        
        occ_precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
        occ_recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
        occ_f1 = 2 * occ_precision * occ_recall / (occ_precision + occ_recall) if (occ_precision + occ_recall) > 0 else 0
        avg_iou = np.mean(stats['ious']) if len(stats['ious']) > 0 else 0
        
        print(f"  {occ_name} (level {occ_level}):")
        print(f"    GT Count: {stats['gt']}")
        print(f"    TP: {stats['tp']}, FP: {stats['fp']}, FN: {stats['fn']}")
        print(f"    Precision@IoU={iou_threshold}: {occ_precision:.4f}, Recall@IoU={iou_threshold}: {occ_recall:.4f}, F1@IoU={iou_threshold}: {occ_f1:.4f}")
        print(f"    Avg IoU of TPs (matched at IoU≥{iou_threshold}): {avg_iou:.4f}")
    
    # 5. Score distribution analysis
    print(f"\n[Score Distribution Analysis]")
    if len(all_scores) > 0:
        print(f"  All predictions:")
        print(f"    Mean: {np.mean(all_scores):.4f}, Std: {np.std(all_scores):.4f}")
        print(f"    Min: {np.min(all_scores):.4f}, Max: {np.max(all_scores):.4f}")
        print(f"    Median: {np.median(all_scores):.4f}")
    
    if len(all_scores_matched) > 0:
        print(f"  True Positives (matched):")
        print(f"    Mean: {np.mean(all_scores_matched):.4f}, Std: {np.std(all_scores_matched):.4f}")
        print(f"    Min: {np.min(all_scores_matched):.4f}, Max: {np.max(all_scores_matched):.4f}")
    
    if len(all_scores_unmatched) > 0:
        print(f"  False Positives (unmatched):")
        print(f"    Mean: {np.mean(all_scores_unmatched):.4f}, Std: {np.std(all_scores_unmatched):.4f}")
        print(f"    Min: {np.min(all_scores_unmatched):.4f}, Max: {np.max(all_scores_unmatched):.4f}")

    # 5b. Misclassification analysis
    if len(confusion_classes) > 0 and len(misclassifications) > 0:
        conf_class_names = ', '.join([name for _, name in confusion_classes])
        print(f"\n[Misclassification Analysis: {class_name} -> {{{conf_class_names}}}]")
        print(f"  Total misclassifications: {len(misclassifications)}")
        unique_misclass_images = len(set(m['image_id'] for m in misclassifications))
        print(f"  Unique images with misclassifications: {unique_misclass_images}")

        # Group by confusion class
        print(f"\n  By confusion class:")
        for conf_class_id, conf_class_name in confusion_classes:
            count = sum(1 for m in misclassifications if m['confusion_class_name'] == conf_class_name)
            print(f"    {class_name} -> {conf_class_name}: {count}")

        # Group by occlusion level
        print(f"\n  By occlusion level:")
        misclass_by_occ = {0: 0, 1: 0, 2: 0}
        for m in misclassifications:
            misclass_by_occ[m['gt_occlusion']] += 1
        for occ_level in [0, 1, 2]:
            occ_name = ['NONE', 'PARTIAL', 'HEAVY'][occ_level]
            print(f"    {occ_name}: {misclass_by_occ[occ_level]}")

        # Score distribution of misclassifications
        misclass_scores = [m['pred_score'] for m in misclassifications]
        misclass_ious = [m['iou'] for m in misclassifications]
        print(f"  Misclassification scores: Mean={np.mean(misclass_scores):.4f}, Min={np.min(misclass_scores):.4f}, Max={np.max(misclass_scores):.4f}")
        print(f"  Misclassification IoUs: Mean={np.mean(misclass_ious):.4f}, Min={np.min(misclass_ious):.4f}, Max={np.max(misclass_ious):.4f}")

    # 6. Generate visualizations
    print(f"\n[Generating Visualizations]")
    
    # Plot 1: Score distribution histogram
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Score histogram
    ax = axes[0, 0]
    if len(all_scores) > 0:
        ax.hist(all_scores, bins=50, alpha=0.5, label='All predictions', color='blue')
    if len(all_scores_matched) > 0:
        ax.hist(all_scores_matched, bins=50, alpha=0.5, label='True Positives', color='green')
    if len(all_scores_unmatched) > 0:
        ax.hist(all_scores_unmatched, bins=50, alpha=0.5, label='False Positives', color='red')
    ax.axvline(score_threshold, color='black', linestyle='--', label=f'Threshold={score_threshold}')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title(f'{class_name}: Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-occlusion metrics
    ax = axes[0, 1]
    occ_names = ['None', 'Partial', 'Heavy']
    precisions = [occlusion_stats[i]['tp'] / (occlusion_stats[i]['tp'] + occlusion_stats[i]['fp']) 
                  if (occlusion_stats[i]['tp'] + occlusion_stats[i]['fp']) > 0 else 0 for i in range(3)]
    recalls = [occlusion_stats[i]['tp'] / (occlusion_stats[i]['tp'] + occlusion_stats[i]['fn']) 
               if (occlusion_stats[i]['tp'] + occlusion_stats[i]['fn']) > 0 else 0 for i in range(3)]
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
    
    x = np.arange(len(occ_names))
    width = 0.25
    ax.bar(x - width, precisions, width, label='Precision', color='blue', alpha=0.7)
    ax.bar(x, recalls, width, label='Recall', color='green', alpha=0.7)
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='orange', alpha=0.7)
    ax.set_xlabel('Occlusion Level')
    ax.set_ylabel('Score')
    ax.set_title(f'{class_name}: Per-Occlusion Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(occ_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # GT distribution by occlusion
    ax = axes[1, 0]
    gt_counts = [occlusion_stats[i]['gt'] for i in range(3)]
    tp_counts = [occlusion_stats[i]['tp'] for i in range(3)]
    fn_counts = [occlusion_stats[i]['fn'] for i in range(3)]
    
    ax.bar(x, gt_counts, label='Total GT', color='blue', alpha=0.5)
    ax.bar(x, tp_counts, label='Detected (TP)', color='green', alpha=0.7)
    ax.bar(x, fn_counts, bottom=tp_counts, label='Missed (FN)', color='red', alpha=0.7)
    ax.set_xlabel('Occlusion Level')
    ax.set_ylabel('Count')
    ax.set_title(f'{class_name}: Detection vs Missed by Occlusion')
    ax.set_xticks(x)
    ax.set_xticklabels(occ_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Confusion between TP/FP/FN
    ax = axes[1, 1]
    categories = ['TP', 'FP', 'FN']
    counts = [tp_count, fp_count, fn_count]
    colors = ['green', 'red', 'orange']
    ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title(f'{class_name}: TP/FP/FN Distribution')
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax.text(i, count + max(counts)*0.02, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = os.path.join(diag_dir, f'{class_name.lower()}_diagnostic_plots.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved diagnostic plots to: {fig_path}")
    
    # 7. Find worst performing images (highest FN rate)
    print(f"\n[Identifying Problematic Images]")
    image_stats = {}
    
    for pred, target in zip(class_predictions, class_targets):
        img_id = target['image_id'].item()
        
        gt_boxes = target['boxes'].numpy() if len(target['boxes']) > 0 else np.array([])
        pred_boxes = pred['boxes'].numpy() if len(pred['boxes']) > 0 else np.array([])
        pred_scores = pred['scores'].numpy() if len(pred['scores']) > 0 else np.array([])
        
        # Filter by score
        valid_mask = pred_scores >= score_threshold
        pred_boxes_filtered = pred_boxes[valid_mask]
        
        # Count matches
        matched_gts = set()
        for pred_box in pred_boxes_filtered:
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gts:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    matched_gts.add(gt_idx)
                    break
        
        tp = len(matched_gts)
        fn = len(gt_boxes) - tp
        fp = len(pred_boxes_filtered) - tp
        
        if len(gt_boxes) > 0:  # Only consider images with GT instances
            image_stats[img_id] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'gt_count': len(gt_boxes),
                'recall': tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
            }
    
    # Sort by recall (ascending) to find worst performers
    sorted_images = sorted(image_stats.items(), key=lambda x: (x[1]['recall'], -x[1]['gt_count']))
    
    print(f"  Top 10 worst performing images (by recall):")
    for idx, (img_id, stats) in enumerate(sorted_images[:10]):
        print(f"    {idx+1}. Image ID {img_id}: Recall={stats['recall']:.2f}, " + 
              f"TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}, GT={stats['gt_count']}")
    
    # 8. Save detailed report
    report_path = os.path.join(diag_dir, f'{class_name.lower()}_diagnostic_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"DIAGNOSTIC REPORT FOR CLASS: {class_name}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Detection Statistics:\n")
        f.write(f"  Total GT instances: {total_gt}\n")
        f.write(f"  Total predictions: {total_pred}\n")
        f.write(f"  Filtered predictions (score>={score_threshold}): {total_pred_filtered}\n\n")
        
        f.write(f"Matching Statistics (IoU>={iou_threshold}, Score>={score_threshold}):\n")
        f.write(f"  TP: {tp_count}, FP: {fp_count}, FN: {fn_count}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1-Score: {f1:.4f}\n\n")
        
        f.write(f"Per-Occlusion Analysis:\n")
        for occ_level in [0, 1, 2]:
            occ_name = ['NONE', 'PARTIAL', 'HEAVY'][occ_level]
            stats = occlusion_stats[occ_level]
            occ_precision = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
            occ_recall = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
            avg_iou = np.mean(stats['ious']) if len(stats['ious']) > 0 else 0
            
            f.write(f"  {occ_name}:\n")
            f.write(f"    GT: {stats['gt']}, TP: {stats['tp']}, FP: {stats['fp']}, FN: {stats['fn']}\n")
            f.write(f"    Precision: {occ_precision:.4f}, Recall: {occ_recall:.4f}\n")
            f.write(f"    Avg IoU: {avg_iou:.4f}\n\n")
        
        f.write(f"\nWorst Performing Images (Top 20):\n")
        for idx, (img_id, stats) in enumerate(sorted_images[:20]):
            f.write(f"  {idx+1}. Image ID {img_id}: Recall={stats['recall']:.4f}, " +
                   f"TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}, GT={stats['gt_count']}\n")

        # Add misclassification analysis to report
        if len(confusion_classes) > 0 and len(misclassifications) > 0:
            conf_class_names = ', '.join([name for _, name in confusion_classes])
            f.write(f"\n\nMisclassification Analysis ({class_name} -> {{{conf_class_names}}}):\n")
            f.write(f"  Total misclassifications: {len(misclassifications)}\n")
            unique_misclass_images = len(set(m['image_id'] for m in misclassifications))
            f.write(f"  Unique images with misclassifications: {unique_misclass_images}\n\n")

            # Breakdown by confusion class
            f.write(f"  By confusion class:\n")
            for conf_class_id, conf_class_name in confusion_classes:
                count = sum(1 for m in misclassifications if m['confusion_class_name'] == conf_class_name)
                f.write(f"    {class_name} -> {conf_class_name}: {count}\n")

            misclass_by_occ = {0: 0, 1: 0, 2: 0}
            for m in misclassifications:
                misclass_by_occ[m['gt_occlusion']] += 1
            f.write(f"\n  By occlusion level:\n")
            for occ_level in [0, 1, 2]:
                occ_name = ['NONE', 'PARTIAL', 'HEAVY'][occ_level]
                f.write(f"    {occ_name}: {misclass_by_occ[occ_level]}\n")

            misclass_scores = [m['pred_score'] for m in misclassifications]
            misclass_ious = [m['iou'] for m in misclassifications]
            f.write(f"\n  Score distribution:\n")
            f.write(f"    Mean: {np.mean(misclass_scores):.4f}, Min: {np.min(misclass_scores):.4f}, Max: {np.max(misclass_scores):.4f}\n")
            f.write(f"  IoU distribution:\n")
            f.write(f"    Mean: {np.mean(misclass_ious):.4f}, Min: {np.min(misclass_ious):.4f}, Max: {np.max(misclass_ious):.4f}\n")

            f.write(f"\n  Misclassified instances (sorted by score):\n")
            sorted_misclass = sorted(misclassifications, key=lambda x: -x['pred_score'])
            for idx, m in enumerate(sorted_misclass[:50]):  # Top 50
                occ_name = ['NONE', 'PARTIAL', 'HEAVY'][m['gt_occlusion']]
                f.write(f"    {idx+1}. Image ID {m['image_id']}: {class_name}->{m['confusion_class_name']}, Score={m['pred_score']:.4f}, IoU={m['iou']:.4f}, Occlusion={occ_name}\n")

    print(f"  Saved detailed report to: {report_path}")
    
    # 9. Save visualizations of worst performing images
    print(f"\n[Saving Failure Case Visualizations]")
    
    # Create subdirectory for failure images
    failure_img_dir = os.path.join(diag_dir, 'failure_cases')
    os.makedirs(failure_img_dir, exist_ok=True)
    
    # Get worst 20 images (or fewer if not enough problematic images)
    num_to_visualize = min(20, len([img for img, stats in sorted_images if stats['recall'] < 0.9]))
    
    print(f"  Visualizing top {num_to_visualize} worst performing images...")
    
    # Load and visualize images from COCO API
    saved_count = 0
    try:
        # Try to get image root directory from COCO API / dataset
        img_root = None
        
        # Try multiple ways to get the image folder
        if hasattr(coco_api, 'dataset'):
            # If coco_api wraps a dataset
            if hasattr(coco_api.dataset, 'img_folder'):
                img_root = coco_api.dataset.img_folder
        
        # If coco_api IS the dataset directly
        if img_root is None and hasattr(coco_api, 'img_folder'):
            img_root = coco_api.img_folder
        
        # Try to infer from annotation file path if available
        if img_root is None and hasattr(coco_api, 'dataset'):
            if hasattr(coco_api.dataset, 'ann_file'):
                ann_file = coco_api.dataset.ann_file
                # Typically images are in ../images/ relative to annotations
                ann_dir = os.path.dirname(ann_file)
                parent_dir = os.path.dirname(ann_dir)
                possible_img_folder = os.path.join(parent_dir, 'images')
                if os.path.exists(possible_img_folder):
                    img_root = possible_img_folder
        
        # Last resort: check common paths from config
        if img_root is None:
            common_paths = [
                './configs/dataset/MergedAll/images/',
                'configs/dataset/MergedAll/images/',
                './dataset/images/',
                './data/images/',
            ]
            for path in common_paths:
                if os.path.exists(path):
                    img_root = path
                    break
        
        if img_root:
            print(f"  Using image folder: {img_root}")
        
        for idx, (img_id, stats) in enumerate(sorted_images[:num_to_visualize]):
            try:
                # Get image info from COCO API
                img_info = coco_api.loadImgs(img_id)[0]
                img_filename = img_info.get('file_name', None)
                
                if img_filename is None:
                    continue
                
                # Construct full path
                possible_paths = []
                
                if img_root:
                    # Use the detected image folder
                    possible_paths.append(os.path.join(img_root, img_filename))
                    # Also try without joining in case filename is already relative to current dir
                    if not os.path.isabs(img_filename):
                        possible_paths.append(img_filename)
                
                # Try common structures as fallback
                possible_paths.extend([
                    img_filename,  # Absolute or relative path
                    os.path.join('./configs/dataset/MergedAll/images/', img_filename),
                    os.path.join('configs/dataset/MergedAll/images/', img_filename),
                    os.path.join('data', img_filename),
                    os.path.join('dataset', img_filename),
                    os.path.join('images', img_filename),
                ])
                
                # Find first existing path
                img_full_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        img_full_path = path
                        break
                
                if img_full_path is None:
                    if idx == 0:  # Only print warning once
                        print(f"    Warning: Could not find images. Tried paths like: {possible_paths[0]}")
                        print(f"    Skipping image visualization (image files not accessible)")
                    continue
                
                # Load image
                image = cv2.imread(img_full_path)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Find this image's predictions and targets from class-filtered lists
                img_pred = None
                img_target = None
                for p, t in zip(class_predictions, class_targets):
                    t_img_id = t['image_id'].item() if torch.is_tensor(t['image_id']) else t['image_id']
                    if t_img_id == img_id:
                        img_pred = p
                        img_target = t
                        break
                
                if img_pred is None or img_target is None:
                    continue
                
                # Create visualization with detailed annotations
                fig, ax = plt.subplots(1, 1, figsize=(16, 10))
                ax.imshow(image)
                
                # Get ground truth boxes and occlusions
                gt_boxes = img_target['boxes'].numpy() if len(img_target['boxes']) > 0 else np.array([])
                gt_occlusions = img_target['occlusion'].numpy() if len(img_target['occlusion']) > 0 else np.array([])
                
                # Get predictions
                pred_boxes = img_pred['boxes'].numpy() if len(img_pred['boxes']) > 0 else np.array([])
                pred_scores = img_pred['scores'].numpy() if len(img_pred['scores']) > 0 else np.array([])
                
                # Filter by score
                valid_mask = pred_scores >= score_threshold
                pred_boxes_filtered = pred_boxes[valid_mask]
                pred_scores_filtered = pred_scores[valid_mask]
                
                # Match predictions to GTs and track IoU values
                matched_gts = {}  # gt_idx -> (pred_idx, iou)
                pred_info = []  # List of (pred_box, pred_score, is_tp, matched_gt_idx, best_iou)
                
                for pred_idx, pred_box in enumerate(pred_boxes_filtered):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    # Find best matching GT
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    # Check if this is a TP
                    is_tp = False
                    if best_iou >= iou_threshold and best_gt_idx >= 0:
                        # Check if this GT wasn't already matched to a better prediction
                        if best_gt_idx not in matched_gts or matched_gts[best_gt_idx][1] < best_iou:
                            # Update or set the match
                            if best_gt_idx in matched_gts:
                                # Mark previous match as FP
                                old_pred_idx = matched_gts[best_gt_idx][0]
                                pred_info[old_pred_idx] = (pred_info[old_pred_idx][0], pred_info[old_pred_idx][1], 
                                                          False, -1, pred_info[old_pred_idx][4])
                            matched_gts[best_gt_idx] = (pred_idx, best_iou)
                            is_tp = True
                    
                    pred_info.append((pred_box, pred_scores_filtered[pred_idx], is_tp, 
                                    best_gt_idx if is_tp else -1, best_iou))
                
                # Draw ground truth boxes in GREEN (thicker for unmatched/FN)
                for gt_idx, (gt_box, gt_occ) in enumerate(zip(gt_boxes, gt_occlusions)):
                    x1, y1, x2, y2 = gt_box
                    width = x2 - x1
                    height = y2 - y1
                    occ_name = ['None', 'Partial', 'Heavy'][int(gt_occ)]
                    
                    # Thicker line for False Negatives (missed detections)
                    is_fn = gt_idx not in matched_gts
                    linewidth = 4 if is_fn else 2
                    linestyle = '-' if is_fn else '--'
                    
                    rect = plt.Rectangle((x1, y1), width, height, fill=False, 
                                        edgecolor='lime', linewidth=linewidth, linestyle=linestyle)
                    ax.add_patch(rect)
                    
                    # Label with occlusion and FN marker
                    label = f'GT-{occ_name}'
                    if is_fn:
                        label += ' [MISSED]'
                    
                    ax.text(x1, y1-5, label, color='lime', 
                           fontsize=9, weight='bold', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
                
                # Draw predictions with IoU information
                for pred_box, pred_score, is_tp, matched_gt_idx, best_iou in pred_info:
                    x1, y1, x2, y2 = pred_box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Color code: BLUE for TP, RED for FP
                    if is_tp:
                        color = 'cyan'
                        pred_type = 'TP'
                        label = f'{pred_type} {pred_score:.2f}\nIoU={best_iou:.2f}'
                    else:
                        color = 'red'
                        pred_type = 'FP'
                        if best_iou > 0:
                            label = f'{pred_type} {pred_score:.2f}\nIoU={best_iou:.2f} (below thresh)'
                        else:
                            label = f'{pred_type} {pred_score:.2f}\nNo overlap'
                    
                    rect = plt.Rectangle((x1, y1), width, height, fill=False, 
                                        edgecolor=color, linewidth=2, linestyle='-')
                    ax.add_patch(rect)
                    ax.text(x2+5, y1, label, color=color, 
                           fontsize=8, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
                
                # Add comprehensive title
                title = f"Image ID {img_id} - {class_name} Performance\n"
                title += f"Recall={stats['recall']:.2%} | "
                title += f"TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']}, GT={stats['gt_count']}\n"
                title += f"GREEN=Ground Truth (thick=missed), CYAN=True Positive (with IoU), RED=False Positive"
                
                ax.set_title(title, fontsize=11, weight='bold', pad=10)
                ax.axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(failure_img_dir, 
                                        f'failure_{idx+1:02d}_imgid_{img_id}_recall_{stats["recall"]:.2f}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                saved_count += 1
                
            except Exception as e:
                print(f"    Warning: Failed to visualize image {img_id}: {e}")
                continue
        
        print(f"  Successfully saved {saved_count} failure case visualizations to: {failure_img_dir}")
        
    except Exception as e:
        print(f"  Warning: Failed to save failure case visualizations: {e}")
        import traceback
        traceback.print_exc()

    # 10. Save misclassification visualizations
    if len(confusion_classes) > 0 and len(misclassifications) > 0:
        conf_class_names = ', '.join([name for _, name in confusion_classes])
        print(f"\n[Saving Misclassification Visualizations: {class_name} -> {{{conf_class_names}}}]")

        # Create separate directories for misclassified_as_car and misclassified_as_other
        misclass_car_dir = os.path.join(diag_dir, 'misclassified_as_car')
        misclass_other_dir = os.path.join(diag_dir, 'misclassified_as_other')
        os.makedirs(misclass_car_dir, exist_ok=True)
        os.makedirs(misclass_other_dir, exist_ok=True)

        # Sort by prediction score (highest first - most confident misclassifications)
        sorted_misclass = sorted(misclassifications, key=lambda x: -x['pred_score'])

        # Separate misclassifications into car vs other
        misclass_as_car = [m for m in sorted_misclass if m['confusion_class_name'].lower() == 'car']
        misclass_as_other = [m for m in sorted_misclass if m['confusion_class_name'].lower() != 'car']

        print(f"  Misclassified as car: {len(misclass_as_car)}")
        print(f"  Misclassified as other (motorcycle, truck, tricycle, bus): {len(misclass_as_other)}")

        num_car_to_visualize = min(30, len(misclass_as_car))
        num_other_to_visualize = min(30, len(misclass_as_other))
        print(f"  Visualizing top {num_car_to_visualize} car misclassifications and top {num_other_to_visualize} other misclassifications...")

        def save_misclass_visualizations(misclass_list, output_dir, max_count, category_label):
            """Helper function to save misclassification visualizations to a directory"""
            saved_count = 0
            for idx, misclass in enumerate(misclass_list[:max_count]):
                try:
                    img_id = misclass['image_id']
                    conf_class_name = misclass['confusion_class_name']
                    img_info = coco_api.loadImgs(img_id)[0]
                    file_name = img_info['file_name']

                    # Find image file
                    img_full_path = None
                    possible_paths = [
                        os.path.join(img_root, file_name) if img_root else None,
                        file_name,
                        os.path.join('images', file_name),
                    ]
                    possible_paths = [p for p in possible_paths if p is not None]

                    for path in possible_paths:
                        if os.path.exists(path):
                            img_full_path = path
                            break

                    if img_full_path is None:
                        if idx == 0:
                            print(f"    Warning: Could not find images for {category_label} misclassifications")
                        continue

                    # Load image
                    image = cv2.imread(img_full_path)
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Create visualization
                    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
                    ax.imshow(image)

                    # Draw GT box in GREEN
                    gt_box = misclass['gt_box']
                    x1, y1, x2, y2 = gt_box
                    width = x2 - x1
                    height = y2 - y1
                    occ_name = ['None', 'Partial', 'Heavy'][misclass['gt_occlusion']]

                    rect = plt.Rectangle((x1, y1), width, height, fill=False,
                                        edgecolor='lime', linewidth=3, linestyle='-')
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f'GT: {class_name} ({occ_name})', color='lime',
                           fontsize=10, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

                    # Draw misclassified prediction box in ORANGE
                    pred_box = misclass['pred_box']
                    x1, y1, x2, y2 = pred_box
                    width = x2 - x1
                    height = y2 - y1

                    rect = plt.Rectangle((x1, y1), width, height, fill=False,
                                        edgecolor='orange', linewidth=3, linestyle='-')
                    ax.add_patch(rect)
                    ax.text(x2+5, y1, f'Pred: {conf_class_name} [MISCLASS]\nScore={misclass["pred_score"]:.2f}\nIoU={misclass["iou"]:.2f}',
                           color='orange', fontsize=9, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

                    # Add title
                    title = f"MISCLASSIFICATION: {class_name} predicted as {conf_class_name}\n"
                    title += f"Image ID {img_id} | Score={misclass['pred_score']:.2f} | IoU={misclass['iou']:.2f} | Occlusion={occ_name}\n"
                    title += f"GREEN=Ground Truth ({class_name}), ORANGE=Misclassified Prediction ({conf_class_name})"

                    ax.set_title(title, fontsize=11, weight='bold', pad=10, color='darkorange')
                    ax.axis('off')

                    plt.tight_layout()
                    save_path = os.path.join(output_dir,
                                            f'misclass_{idx+1:02d}_{class_name}_as_{conf_class_name}_imgid_{img_id}_score_{misclass["pred_score"]:.2f}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    saved_count += 1

                except Exception as e:
                    print(f"    Warning: Failed to visualize {category_label} misclassification for image {misclass.get('image_id', 'unknown')}: {e}")
                    continue
            return saved_count

        try:
            # Save misclassified as car
            car_saved = save_misclass_visualizations(misclass_as_car, misclass_car_dir, num_car_to_visualize, "car")
            print(f"  Successfully saved {car_saved} 'misclassified as car' visualizations to: {misclass_car_dir}")

            # Save misclassified as other (motorcycle, truck, tricycle, bus)
            other_saved = save_misclass_visualizations(misclass_as_other, misclass_other_dir, num_other_to_visualize, "other")
            print(f"  Successfully saved {other_saved} 'misclassified as other' visualizations to: {misclass_other_dir}")

        except Exception as e:
            print(f"  Warning: Failed to save misclassification visualizations: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}\n")

    return {
        'tp': tp_count,
        'fp': fp_count,
        'fn': fn_count,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'occlusion_stats': occlusion_stats,
        'worst_images': sorted_images[:10],
        'misclassifications': misclassifications if len(confusion_classes) > 0 else []
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir, epoch=None, save_visualizations=True, save_logs=False):
    model.eval()
    criterion.eval()

    # Note: We'll setup log redirection AFTER data loading to avoid multiprocessing issues
    # Store the flag and path for later
    should_save_logs = save_logs and output_dir
    log_path = os.path.join(output_dir, 'test_logs.txt') if should_save_logs else None

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    
    # Create standard evaluator for general metrics
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None

    # Store samples for visualization - collect 1 random image from each batch
    vis_samples = []
    vis_predictions = []
    vis_targets = []
    import random
    import time
    
    # For advanced metrics
    inference_times = []
    all_predictions = []  # Store all predictions for advanced metrics
    all_targets = []  # Store all targets

    for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Measure inference time
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        outputs = model(samples)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_times.append(time.time() - start_time)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        
        # Store all predictions and targets for advanced metrics
        first_image_processed = False
        for idx, (pred, target) in enumerate(zip(results, targets)):
            # Scale target boxes to original image size to match predictions
            # Boxes in target are in model input space (640x640), predictions are in original space
            # We need to scale from model input size to orig_size
            # orig_size is stored as [width, height] in coco_dataset.py line 167
            orig_w, orig_h = target['orig_size'][0].item(), target['orig_size'][1].item()
            
            # Model input size is 640x640 (from config)
            model_input_size = 640
            h_scale = orig_h / model_input_size
            w_scale = orig_w / model_input_size
            
            scaled_target_boxes = target['boxes'].clone().cpu()
            scaled_target_boxes[:, [0, 2]] *= w_scale  # x coordinates
            scaled_target_boxes[:, [1, 3]] *= h_scale  # y coordinates
            
            all_predictions.append({
                'boxes': pred['boxes'].cpu(),
                'scores': pred['scores'].cpu(),
                'labels': pred['labels'].cpu(),
                'image_id': target['image_id'].cpu()
            })
            all_targets.append({
                'boxes': scaled_target_boxes,
                'labels': target['labels'].cpu(),
                'image_id': target['image_id'].cpu(),
                'occlusion': target.get('occlusion', torch.zeros(len(scaled_target_boxes), dtype=torch.int64)).cpu()
            })

        # Store 1 random image from every 10th batch for visualization
        if save_visualizations and epoch is not None and len(samples) > 0 and batch_idx % 10 == 0:
            # Randomly select one image index from this batch
            random_idx = random.randint(0, len(samples) - 1)
            
            # Store the selected image
            vis_samples.append(samples[random_idx:random_idx+1].cpu())
            
            # Deep copy the prediction and target dictionaries for the selected image
            pred_copy = {k: v.cpu() if torch.is_tensor(v) else v for k, v in results[random_idx].items()}
            target_copy = {k: v.cpu() if torch.is_tensor(v) else v for k, v in targets[random_idx].items()}
            vis_predictions.append(pred_copy)
            vis_targets.append(target_copy)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # NOW setup log redirection after data loading is complete (to avoid multiprocessing issues)
    log_file = None
    original_stdout = None
    log_buffer = []  # Store log lines for summary
    
    if should_save_logs:
        log_file = open(log_path, 'w', encoding='utf-8')
        original_stdout = sys.stdout
        
        # Create a custom writer that writes to both stdout and file, and stores in buffer
        class TeeWriter:
            def __init__(self, *files, buffer=None):
                self.files = files
                self.buffer = buffer
            def write(self, text):
                for f in self.files:
                    f.write(text)
                    f.flush()
                if self.buffer is not None:
                    self.buffer.append(text)
            def flush(self):
                for f in self.files:
                    f.flush()
        
        sys.stdout = TeeWriter(original_stdout, log_file, buffer=log_buffer)
        print(f"Logging test results to: {log_path}")
        print("="*80)
    
    print("Averaged stats:", metric_logger)
    
    # Compute FPS/Inference Speed
    if len(inference_times) > 0:
        avg_inference_time = sum(inference_times) / len(inference_times)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        print(f"\n[Performance] Average inference time: {avg_inference_time*1000:.2f}ms, FPS: {fps:.2f}")
    else:
        avg_inference_time = 0
        fps = 0
    
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        print("\n[General Metrics - No Occlusion Segregation]")
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    # Compute advanced metrics
    advanced_metrics = {}
    confusion_matrix = None
    
    if coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
        coco_eval = coco_evaluator.coco_eval['bbox']
        precision = coco_eval.eval['precision']
        recall = coco_eval.eval['recall']
        
        # Get number of categories
        num_categories = len(base_ds.getCatIds()) if hasattr(base_ds, 'getCatIds') else precision.shape[2]
        
        # Per-class AP and AR
        print("\n[Per-Class Metrics - COCO AP/AR averaged across IoU=0.5:0.95]")
        per_class_ap = {}
        per_class_ar = {}
        
        for cat_idx in range(num_categories):
            # AP per class (average over all IoU thresholds)
            cat_precision = precision[:, :, cat_idx, 0, 2]  # All IoU, all recall, this cat, all areas, maxDet=100
            valid_cat_precision = cat_precision[cat_precision > -1]
            ap = valid_cat_precision.mean() if len(valid_cat_precision) > 0 else 0.0
            
            # AR per class (average over all IoU thresholds)
            cat_recall = recall[:, cat_idx, 0, 2]  # All IoU, this cat, all areas, maxDet=100
            valid_cat_recall = cat_recall[cat_recall > -1]
            ar = valid_cat_recall.mean() if len(valid_cat_recall) > 0 else 0.0
            
            # Get category name
            cat_name = f"class_{cat_idx}"
            if hasattr(base_ds, 'loadCats'):
                try:
                    cat_ids = base_ds.getCatIds()
                    cats = base_ds.loadCats([cat_ids[cat_idx]])
                    if cats:
                        cat_name = cats[0]['name']
                except:
                    pass
            
            per_class_ap[cat_name] = ap
            per_class_ar[cat_name] = ar
            print(f"  {cat_name}: AP={ap:.4f}, AR={ar:.4f}")
            
            advanced_metrics[f'AP_{cat_name}'] = ap
            advanced_metrics[f'AR_{cat_name}'] = ar
        
        # Per-class AP/AR at specific IoU thresholds to show degradation
        print("\n[Per-Class Metrics at Specific IoU Thresholds]")
        # IoU thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        # Indices:        [0,   1,    2,   3,    4,   5,    6,   7,    8,   9]
        iou_thresholds_to_show = {
            '0.50': 0,   # Lenient
            '0.75': 5,   # Medium
            '0.90': 8    # Strict
        }
        
        for cat_idx in range(num_categories):
            # Get category name
            cat_name = f"class_{cat_idx}"
            if hasattr(base_ds, 'loadCats'):
                try:
                    cat_ids = base_ds.getCatIds()
                    cats = base_ds.loadCats([cat_ids[cat_idx]])
                    if cats:
                        cat_name = cats[0]['name']
                except:
                    pass
            
            # Compute AP and AR at each specific IoU threshold
            metrics_str = f"  {cat_name:12}"
            for iou_name, iou_idx in iou_thresholds_to_show.items():
                # AP at this IoU threshold
                cat_precision_iou = precision[iou_idx, :, cat_idx, 0, 2]  # This IoU, all recall, this cat
                valid_cat_precision_iou = cat_precision_iou[cat_precision_iou > -1]
                ap_iou = valid_cat_precision_iou.mean() if len(valid_cat_precision_iou) > 0 else 0.0
                
                # AR at this IoU threshold
                cat_recall_iou = recall[iou_idx, cat_idx, 0, 2]  # This IoU, this cat
                ar_iou = cat_recall_iou if cat_recall_iou > -1 else 0.0
                
                metrics_str += f" | AP@{iou_name}={ap_iou:.4f}, AR@{iou_name}={ar_iou:.4f}"
                
                # Store in advanced metrics
                advanced_metrics[f'AP_{cat_name}@{iou_name}'] = ap_iou
                advanced_metrics[f'AR_{cat_name}@{iou_name}'] = ar_iou
            
            print(metrics_str)
        
        # Compute overall Precision and Recall at IoU=0.5
        print("\n[Overall Precision/Recall/F1 at IoU=0.5]")
        # Note: COCO precision/recall arrays are computed across different confidence thresholds
        # Shape: precision[T, R, K, A, M] where:
        #   T = IoU thresholds (10: 0.5:0.05:0.95)
        #   R = recall thresholds (101: 0:0.01:1)
        #   K = categories
        #   A = area ranges (4: all, small, medium, large)
        #   M = max detections (3: 1, 10, 100)
        
        # For Precision/Recall at IoU=0.5, we compute the maximum precision achieved
        # across all recall thresholds for each class, then average across classes.
        # This represents the best precision the model can achieve at IoU=0.5.
        
        # Precision at IoU=0.5 - take maximum precision across all recall thresholds
        # precision[0, :, k, 0, 2] gives precision at all recall levels for class k at IoU=0.5
        precision_50_per_class = []
        for k in range(precision.shape[2]):  # iterate over classes
            prec_curve = precision[0, :, k, 0, 2]  # precision-recall curve for this class at IoU=0.5
            # Filter out -1 values (no predictions)
            valid_prec = prec_curve[prec_curve > -1]
            if len(valid_prec) > 0:
                # Take maximum precision across the P-R curve
                precision_50_per_class.append(valid_prec.max())
        overall_precision_50 = np.mean(precision_50_per_class) if len(precision_50_per_class) > 0 else 0.0
        
        # Recall at IoU=0.5 - maximum recall achievable (at lowest confidence threshold)
        recall_50_all_classes = recall[0, :, 0, 2]  # [K] for IoU=0.5, maxDet=100
        valid_recall_50 = recall_50_all_classes[recall_50_all_classes > -1]
        overall_recall_50 = valid_recall_50.mean() if len(valid_recall_50) > 0 else 0.0
        
        # F1-Score at IoU=0.5
        if overall_precision_50 > 0 and overall_recall_50 > 0:
            overall_f1_50 = 2 * (overall_precision_50 * overall_recall_50) / (overall_precision_50 + overall_recall_50)
        else:
            overall_f1_50 = 0.0
        
        print(f"  Overall Precision@IoU=0.5: {overall_precision_50:.4f}")
        print(f"  Overall Recall@IoU=0.5: {overall_recall_50:.4f}")
        print(f"  Overall F1-Score@IoU=0.5: {overall_f1_50:.4f}")
        
        advanced_metrics['precision_0.5'] = overall_precision_50
        advanced_metrics['recall_0.5'] = overall_recall_50
        advanced_metrics['f1_score_0.5'] = overall_f1_50
        
        # Compute overall Precision and Recall at IoU=0.75
        print("\n[Overall Precision/Recall/F1 at IoU=0.75]")
        # IoU=0.75 is at index 5 in COCO thresholds (0.5, 0.55, 0.60, 0.65, 0.70, 0.75, ...)
        # COCO uses: [0.50:0.05:0.95] so index 5 = 0.75
        
        # Precision at IoU=0.75 - take maximum precision across all recall thresholds
        precision_75_per_class = []
        for k in range(precision.shape[2]):  # iterate over classes
            prec_curve = precision[5, :, k, 0, 2]  # precision-recall curve for this class at IoU=0.75
            # Filter out -1 values (no predictions)
            valid_prec = prec_curve[prec_curve > -1]
            if len(valid_prec) > 0:
                # Take maximum precision across the P-R curve
                precision_75_per_class.append(valid_prec.max())
        overall_precision_75 = np.mean(precision_75_per_class) if len(precision_75_per_class) > 0 else 0.0
        
        # Recall at IoU=0.75 - maximum recall achievable
        recall_75_all_classes = recall[5, :, 0, 2]  # [K] for IoU=0.75, maxDet=100
        valid_recall_75 = recall_75_all_classes[recall_75_all_classes > -1]
        overall_recall_75 = valid_recall_75.mean() if len(valid_recall_75) > 0 else 0.0
        
        # F1-Score at IoU=0.75
        if overall_precision_75 > 0 and overall_recall_75 > 0:
            overall_f1_75 = 2 * (overall_precision_75 * overall_recall_75) / (overall_precision_75 + overall_recall_75)
        else:
            overall_f1_75 = 0.0
        
        print(f"  Overall Precision@IoU=0.75: {overall_precision_75:.4f}")
        print(f"  Overall Recall@IoU=0.75: {overall_recall_75:.4f}")
        print(f"  Overall F1-Score@IoU=0.75: {overall_f1_75:.4f}")
        
        advanced_metrics['precision_0.75'] = overall_precision_75
        advanced_metrics['recall_0.75'] = overall_recall_75
        advanced_metrics['f1_score_0.75'] = overall_f1_75
        
        # Compute Confusion Matrix and False Positive Rate
        print("\n[Computing Confusion Matrix and FP Rate]")
        confusion_matrix = compute_confusion_matrix(all_predictions, all_targets, num_categories, iou_threshold=0.5, score_threshold=0.3)
        
        # Print confusion matrix
        # Get class names and IDs - ensure they're sorted
        class_names = []
        cat_ids = []
        if hasattr(base_ds, 'loadCats'):
            try:
                cat_ids = sorted(base_ds.getCatIds())  # Ensure sorted order
                cats = base_ds.loadCats(cat_ids)
                # Sort cats by id to match cat_ids order
                cats_dict = {cat['id']: cat for cat in cats}
                class_names = [cats_dict[cat_id]['name'] for cat_id in cat_ids]
            except:
                cat_ids = list(range(1, num_categories + 1))
                class_names = [f"class_{i}" for i in cat_ids]
        else:
            cat_ids = list(range(1, num_categories + 1))
            class_names = [f"class_{i}" for i in cat_ids]
        
        print("\nConfusion Matrix (rows=predicted, cols=ground truth):")
        # Print header
        print("        ", end="")
        for name in class_names:
            print(f"{name[:8]:>8}", end=" ")
        print()
        
        # Print matrix with row labels
        # Note: Actual labels in predictions/targets are 0-indexed [0-5]
        # but cat_ids from COCO API are 1-indexed [1-6]
        # So we use index i (0-5) to access the matrix, not cat_id
        for i, (cat_id, row_name) in enumerate(zip(cat_ids, class_names)):
            print(f"{row_name[:8]:>8}", end=" ")
            for j, gt_cat_id in enumerate(cat_ids):
                # Use index i,j (0-based) to access the matrix, not cat_id
                print(f"{int(confusion_matrix[i, j]):>8}", end=" ")
            print()
        
        # Save confusion matrix to advanced_metrics (only the relevant portion)
        relevant_confusion = np.zeros((num_categories, num_categories), dtype=np.float32)
        for i, cat_id_i in enumerate(cat_ids):
            for j, cat_id_j in enumerate(cat_ids):
                # Use indices i,j (0-based) to access the matrix
                relevant_confusion[i, j] = confusion_matrix[i, j]
        advanced_metrics['confusion_matrix'] = relevant_confusion.tolist()
        advanced_metrics['confusion_matrix_classes'] = class_names
        
        # False Positive Rate per class
        fp_rates = {}
        for i, (cat_id, cat_name) in enumerate(zip(cat_ids, class_names)):
            # Use index i (0-based) to access the matrix, not cat_id
            # FP = predicted as this class but wrong (either wrong class or no match)
            fp = confusion_matrix[i, :].sum() - confusion_matrix[i, i]
            tp = confusion_matrix[i, i]
            fp_rate = fp / (fp + tp) if (fp + tp) > 0 else 0.0
            
            fp_rates[cat_name] = fp_rate
            advanced_metrics[f'FPR_{cat_name}'] = fp_rate
            print(f"  {cat_name}: FPR@IoU=0.5={fp_rate:.4f}")
        
        # Overall FP rate - compute only for relevant classes
        total_fp = 0
        total_tp = 0
        for i in range(num_categories):
            # Use index i (0-based) to access the matrix
            total_tp += confusion_matrix[i, i]
            total_fp += confusion_matrix[i, :].sum() - confusion_matrix[i, i]
        overall_fp_rate = total_fp / (total_fp + total_tp) if (total_fp + total_tp) > 0 else 0.0
        advanced_metrics['FPR_overall'] = overall_fp_rate
        print(f"  Overall FPR@IoU=0.5: {overall_fp_rate:.4f}")
        
        # Compute TP/FP/FN counts from predictions and ground truth
        # Note: Confusion matrix only contains matched predictions, not all GTs
        print("\n[TP/FP/FN Counts at IoU=0.5, Score>=0.3]")
        
        total_tp_count = 0
        total_fp_count = 0
        total_fn_count = 0
        total_gt_count = 0
        
        # Count from all_predictions and all_targets
        for pred, target in zip(all_predictions, all_targets):
            pred_boxes = pred['boxes'].cpu().numpy() if pred['boxes'].is_cuda else pred['boxes'].numpy()
            pred_scores = pred['scores'].cpu().numpy() if pred['scores'].is_cuda else pred['scores'].numpy()
            pred_labels = pred['labels'].cpu().numpy() if pred['labels'].is_cuda else pred['labels'].numpy()
            
            gt_boxes = target['boxes'].cpu().numpy() if target['boxes'].is_cuda else target['boxes'].numpy()
            gt_labels = target['labels'].cpu().numpy() if target['labels'].is_cuda else target['labels'].numpy()
            
            total_gt_count += len(gt_boxes)
            
            # Filter predictions by score threshold
            valid_mask = pred_scores >= 0.3
            pred_boxes_filtered = pred_boxes[valid_mask]
            pred_labels_filtered = pred_labels[valid_mask]
            pred_scores_filtered = pred_scores[valid_mask]
            
            # Track matched ground truths
            matched_gts = set()
            
            # Sort predictions by score (highest first)
            sorted_indices = np.argsort(-pred_scores_filtered)
            
            for pred_idx in sorted_indices:
                pred_box = pred_boxes_filtered[pred_idx]
                pred_label = pred_labels_filtered[pred_idx]
                
                # Find best matching ground truth with same class
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                    if gt_idx in matched_gts:
                        continue
                    if gt_label != pred_label:
                        continue
                    
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # If IoU >= 0.5 and class matches, it's a TP
                if best_iou >= 0.5 and best_gt_idx >= 0:
                    matched_gts.add(best_gt_idx)
                    total_tp_count += 1
                else:
                    # Otherwise it's a FP
                    total_fp_count += 1
            
            # Unmatched GTs are FNs
            total_fn_count += (len(gt_boxes) - len(matched_gts))
        
        print(f"  True Positives (TP): {total_tp_count}")
        print(f"  False Positives (FP): {total_fp_count}")
        print(f"  False Negatives (FN): {total_fn_count}")
        print(f"  Ground Truth Objects: {total_gt_count}")
        print(f"  Precision (TP/(TP+FP)): {total_tp_count/(total_tp_count+total_fp_count) if (total_tp_count+total_fp_count) > 0 else 0:.4f}")
        print(f"  Recall (TP/(TP+FN)): {total_tp_count/(total_tp_count+total_fn_count) if (total_tp_count+total_fn_count) > 0 else 0:.4f}")
        
        advanced_metrics['total_tp'] = total_tp_count
        advanced_metrics['total_fp'] = total_fp_count
        advanced_metrics['total_fn'] = total_fn_count
        advanced_metrics['total_gt'] = total_gt_count
        
        # Compute Per-Occlusion Level Metrics using same logic as diagnostics
        print("\n[Per-Occlusion Level Metrics - OBJECT-LEVEL AGGREGATION]")
        print("NOTE: This section uses individual object occlusion annotations.")
        print("      Each object is grouped by its own occlusion level (NONE/PARTIAL/HEAVY).")
        print("      Each prediction is assigned to exactly ONE occlusion level to avoid double-counting.")
        print("      Using same calculation method as comprehensive diagnostics section.")
        print("\nGrouping objects by their occlusion level...")
        
        # Mapping from numeric levels to string names
        occlusion_names = {0: 'none', 1: 'partial', 2: 'heavy'}
        
        # Use same data structure as comprehensive_occlusion_diagnostics
        occlusion_metrics_data = {
            0: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'ious': [], 'class_breakdown': {}},
            1: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'ious': [], 'class_breakdown': {}},
            2: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'ious': [], 'class_breakdown': {}}
        }
        
        unique_images_per_level = {0: set(), 1: set(), 2: set()}
        
        # Get class names and build robust mapping
        # Build mapping from model labels (0-indexed) to class names
        class_names_dict = {}  # Maps 0-indexed label -> class name
        class_name_to_label = {}  # Maps class name -> 0-indexed label
        cat_id_to_label = {}  # Maps COCO cat_id -> 0-indexed label
        label_to_cat_id = {}  # Maps 0-indexed label -> COCO cat_id
        
        if hasattr(base_ds, 'loadCats'):
            try:
                cat_ids = sorted(base_ds.getCatIds())
                cats = base_ds.loadCats(cat_ids)
                
                # Detect if dataset is 0-indexed or 1-indexed
                min_cat_id = min(cat_ids) if cat_ids else 1
                is_zero_indexed = (min_cat_id == 0)
                
                # Build mappings based on detected indexing
                for cat in cats:
                    coco_cat_id = cat['id']
                    class_name = cat['name']
                    
                    if is_zero_indexed:
                        # 0-indexed dataset: category_id IS the label
                        zero_indexed_label = coco_cat_id
                    else:
                        # 1-indexed dataset: convert category_id to 0-indexed label
                        zero_indexed_label = coco_cat_id - 1
                    
                    # Store mappings (bidirectional)
                    class_names_dict[zero_indexed_label] = class_name
                    class_name_to_label[class_name] = zero_indexed_label
                    cat_id_to_label[coco_cat_id] = zero_indexed_label
                    label_to_cat_id[zero_indexed_label] = coco_cat_id
                    
                print(f"\n[Per-Occlusion Metrics] Detected {'0-indexed' if is_zero_indexed else '1-indexed'} dataset")
                print(f"[Per-Occlusion Metrics] Class mapping: {class_names_dict}")
            except Exception as e:
                print(f"[Per-Occlusion Metrics] Warning: Could not build class mapping: {e}")
                pass
        
        # Process each image
        for pred, target in zip(all_predictions, all_targets):
            pred_boxes = pred['boxes'].cpu().numpy() if len(pred['boxes']) > 0 else np.array([])
            pred_scores = pred['scores'].cpu().numpy() if len(pred['scores']) > 0 else np.array([])
            pred_labels = pred['labels'].cpu().numpy() if len(pred['labels']) > 0 else np.array([])
            
            gt_boxes = target['boxes'].cpu().numpy() if len(target['boxes']) > 0 else np.array([])
            gt_labels = target['labels'].cpu().numpy() if len(target['labels']) > 0 else np.array([])
            gt_occlusions = target['occlusion'].cpu().numpy() if 'occlusion' in target and len(target['occlusion']) > 0 else np.array([])
            img_id = target['image_id'].item()
            
            # Count GT objects by occlusion level
            for gt_occ, gt_label in zip(gt_occlusions, gt_labels):
                occ_level = int(gt_occ)
                occlusion_metrics_data[occ_level]['gt'] += 1
                unique_images_per_level[occ_level].add(img_id)
                
                # Track per-class GT counts
                class_name = class_names_dict.get(int(gt_label), f"class_{int(gt_label)}")
                if class_name not in occlusion_metrics_data[occ_level]['class_breakdown']:
                    occlusion_metrics_data[occ_level]['class_breakdown'][class_name] = {'gt': 0, 'tp': 0, 'fn': 0, 'fp': 0}
                occlusion_metrics_data[occ_level]['class_breakdown'][class_name]['gt'] += 1
            
            # Filter predictions by score threshold (0.3)
            valid_mask = pred_scores >= 0.3
            pred_boxes_filtered = pred_boxes[valid_mask]
            pred_scores_filtered = pred_scores[valid_mask]
            pred_labels_filtered = pred_labels[valid_mask]
            
            matched_gts = set()
            
            # Sort predictions by score (descending)
            sorted_indices = np.argsort(-pred_scores_filtered)
            
            # Match predictions to GTs
            for pred_idx in sorted_indices:
                pred_box = pred_boxes_filtered[pred_idx]
                pred_label = pred_labels_filtered[pred_idx]
                
                best_iou = 0
                best_gt_idx = -1
                best_gt_occ = -1
                best_gt_label = -1
                
                # Find best matching GT
                for gt_idx, (gt_box, gt_label, gt_occ) in enumerate(zip(gt_boxes, gt_labels, gt_occlusions)):
                    if gt_idx in matched_gts:
                        continue
                    
                    # Must match class (both pred_label and gt_label are 0-indexed)
                    if int(pred_label) != int(gt_label):
                        continue
                    
                    # Compute IoU
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        best_gt_occ = int(gt_occ)
                        best_gt_label = int(gt_label)
                
                # Classify as TP or FP
                if best_iou >= 0.5 and best_gt_idx >= 0:
                    # True Positive - assign to GT's occlusion level
                    matched_gts.add(best_gt_idx)
                    occlusion_metrics_data[best_gt_occ]['tp'] += 1
                    occlusion_metrics_data[best_gt_occ]['ious'].append(best_iou)
                    
                    # Track per-class TP
                    class_name = class_names_dict.get(best_gt_label, f"class_{best_gt_label}")
                    if class_name in occlusion_metrics_data[best_gt_occ]['class_breakdown']:
                        occlusion_metrics_data[best_gt_occ]['class_breakdown'][class_name]['tp'] += 1
                else:
                    # False Positive - assign to occlusion level of closest GT (same class) or dominant occlusion
                    if best_gt_idx >= 0:
                        # Has a GT of same class (but IoU too low)
                        occlusion_metrics_data[best_gt_occ]['fp'] += 1
                        
                        # Track per-class FP
                        class_name = class_names_dict.get(best_gt_label, f"class_{best_gt_label}")
                        if class_name in occlusion_metrics_data[best_gt_occ]['class_breakdown']:
                            occlusion_metrics_data[best_gt_occ]['class_breakdown'][class_name]['fp'] += 1
                    elif len(gt_occlusions) > 0:
                        # Assign to dominant occlusion level in image
                        dominant_occ = int(np.bincount(gt_occlusions.astype(int)).argmax())
                        occlusion_metrics_data[dominant_occ]['fp'] += 1
            
            # Count False Negatives
            for gt_idx, (gt_occ, gt_label) in enumerate(zip(gt_occlusions, gt_labels)):
                if gt_idx not in matched_gts:
                    occ_level = int(gt_occ)
                    occlusion_metrics_data[occ_level]['fn'] += 1
                    
                    # Track per-class FN
                    class_name = class_names_dict.get(int(gt_label), f"class_{int(gt_label)}")
                    if class_name in occlusion_metrics_data[occ_level]['class_breakdown']:
                        occlusion_metrics_data[occ_level]['class_breakdown'][class_name]['fn'] += 1
        
        # Now create filtered pred/target lists for proper COCO mAP computation
        occlusion_predictions = {0: [], 1: [], 2: []}
        occlusion_targets = {0: [], 1: [], 2: []}
        
        for pred, target in zip(all_predictions, all_targets):
            gt_occlusions = target.get('occlusion', torch.zeros(len(target['boxes']), dtype=torch.int64))
            
            if len(gt_occlusions) == 0:
                continue
            
            # For each occlusion level, filter GT objects
            for occ_level in [0, 1, 2]:
                occ_mask = gt_occlusions == occ_level
                
                if not occ_mask.any():
                    continue
                
                # Create filtered target with only objects at this occlusion level
                filtered_target = {
                    'boxes': target['boxes'][occ_mask],
                    'labels': target['labels'][occ_mask],
                    'image_id': target['image_id'],
                    'occlusion': target['occlusion'][occ_mask] if 'occlusion' in target else torch.full((occ_mask.sum(),), occ_level, dtype=torch.int64)
                }
                
                # Keep all predictions (they'll be matched against filtered GTs during COCO eval)
                filtered_pred = {
                    'boxes': pred['boxes'],
                    'scores': pred['scores'],
                    'labels': pred['labels'],
                    'image_id': pred['image_id']
                }
                
                occlusion_predictions[occ_level].append(filtered_pred)
                occlusion_targets[occ_level].append(filtered_target)
        
        # Display metrics for each occlusion level
        for occ_level in [0, 1, 2]:  # Always show all three levels
            occ_name = occlusion_names[occ_level]
            data = occlusion_metrics_data[occ_level]
            
            print(f"\n  Occlusion Level: {occ_name.upper()} (level {occ_level})")
            print(f"    Number of GT objects at this level: {data['gt']}")
            print(f"    Number of images with objects at this level: {len(unique_images_per_level[occ_level])}")
            
            if data['gt'] == 0:
                print(f"    No objects at this occlusion level - skipping metrics")
                # Store zeros for missing levels
                advanced_metrics[f'occ_{occ_name}_num_objects'] = 0
                advanced_metrics[f'occ_{occ_name}_num_images'] = 0
                advanced_metrics[f'occ_{occ_name}_mAP_0.5_0.95'] = 0.0
                advanced_metrics[f'occ_{occ_name}_mAP_0.5'] = 0.0
                advanced_metrics[f'occ_{occ_name}_precision'] = 0.0
                advanced_metrics[f'occ_{occ_name}_recall'] = 0.0
                advanced_metrics[f'occ_{occ_name}_f1_score'] = 0.0
                advanced_metrics[f'occ_{occ_name}_fpr'] = 0.0
                continue
            
            # Print class distribution using class_breakdown
            print(f"    Class distribution: ", end="")
            for class_name in sorted(data['class_breakdown'].keys()):
                gt_count = data['class_breakdown'][class_name]['gt']
                print(f"{class_name}={gt_count}, ", end="")
            print()
            
            # Compute metrics at multiple IoU thresholds
            # Store metrics at each threshold for proper mAP calculation
            ap_values = []  # For mAP@0.5:0.95
            
            # Compute AP at IoU thresholds from 0.5 to 0.95 in steps of 0.05
            iou_thresholds = [0.5 + i * 0.05 for i in range(10)]  # 0.5, 0.55, 0.60, ..., 0.95
            
            for iou_thresh in iou_thresholds:
                # Recompute TP/FP/FN at this IoU threshold
                tp_at_thresh = 0
                fp_at_thresh = 0
                fn_at_thresh = data['gt']  # Start with all GT as FN
                
                # Re-process predictions at this threshold
                for pred, target in zip(all_predictions, all_targets):
                    pred_boxes = pred['boxes'].cpu().numpy() if len(pred['boxes']) > 0 else np.array([])
                    pred_scores = pred['scores'].cpu().numpy() if len(pred['scores']) > 0 else np.array([])
                    pred_labels = pred['labels'].cpu().numpy() if len(pred['labels']) > 0 else np.array([])
                    
                    gt_boxes = target['boxes'].cpu().numpy() if len(target['boxes']) > 0 else np.array([])
                    gt_labels = target['labels'].cpu().numpy() if len(target['labels']) > 0 else np.array([])
                    gt_occlusions = target['occlusion'].cpu().numpy() if 'occlusion' in target and len(target['occlusion']) > 0 else np.array([])
                    
                    # Filter by score threshold
                    valid_mask = pred_scores >= 0.3
                    pred_boxes_filtered = pred_boxes[valid_mask]
                    pred_scores_filtered = pred_scores[valid_mask]
                    pred_labels_filtered = pred_labels[valid_mask]
                    
                    matched_gts_local = set()
                    sorted_indices = np.argsort(-pred_scores_filtered)
                    
                    for pred_idx in sorted_indices:
                        pred_box = pred_boxes_filtered[pred_idx]
                        pred_label = pred_labels_filtered[pred_idx]
                        
                        best_iou = 0
                        best_gt_idx = -1
                        best_gt_occ = -1
                        
                        for gt_idx, (gt_box, gt_label, gt_occ) in enumerate(zip(gt_boxes, gt_labels, gt_occlusions)):
                            if gt_idx in matched_gts_local:
                                continue
                            if int(pred_label) != int(gt_label):
                                continue
                            
                            iou = compute_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                                best_gt_occ = int(gt_occ)
                        
                        # Check if this is a match at current IoU threshold
                        if best_iou >= iou_thresh and best_gt_idx >= 0 and best_gt_occ == occ_level:
                            matched_gts_local.add(best_gt_idx)
                            tp_at_thresh += 1
                        elif best_gt_idx >= 0 and best_gt_occ == occ_level:
                            # IoU too low but same occlusion level
                            fp_at_thresh += 1
                        elif len(gt_occlusions) > 0:
                            # Check if should be counted as FP for this occlusion level
                            if best_gt_idx >= 0:
                                if best_gt_occ == occ_level:
                                    fp_at_thresh += 1
                            else:
                                # Assign to dominant occlusion
                                dominant_occ = int(np.bincount(gt_occlusions.astype(int)).argmax())
                                if dominant_occ == occ_level:
                                    fp_at_thresh += 1
                
                # Compute precision at this IoU threshold (AP approximation)
                precision_at_thresh = tp_at_thresh / (tp_at_thresh + fp_at_thresh) if (tp_at_thresh + fp_at_thresh) > 0 else 0
                ap_values.append(precision_at_thresh)
            
            # mAP@0.5:0.95 is the mean of AP values across all IoU thresholds
            mAP_50_95 = np.mean(ap_values) if ap_values else 0.0
            # mAP@0.5 is just the AP at IoU=0.5 (first threshold)
            mAP_50 = ap_values[0] if ap_values else 0.0
            
            # Compute metrics at IoU=0.5 (already computed in first loop iteration)
            occ_precision = data['tp'] / (data['tp'] + data['fp']) if (data['tp'] + data['fp']) > 0 else 0
            occ_recall = data['tp'] / data['gt'] if data['gt'] > 0 else 0
            occ_f1 = 2 * occ_precision * occ_recall / (occ_precision + occ_recall) if (occ_precision + occ_recall) > 0 else 0
            
            # FPR = FP / (FP + TN), but we don't have TN, so we use FP/(TP+FP) as proxy
            occ_fpr = data['fp'] / (data['tp'] + data['fp']) if (data['tp'] + data['fp']) > 0 else 0
            
            print(f"    mAP@IoU=0.5:0.95: {mAP_50_95:.4f}")
            print(f"    mAP@IoU=0.5: {mAP_50:.4f}")
            print(f"    Precision@IoU=0.5 (conf≥0.3): {occ_precision:.4f}")
            print(f"    Recall@IoU=0.5 (conf≥0.3): {occ_recall:.4f}")
            print(f"    F1-Score@IoU=0.5: {occ_f1:.4f}")
            print(f"    FPR@IoU=0.5: {occ_fpr:.4f}")
            
            # Store in advanced_metrics with string names
            advanced_metrics[f'occ_{occ_name}_num_objects'] = data['gt']
            advanced_metrics[f'occ_{occ_name}_num_images'] = len(unique_images_per_level[occ_level])
            advanced_metrics[f'occ_{occ_name}_mAP_0.5_0.95'] = mAP_50_95
            advanced_metrics[f'occ_{occ_name}_mAP_0.5'] = mAP_50
            advanced_metrics[f'occ_{occ_name}_precision'] = occ_precision
            advanced_metrics[f'occ_{occ_name}_recall'] = occ_recall
            advanced_metrics[f'occ_{occ_name}_f1_score'] = occ_f1
            advanced_metrics[f'occ_{occ_name}_fpr'] = occ_fpr
            
            # Display per-class metrics
            print(f"    \n    Per-Class Metrics:")
            
            # Compute per-class metrics using the class_breakdown data (same logic as overall)
            print(f"    \n    Per-Class Metrics:")
            
            for class_name in sorted(data['class_breakdown'].keys()):
                class_data = data['class_breakdown'][class_name]
                
                # Get TP, FP, FN, GT from class_breakdown (computed with correct occlusion assignment)
                tp = class_data['tp']
                fp = class_data['fp']
                fn = class_data['fn']
                gt_count = class_data['gt']
                
                # Compute metrics at IoU=0.5 (already computed)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / gt_count if gt_count > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Compute AP at multiple IoU thresholds for this class
                # We need to recompute per-class TP/FP at each IoU threshold
                class_ap_values = []
                class_ar_values = []  # For AR@0.5:0.95
                for iou_thresh in iou_thresholds:
                    # Count TP/FP for this class at this IoU threshold
                    class_tp_at_thresh = 0
                    class_fp_at_thresh = 0
                    
                    for pred, target in zip(all_predictions, all_targets):
                        pred_boxes = pred['boxes'].cpu().numpy() if len(pred['boxes']) > 0 else np.array([])
                        pred_scores = pred['scores'].cpu().numpy() if len(pred['scores']) > 0 else np.array([])
                        pred_labels = pred['labels'].cpu().numpy() if len(pred['labels']) > 0 else np.array([])
                        
                        gt_boxes = target['boxes'].cpu().numpy() if len(target['boxes']) > 0 else np.array([])
                        gt_labels = target['labels'].cpu().numpy() if len(target['labels']) > 0 else np.array([])
                        gt_occlusions = target['occlusion'].cpu().numpy() if 'occlusion' in target and len(target['occlusion']) > 0 else np.array([])
                        
                        # Get class label ID (convert class_name back to 0-indexed label)
                        target_class_id = class_name_to_label.get(class_name)
                        
                        if target_class_id is None:
                            continue
                        
                        # Filter by score and class
                        valid_mask = (pred_scores >= 0.3) & (pred_labels == target_class_id)
                        pred_boxes_filtered = pred_boxes[valid_mask]
                        pred_scores_filtered = pred_scores[valid_mask]
                        
                        matched_gts_local = set()
                        sorted_indices = np.argsort(-pred_scores_filtered)
                        
                        for pred_idx in sorted_indices:
                            pred_box = pred_boxes_filtered[pred_idx]
                            
                            best_iou = 0
                            best_gt_idx = -1
                            best_gt_occ = -1
                            
                            for gt_idx, (gt_box, gt_label, gt_occ) in enumerate(zip(gt_boxes, gt_labels, gt_occlusions)):
                                if gt_idx in matched_gts_local:
                                    continue
                                if int(gt_label) != target_class_id:
                                    continue
                                
                                iou = compute_iou(pred_box, gt_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = gt_idx
                                    best_gt_occ = int(gt_occ)
                            
                            if best_iou >= iou_thresh and best_gt_idx >= 0 and best_gt_occ == occ_level:
                                matched_gts_local.add(best_gt_idx)
                                class_tp_at_thresh += 1
                            elif best_gt_idx >= 0 and best_gt_occ == occ_level:
                                class_fp_at_thresh += 1
                    
                    # Compute precision (AP proxy) at this threshold
                    class_precision_at_thresh = class_tp_at_thresh / (class_tp_at_thresh + class_fp_at_thresh) if (class_tp_at_thresh + class_fp_at_thresh) > 0 else 0
                    class_ap_values.append(class_precision_at_thresh)

                    # Compute recall at this threshold for AR calculation
                    class_recall_at_thresh = class_tp_at_thresh / gt_count if gt_count > 0 else 0
                    class_ar_values.append(class_recall_at_thresh)
                
                # Compute mAP for this class
                class_mAP_50_95 = np.mean(class_ap_values) if class_ap_values else 0.0
                class_mAP_50 = class_ap_values[0] if class_ap_values else 0.0
                # Compute AR for this class
                class_AR_50_95 = np.mean(class_ar_values) if class_ar_values else 0.0

                print(f"      {class_name}:")
                print(f"        AP@IoU=0.5:0.95: {class_mAP_50_95:.4f}")
                print(f"        AP@IoU=0.5: {class_mAP_50:.4f}")
                print(f"        AR@IoU=0.5:0.95: {class_AR_50_95:.4f}")
                print(f"        Precision@IoU=0.5: {precision:.4f}")
                print(f"        Recall@IoU=0.5: {recall:.4f}")
                print(f"        F1-Score@IoU=0.5: {f1:.4f}")
                print(f"        GT Count: {gt_count}")
                
                # Store per-class per-occlusion metrics
                advanced_metrics[f'occ_{occ_name}_{class_name}_ap_0.5_0.95'] = class_mAP_50_95
                advanced_metrics[f'occ_{occ_name}_{class_name}_ap_0.5'] = class_mAP_50
                advanced_metrics[f'occ_{occ_name}_{class_name}_ar_0.5_0.95'] = class_AR_50_95
                advanced_metrics[f'occ_{occ_name}_{class_name}_precision'] = precision
                advanced_metrics[f'occ_{occ_name}_{class_name}_recall'] = recall
                advanced_metrics[f'occ_{occ_name}_{class_name}_f1_score'] = f1
                advanced_metrics[f'occ_{occ_name}_{class_name}_gt_count'] = gt_count
    
    # Save a summary log file (without diagnostics) before running detailed diagnostics
    if save_logs and output_dir and log_buffer:
        try:
            # Create summary log file from buffer
            summary_log_path = os.path.join(output_dir, 'test_logs_summary.txt')
            with open(summary_log_path, 'w', encoding='utf-8') as summary_file:
                # Write everything captured in buffer so far
                summary_file.write(''.join(log_buffer))
            
            print(f"\n{'='*80}")
            print(f"Summary log (without diagnostics) saved to: {summary_log_path}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Warning: Failed to save summary log: {e}")
    
    # Run comprehensive diagnostic analysis for ALL classes
    # Only run during testing (--test-only flag sets save_logs=True)
    if save_logs and output_dir and all_predictions and all_targets:
        try:
            print(f"\n{'='*80}")
            print(f"COMPREHENSIVE CLASS-WISE DIAGNOSTIC ANALYSIS")
            print(f"{'='*80}")
            
            # Get all class IDs and names
            class_info = []  # List of (class_id, class_name) tuples
            
            if hasattr(base_ds, 'loadCats'):
                try:
                    cat_ids = sorted(base_ds.getCatIds())
                    cats = base_ds.loadCats(cat_ids)
                    
                    # Detect if dataset is 0-indexed or 1-indexed
                    min_cat_id = min(cat_ids) if cat_ids else 1
                    is_zero_indexed = (min_cat_id == 0)
                    
                    print(f"\n[Class-wise Diagnostics] Detected {'0-indexed' if is_zero_indexed else '1-indexed'} dataset")
                    
                    for cat in cats:
                        coco_cat_id = cat['id']
                        class_name = cat['name']
                        
                        if is_zero_indexed:
                            # 0-indexed dataset: category_id IS the label
                            class_id = coco_cat_id
                        else:
                            # 1-indexed dataset: convert category_id to 0-indexed label
                            class_id = coco_cat_id - 1
                        
                        class_info.append((class_id, class_name))
                except:
                    # Fallback if COCO API fails
                    for i in range(num_categories):
                        class_info.append((i, f"class_{i}"))
            else:
                # Fallback if no COCO API
                for i in range(num_categories):
                    class_info.append((i, f"class_{i}"))
            
            print(f"\nRunning diagnostics for {len(class_info)} classes...")
            print(f"This includes: {', '.join([name for _, name in class_info])}\n")

            # Build a name -> id mapping for confusion class lookups
            class_name_to_id = {name.lower(): cid for cid, name in class_info}

            # Run diagnostics for each class
            for class_id, class_name in class_info:
                print(f"\n{'='*80}")
                print(f"Analyzing Class: {class_name} (tensor label ID={class_id})")
                print(f"{'='*80}")

                # Set up confusion class tracking for specific classes
                confusion_classes = []

                # Track motorcycle -> car/truck/bus/van/tricycle misclassifications (all vehicle types)
                if class_name.lower() == 'motorcycle':
                    for conf_name in ['car', 'truck', 'bus', 'van', 'tricycle']:
                        if conf_name in class_name_to_id:
                            confusion_classes.append((class_name_to_id[conf_name], conf_name))
                    if confusion_classes:
                        conf_names = ', '.join([name for _, name in confusion_classes])
                        print(f"  Will track misclassifications: {class_name} -> {{{conf_names}}}")

                # Track van -> car/truck/bus/motorcycle/tricycle misclassifications (all vehicle types)
                if class_name.lower() == 'van':
                    for conf_name in ['car', 'truck', 'bus', 'motorcycle', 'tricycle']:
                        if conf_name in class_name_to_id:
                            confusion_classes.append((class_name_to_id[conf_name], conf_name))
                    if confusion_classes:
                        conf_names = ', '.join([name for _, name in confusion_classes])
                        print(f"  Will track misclassifications: {class_name} -> {{{conf_names}}}")

                try:
                    visualize_and_diagnose_class(
                        predictions=all_predictions,
                        targets=all_targets,
                        class_id=class_id,
                        class_name=class_name,
                        coco_api=base_ds,
                        output_dir=output_dir,
                        iou_threshold=0.5,
                        score_threshold=0.3,
                        confusion_classes=confusion_classes
                    )
                except Exception as e:
                    print(f"\nWarning: Failed to run diagnostics for {class_name}: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"\n{'='*80}")
            print(f"COMPLETED DIAGNOSTIC ANALYSIS FOR ALL CLASSES")
            print(f"{'='*80}\n")
            
            # Now run comprehensive per-occlusion diagnostics (aggregated across all classes)
            print(f"\n{'='*80}")
            print(f"STARTING PER-OCCLUSION LEVEL DIAGNOSTICS")
            print(f"{'='*80}\n")
            
            try:
                comprehensive_occlusion_diagnostics(
                    predictions=all_predictions,
                    targets=all_targets,
                    coco_api=base_ds,
                    output_dir=output_dir,
                    iou_threshold=0.5,
                    score_threshold=0.3
                )
            except Exception as e:
                print(f"\nWarning: Failed to run per-occlusion diagnostics: {e}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"\nWarning: Failed to run comprehensive diagnostics: {e}")
            import traceback
            traceback.print_exc()
    
    # Model complexity metrics are computed once at initialization in det_solver.py
    # and added to log_stats there, so we don't recalculate them here
    
    # Add FPS to metrics
    advanced_metrics['inference_time_ms'] = avg_inference_time * 1000
    advanced_metrics['fps'] = fps

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    
    # Add advanced metrics to stats
    stats.update(advanced_metrics)

    # Generate visualizations after evaluation completes
    if save_visualizations and epoch is not None and len(vis_samples) > 0 and output_dir is not None:
        try:
            from .visualization_utils import visualize_predictions
            
            num_collected = len(vis_samples)
            print(f"\n[Visualization] Collected {num_collected} images (1 from every 10th batch)")
            
            # Concatenate all stored samples
            all_images = torch.cat(vis_samples, dim=0)
            
            # Extract metrics for visualization
            metrics_dict = {}
            if coco_evaluator and 'bbox' in coco_evaluator.coco_eval:
                coco_stats = coco_evaluator.coco_eval['bbox'].stats
                metrics_dict['mAP@0.5:0.95'] = coco_stats[0]
                metrics_dict['mAP@0.5'] = coco_stats[1]
                
                # Calculate precision and recall correctly
                coco_eval = coco_evaluator.coco_eval['bbox']
                precision = coco_eval.eval['precision']
                recall = coco_eval.eval['recall']
                
                # Precision @ IoU=0.5 (average over all categories)
                precision_50 = precision[0, :, :, 0, 2]  # IoU=0.5, all recall, all cats, all areas, maxDet=100
                valid_precision_50 = precision_50[precision_50 > -1]
                avg_precision_50 = valid_precision_50.mean() if len(valid_precision_50) > 0 else 0.0
                
                # Recall @ IoU=0.5 (average over all categories)
                recall_50 = recall[0, :, 0, 2]  # IoU=0.5, all cats, all areas, maxDet=100
                valid_recall_50 = recall_50[recall_50 > -1]
                avg_recall_50 = valid_recall_50.mean() if len(valid_recall_50) > 0 else 0.0
                
                metrics_dict['precision'] = avg_precision_50
                metrics_dict['recall'] = avg_recall_50
                
                # Calculate F1 score
                if avg_precision_50 > 0 and avg_recall_50 > 0:
                    f1_score = 2 * (avg_precision_50 * avg_recall_50) / (avg_precision_50 + avg_recall_50)
                else:
                    f1_score = 0.0
                metrics_dict['f1_score'] = f1_score
            
            # Get class names if available
            class_names = None
            cat_ids_list = []
            if hasattr(base_ds, 'dataset') and hasattr(base_ds.dataset, 'classes'):
                class_names = base_ds.dataset.classes
            elif hasattr(base_ds, 'loadCats'):
                # COCO API object
                try:
                    cat_ids_list = base_ds.getCatIds()
                    cats = base_ds.loadCats(cat_ids_list)
                    class_names = [cat['name'] for cat in cats]
                except:
                    pass
            
            # Visualize predictions
            visualize_predictions(
                images=all_images,
                predictions=vis_predictions,
                targets=vis_targets,
                coco_api=base_ds,
                output_dir=pathlib.Path(output_dir),
                epoch=epoch,
                metrics=metrics_dict,
                max_images=num_collected,  # Save all collected images (1 per batch)
                class_names=class_names
            )
        except Exception as e:
            print(f"Warning: Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()

    # Close log file if it was opened
    if log_file is not None:
        print("="*80)
        print(f"Test logs saved to: {os.path.join(output_dir, 'test_logs.txt')}")
        sys.stdout = original_stdout  # Restore original stdout
        log_file.close()

    return stats, coco_evaluator



