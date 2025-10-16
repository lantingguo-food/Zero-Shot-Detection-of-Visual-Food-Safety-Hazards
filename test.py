"""
Testing script for Zero-Shot Food Safety Hazard Detection
Evaluates model on ZSD and GZSD settings
"""

import torch
import torch.nn as nn
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

from data.dataset import get_data_loaders
from models.kefs import KEFS
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou


def evaluate_detection(predictions, ground_truths, iou_threshold=0.5):
    """
    Evaluate detection performance
    
    Args:
        predictions: List of dicts with 'boxes', 'labels', 'scores'
        ground_truths: List of dicts with 'boxes', 'labels'
        
    Returns:
        Dictionary with mAP and other metrics
    """
    # Organize by class
    all_classes = set()
    for gt in ground_truths:
        all_classes.update(gt['labels'].tolist())
    
    class_aps = {}
    
    for cls in all_classes:
        # Collect predictions and ground truths for this class
        cls_preds = []
        cls_gts = []
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Get predictions for this class
            cls_mask = pred['labels'] == cls
            if cls_mask.sum() > 0:
                cls_boxes = pred['boxes'][cls_mask]
                cls_scores = pred['scores'][cls_mask]
                for box, score in zip(cls_boxes, cls_scores):
                    cls_preds.append({
                        'image_id': i,
                        'box': box,
                        'score': score
                    })
            
            # Get ground truths for this class
            gt_mask = gt['labels'] == cls
            if gt_mask.sum() > 0:
                gt_boxes = gt['boxes'][gt_mask]
                for box in gt_boxes:
                    cls_gts.append({
                        'image_id': i,
                        'box': box,
                        'matched': False
                    })
        
        if len(cls_gts) == 0:
            continue
        
        # Sort predictions by score
        cls_preds.sort(key=lambda x: x['score'], reverse=True)
        
        # Compute precision-recall
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        
        for i, pred in enumerate(cls_preds):
            # Find matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(cls_gts):
                if gt['image_id'] == pred['image_id'] and not gt['matched']:
                    iou = compute_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            if best_iou >= iou_threshold:
                tp[i] = 1
                cls_gts[best_gt_idx]['matched'] = True
            else:
                fp[i] = 1
        
        # Compute cumulative precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(cls_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP (area under PR curve)
        ap = 0
        for i in range(len(precisions) - 1):
            ap += (recalls[i+1] - recalls[i]) * precisions[i+1]
        
        class_aps[cls] = ap
    
    # Compute mAP
    if len(class_aps) > 0:
        mAP = np.mean(list(class_aps.values()))
    else:
        mAP = 0.0
    
    return {
        'mAP': mAP,
        'class_aps': class_aps
    }


def test_zsd(config: dict, args: argparse.Namespace):
    """
    Test in Zero-Shot Detection mode (only unseen classes)
    """
    print("=" * 80)
    print("Testing: Zero-Shot Detection (ZSD)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = config['dataset']['num_unseen_classes'] + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    _, test_loader = get_data_loaders(config, args.data_dir)
    
    # Run inference
    print("Running inference...")
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['images'].to(device)
            
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                # Filter by score threshold
                keep = output['scores'] > config['evaluation']['detection_threshold']
                
                predictions.append({
                    'boxes': output['boxes'][keep].cpu(),
                    'labels': output['labels'][keep].cpu(),
                    'scores': output['scores'][keep].cpu()
                })
                
                ground_truths.append({
                    'boxes': batch['boxes'][i],
                    'labels': batch['labels'][i]
                })
    
    # Evaluate
    print("Evaluating...")
    results = evaluate_detection(
        predictions, 
        ground_truths,
        iou_threshold=config['evaluation']['iou_threshold']
    )
    
    print(f"\nResults (ZSD):")
    print(f"mAP@{config['evaluation']['iou_threshold']}: {results['mAP']*100:.1f}%")
    
    # Save results
    with open(os.path.join(os.path.dirname(args.checkpoint), 'zsd_results.json'), 'w') as f:
        json.dump({
            'mAP': float(results['mAP']),
            'class_aps': {int(k): float(v) for k, v in results['class_aps'].items()}
        }, f, indent=2)
    
    return results


def test_gzsd(config: dict, args: argparse.Namespace):
    """
    Test in Generalized Zero-Shot Detection mode (seen + unseen classes)
    """
    print("=" * 80)
    print("Testing: Generalized Zero-Shot Detection (GZSD)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    total_classes = (config['dataset']['num_seen_classes'] + 
                    config['dataset']['num_unseen_classes'] + 1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, total_classes
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test data
    _, test_loader = get_data_loaders(config, args.data_dir)
    
    # Run inference with calibration
    print("Running inference...")
    predictions = []
    ground_truths = []
    
    num_seen = config['dataset']['num_seen_classes']
    calibration = args.calibration_factor
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['images'].to(device)
            
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                scores = output['scores'].clone()
                labels = output['labels']
                
                # Apply calibration to seen classes (Equation 21)
                seen_mask = labels <= num_seen
                scores[seen_mask] = scores[seen_mask] * calibration
                
                # Filter by score threshold
                keep = scores > config['evaluation']['detection_threshold']
                
                predictions.append({
                    'boxes': output['boxes'][keep].cpu(),
                    'labels': labels[keep].cpu(),
                    'scores': scores[keep].cpu()
                })
                
                ground_truths.append({
                    'boxes': batch['boxes'][i],
                    'labels': batch['labels'][i]
                })
    
    # Evaluate overall
    print("Evaluating overall performance...")
    results_all = evaluate_detection(
        predictions, 
        ground_truths,
        iou_threshold=config['evaluation']['iou_threshold']
    )
    
    # Separate seen and unseen
    print("Evaluating seen classes...")
    seen_preds = []
    seen_gts = []
    for pred, gt in zip(predictions, ground_truths):
        seen_mask_pred = pred['labels'] <= num_seen
        seen_mask_gt = gt['labels'] <= num_seen
        
        seen_preds.append({
            'boxes': pred['boxes'][seen_mask_pred],
            'labels': pred['labels'][seen_mask_pred],
            'scores': pred['scores'][seen_mask_pred]
        })
        
        seen_gts.append({
            'boxes': gt['boxes'][seen_mask_gt],
            'labels': gt['labels'][seen_mask_gt]
        })
    
    results_seen = evaluate_detection(seen_preds, seen_gts,
                                     iou_threshold=config['evaluation']['iou_threshold'])
    
    print("Evaluating unseen classes...")
    unseen_preds = []
    unseen_gts = []
    for pred, gt in zip(predictions, ground_truths):
        unseen_mask_pred = pred['labels'] > num_seen
        unseen_mask_gt = gt['labels'] > num_seen
        
        unseen_preds.append({
            'boxes': pred['boxes'][unseen_mask_pred],
            'labels': pred['labels'][unseen_mask_pred],
            'scores': pred['scores'][unseen_mask_pred]
        })
        
        unseen_gts.append({
            'boxes': gt['boxes'][unseen_mask_gt],
            'labels': gt['labels'][unseen_mask_gt]
        })
    
    results_unseen = evaluate_detection(unseen_preds, unseen_gts,
                                       iou_threshold=config['evaluation']['iou_threshold'])
    
    # Compute harmonic mean
    S = results_seen['mAP']
    U = results_unseen['mAP']
    HM = 2 * S * U / (S + U + 1e-8)
    
    print(f"\nResults (GZSD):")
    print(f"Seen mAP: {S*100:.1f}%")
    print(f"Unseen mAP: {U*100:.1f}%")
    print(f"Harmonic Mean: {HM*100:.1f}%")
    
    # Save results
    with open(os.path.join(os.path.dirname(args.checkpoint), 'gzsd_results.json'), 'w') as f:
        json.dump({
            'seen_mAP': float(S),
            'unseen_mAP': float(U),
            'harmonic_mean': float(HM),
            'overall_mAP': float(results_all['mAP'])
        }, f, indent=2)
    
    return {
        'seen': S,
        'unseen': U,
        'harmonic_mean': HM,
        'overall': results_all['mAP']
    }


def main():
    parser = argparse.ArgumentParser(description='Test Zero-Shot Food Safety Detection')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['zsd', 'gzsd'],
                       help='Testing mode')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to FSVH dataset')
    parser.add_argument('--calibration_factor', type=float, default=0.7,
                       help='Calibration factor for GZSD (default: 0.7)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run testing
    if args.mode == 'zsd':
        test_zsd(config, args)
    elif args.mode == 'gzsd':
        test_gzsd(config, args)
    
    print("\nTesting completed!")


if __name__ == '__main__':
    main()
