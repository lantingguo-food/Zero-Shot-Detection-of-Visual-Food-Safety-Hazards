"""
Evaluation metrics for Zero-Shot Detection
Implements mAP, Recall@100, and Harmonic Mean calculations
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


class DetectionMetrics:
    """Computes detection metrics for ZSD and GZSD evaluation"""
    
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
        
    def reset(self):
        """Reset all accumulated metrics"""
        self.predictions = []
        self.ground_truths = []
        
    def update(self, pred_boxes: torch.Tensor, pred_labels: torch.Tensor, 
               pred_scores: torch.Tensor, gt_boxes: torch.Tensor, 
               gt_labels: torch.Tensor):
        """
        Update metrics with new predictions and ground truths
        
        Args:
            pred_boxes: [N, 4] predicted boxes
            pred_labels: [N] predicted class labels
            pred_scores: [N] prediction confidence scores
            gt_boxes: [M, 4] ground truth boxes
            gt_labels: [M] ground truth class labels
        """
        self.predictions.append({
            'boxes': pred_boxes.cpu(),
            'labels': pred_labels.cpu(),
            'scores': pred_scores.cpu()
        })
        
        self.ground_truths.append({
            'boxes': gt_boxes.cpu(),
            'labels': gt_labels.cpu()
        })
    
    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Compute IoU between two boxes in [x1, y1, x2, y2] format"""
        x1_inter = max(box1[0].item(), box2[0].item())
        y1_inter = max(box1[1].item(), box2[1].item())
        x2_inter = min(box1[2].item(), box2[2].item())
        y2_inter = min(box1[3].item(), box2[3].item())
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        
        return iou.item()
    
    def compute_ap(self, class_id: int) -> float:
        """
        Compute Average Precision for a single class
        
        Args:
            class_id: Class ID to compute AP for
            
        Returns:
            Average Precision value
        """
        # Collect all predictions and ground truths for this class
        class_preds = []
        class_gts = []
        
        for i, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
            # Get predictions for this class
            cls_mask = pred['labels'] == class_id
            if cls_mask.sum() > 0:
                for box, score in zip(pred['boxes'][cls_mask], pred['scores'][cls_mask]):
                    class_preds.append({
                        'image_id': i,
                        'box': box,
                        'score': score.item()
                    })
            
            # Get ground truths for this class
            gt_mask = gt['labels'] == class_id
            if gt_mask.sum() > 0:
                for box in gt['boxes'][gt_mask]:
                    class_gts.append({
                        'image_id': i,
                        'box': box,
                        'matched': False
                    })
        
        if len(class_gts) == 0:
            return 0.0
        
        if len(class_preds) == 0:
            return 0.0
        
        # Sort predictions by confidence
        class_preds.sort(key=lambda x: x['score'], reverse=True)
        
        # Compute precision-recall curve
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        for i, pred in enumerate(class_preds):
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(class_gts):
                if gt['image_id'] == pred['image_id'] and not gt['matched']:
                    iou = self.compute_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
            
            # Check if match is valid
            if best_iou >= self.iou_threshold:
                tp[i] = 1
                class_gts[best_gt_idx]['matched'] = True
            else:
                fp[i] = 1
        
        # Compute cumulative precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(class_gts)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        return ap
    
    def compute_map(self, class_ids: List[int] = None) -> Dict:
        """
        Compute mean Average Precision
        
        Args:
            class_ids: List of class IDs to compute mAP for. If None, use all classes.
            
        Returns:
            Dictionary with mAP and per-class AP
        """
        if class_ids is None:
            # Get all classes present in ground truth
            all_classes = set()
            for gt in self.ground_truths:
                all_classes.update(gt['labels'].tolist())
            class_ids = sorted(list(all_classes))
        
        # Compute AP for each class
        class_aps = {}
        for class_id in class_ids:
            ap = self.compute_ap(class_id)
            class_aps[class_id] = ap
        
        # Compute mAP
        if len(class_aps) > 0:
            mAP = np.mean(list(class_aps.values()))
        else:
            mAP = 0.0
        
        return {
            'mAP': mAP,
            'class_aps': class_aps
        }
    
    def compute_recall_at_k(self, k: int = 100, class_ids: List[int] = None) -> Dict:
        """
        Compute Recall@k metric
        
        Args:
            k: Number of top predictions to consider
            class_ids: List of class IDs. If None, use all classes.
            
        Returns:
            Dictionary with Recall@k and per-class recall
        """
        if class_ids is None:
            all_classes = set()
            for gt in self.ground_truths:
                all_classes.update(gt['labels'].tolist())
            class_ids = sorted(list(all_classes))
        
        class_recalls = {}
        
        for class_id in class_ids:
            # Collect predictions and ground truths
            class_preds = []
            total_gts = 0
            
            for i, (pred, gt) in enumerate(zip(self.predictions, self.ground_truths)):
                cls_mask_pred = pred['labels'] == class_id
                cls_mask_gt = gt['labels'] == class_id
                
                # Top-k predictions per image
                scores = pred['scores'][cls_mask_pred]
                if len(scores) > k:
                    topk_indices = torch.topk(scores, k)[1]
                    boxes = pred['boxes'][cls_mask_pred][topk_indices]
                else:
                    boxes = pred['boxes'][cls_mask_pred]
                
                for box in boxes:
                    class_preds.append({
                        'image_id': i,
                        'box': box
                    })
                
                total_gts += cls_mask_gt.sum().item()
            
            if total_gts == 0:
                class_recalls[class_id] = 0.0
                continue
            
            # Count true positives
            tp = 0
            for i, gt in enumerate(self.ground_truths):
                cls_mask = gt['labels'] == class_id
                for gt_box in gt['boxes'][cls_mask]:
                    # Check if any prediction matches
                    for pred in class_preds:
                        if pred['image_id'] == i:
                            iou = self.compute_iou(pred['box'], gt_box)
                            if iou >= self.iou_threshold:
                                tp += 1
                                break
            
            recall = tp / total_gts if total_gts > 0 else 0.0
            class_recalls[class_id] = recall
        
        avg_recall = np.mean(list(class_recalls.values())) if class_recalls else 0.0
        
        return {
            f'Recall@{k}': avg_recall,
            'class_recalls': class_recalls
        }
    
    def compute_gzsd_metrics(self, seen_classes: List[int], 
                            unseen_classes: List[int]) -> Dict:
        """
        Compute metrics for Generalized Zero-Shot Detection
        
        Args:
            seen_classes: List of seen class IDs
            unseen_classes: List of unseen class IDs
            
        Returns:
            Dictionary with seen, unseen, and harmonic mean performance
        """
        # Compute mAP for seen classes
        seen_results = self.compute_map(seen_classes)
        
        # Compute mAP for unseen classes
        unseen_results = self.compute_map(unseen_classes)
        
        # Compute harmonic mean
        S = seen_results['mAP']
        U = unseen_results['mAP']
        HM = 2 * S * U / (S + U + 1e-8)
        
        # Compute Recall@100
        seen_recall = self.compute_recall_at_k(100, seen_classes)
        unseen_recall = self.compute_recall_at_k(100, unseen_classes)
        
        recall_S = seen_recall['Recall@100']
        recall_U = unseen_recall['Recall@100']
        recall_HM = 2 * recall_S * recall_U / (recall_S + recall_U + 1e-8)
        
        return {
            'mAP': {
                'seen': S,
                'unseen': U,
                'harmonic_mean': HM
            },
            'Recall@100': {
                'seen': recall_S,
                'unseen': recall_U,
                'harmonic_mean': recall_HM
            },
            'class_aps': {
                'seen': seen_results['class_aps'],
                'unseen': unseen_results['class_aps']
            }
        }


def compute_harmonic_mean(seen_metric: float, unseen_metric: float) -> float:
    """
    Compute harmonic mean between seen and unseen metrics
    HM = 2 * S * U / (S + U)
    """
    if seen_metric + unseen_metric == 0:
        return 0.0
    return 2 * seen_metric * unseen_metric / (seen_metric + unseen_metric)


def format_metrics_table(metrics: Dict) -> str:
    """Format metrics dictionary as a readable table"""
    lines = []
    lines.append("=" * 80)
    lines.append("Detection Metrics")
    lines.append("=" * 80)
    
    if 'mAP' in metrics:
        if isinstance(metrics['mAP'], dict):
            # GZSD format
            lines.append(f"\nmAP@0.5:")
            lines.append(f"  Seen:          {metrics['mAP']['seen']*100:6.2f}%")
            lines.append(f"  Unseen:        {metrics['mAP']['unseen']*100:6.2f}%")
            lines.append(f"  Harmonic Mean: {metrics['mAP']['harmonic_mean']*100:6.2f}%")
            
            if 'Recall@100' in metrics:
                lines.append(f"\nRecall@100:")
                lines.append(f"  Seen:          {metrics['Recall@100']['seen']*100:6.2f}%")
                lines.append(f"  Unseen:        {metrics['Recall@100']['unseen']*100:6.2f}%")
                lines.append(f"  Harmonic Mean: {metrics['Recall@100']['harmonic_mean']*100:6.2f}%")
        else:
            # ZSD format
            lines.append(f"\nmAP@0.5: {metrics['mAP']*100:.2f}%")
    
    lines.append("=" * 80)
    return "\n".join(lines)
