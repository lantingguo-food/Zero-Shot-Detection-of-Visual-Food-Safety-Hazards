"""
Faster R-CNN Detector with custom modifications for zero-shot detection
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class ZeroShotDetector(nn.Module):
    """
    Zero-Shot Detector based on Faster R-CNN
    Supports both seen and unseen class detection
    """
    
    def __init__(self, config: dict, mode: str = 'seen'):
        """
        Args:
            config: Configuration dictionary
            mode: 'seen', 'unseen', or 'gzsd' (generalized zero-shot)
        """
        super().__init__()
        
        self.config = config
        self.mode = mode
        
        # Determine number of classes based on mode
        if mode == 'seen':
            self.num_classes = config['dataset']['num_seen_classes'] + 1  # +1 for background
        elif mode == 'unseen':
            self.num_classes = config['dataset']['num_unseen_classes'] + 1
        else:  # gzsd
            self.num_classes = (config['dataset']['num_seen_classes'] + 
                               config['dataset']['num_unseen_classes'] + 1)
        
        # Build detector
        self.detector = self._build_detector()
        
        # Feature extractor for KEFS
        self.feature_extractor = None
        
    def _build_detector(self):
        """Build Faster R-CNN detector"""
        
        # Load backbone
        if self.config['detector']['backbone'] == 'resnet101':
            backbone = torchvision.models.resnet101(
                pretrained=self.config['detector']['pretrained']
            )
            # Use layers up to layer4
            backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone.out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {self.config['detector']['backbone']}")
        
        # RPN anchor generator
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # ROI pooler
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=self.config['detector']['roi_pool_size'],
            sampling_ratio=2
        )
        
        # Build Faster R-CNN
        model = FasterRCNN(
            backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            # RPN parameters
            rpn_batch_size_per_image=self.config['detector']['rpn_batch_size'],
            rpn_positive_fraction=self.config['detector']['rpn_positive_fraction'],
            rpn_pre_nms_top_n_train=2000,
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=2000,
            rpn_post_nms_top_n_test=1000,
            rpn_nms_thresh=self.config['detector']['nms_threshold'],
            rpn_fg_iou_thresh=self.config['detector']['rpn_positive_iou'],
            rpn_bg_iou_thresh=self.config['detector']['rpn_negative_iou'],
            # Box parameters
            box_batch_size_per_image=self.config['detector']['roi_batch_size'],
            box_positive_fraction=self.config['detector']['roi_positive_fraction'],
            box_score_thresh=self.config['detector']['detection_threshold'],
            box_nms_thresh=self.config['detector']['nms_threshold'],
            box_fg_iou_thresh=self.config['detector']['roi_positive_iou'],
            box_bg_iou_thresh=self.config['detector']['roi_negative_iou'][0],
        )
        
        return model
    
    def forward(self, images, targets=None):
        """
        Forward pass
        
        Args:
            images: List of images or tensor [B, 3, H, W]
            targets: List of target dicts (for training)
            
        Returns:
            If training: loss dict
            If inference: list of detection dicts
        """
        return self.detector(images, targets)
    
    def extract_features(self, images):
        """
        Extract region features for KEFS training
        
        Args:
            images: Tensor [B, 3, H, W]
            
        Returns:
            features: Tensor [N, feature_dim] where N is total number of regions
        """
        self.detector.eval()
        
        with torch.no_grad():
            # Get backbone features
            features = self.detector.backbone(images)
            
            # Get region proposals
            images_list = [img for img in images]
            proposals, _ = self.detector.rpn(images, features)
            
            # ROI pooling
            box_features = self.detector.roi_heads.box_roi_pool(
                features, proposals, [img.shape[-2:] for img in images_list]
            )
            
            # Flatten features
            box_features = self.detector.roi_heads.box_head(box_features)
            
        return box_features
    
    def update_classifier(self, unseen_classifier):
        """
        Update detector with unseen class classifier
        For GZSD mode
        """
        if self.mode != 'gzsd':
            raise ValueError("Can only update classifier in GZSD mode")
        
        # Get current classifier
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        
        # Create new classifier for all classes
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes
        )
        
        # Initialize unseen class weights
        num_seen = self.config['dataset']['num_seen_classes']
        self.detector.roi_heads.box_predictor.cls_score.weight.data[num_seen+1:] = \
            unseen_classifier.weight.data
        self.detector.roi_heads.box_predictor.cls_score.bias.data[num_seen+1:] = \
            unseen_classifier.bias.data
    
    def apply_calibration(self, predictions, calibration_factor=0.7):
        """
        Apply calibration to seen class scores for GZSD (Equation 21)
        
        Args:
            predictions: List of prediction dicts
            calibration_factor: γ in the paper
            
        Returns:
            Calibrated predictions
        """
        if self.mode != 'gzsd':
            return predictions
        
        num_seen = self.config['dataset']['num_seen_classes']
        
        for pred in predictions:
            # Apply calibration to seen classes
            seen_mask = pred['labels'] <= num_seen
            pred['scores'][seen_mask] = pred['scores'][seen_mask] * calibration_factor
        
        return predictions
    
    def set_mode(self, mode: str):
        """Change detector mode"""
        self.mode = mode
        # May need to adjust num_classes and reload classifier
    
    def freeze_backbone(self):
        """Freeze backbone weights"""
        for param in self.detector.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights"""
        for param in self.detector.backbone.parameters():
            param.requires_grad = True
    
    def save(self, path: str):
        """Save detector state"""
        torch.save({
            'model_state_dict': self.detector.state_dict(),
            'config': self.config,
            'mode': self.mode,
            'num_classes': self.num_classes
        }, path)
    
    def load(self, path: str, device='cpu'):
        """Load detector state"""
        checkpoint = torch.load(path, map_location=device)
        self.detector.load_state_dict(checkpoint['model_state_dict'])
        self.mode = checkpoint.get('mode', self.mode)
        self.num_classes = checkpoint.get('num_classes', self.num_classes)


def build_detector(config: dict, mode: str = 'seen'):
    """
    Factory function to build detector
    
    Args:
        config: Configuration dict
        mode: 'seen', 'unseen', or 'gzsd'
        
    Returns:
        ZeroShotDetector instance
    """
    return ZeroShotDetector(config, mode)
