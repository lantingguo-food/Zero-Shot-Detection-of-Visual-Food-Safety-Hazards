"""
FSVH Dataset Loader for Food Safety Visual Hazards
"""

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as T
from typing import Dict, List, Tuple
import numpy as np


class FSVHDataset(Dataset):
    """Food Safety Visual Hazards Dataset"""
    
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 transform=None,
                 config: dict = None):
        """
        Args:
            root_dir: Root directory with images/ and annotations/
            split: 'train' or 'test'
            transform: Image transformations
            config: Configuration dictionary
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or self._default_transform()
        self.config = config
        
        # Load annotations
        ann_file = os.path.join(root_dir, 'annotations', f'{split}.json')
        with open(ann_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.images = self.annotations['images']
        self.annotations_list = self.annotations['annotations']
        
        # Create image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations_list:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Class information
        self.categories = self.annotations['categories']
        self.num_classes = len(self.categories)
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary with:
                - image: Tensor [3, H, W]
                - boxes: Tensor [N, 4] in xyxy format
                - labels: Tensor [N]
                - image_id: int
                - original_size: tuple (H, W)
        """
        # Get image info
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.root_dir, 'images', self.split, img_info['file_name'])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        
        # Extract boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'original_size': original_size
        }
    
    def _default_transform(self):
        """Default image transformations"""
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Custom collate function for batching"""
        images = torch.stack([item['image'] for item in batch])
        
        return {
            'images': images,
            'boxes': [item['boxes'] for item in batch],
            'labels': [item['labels'] for item in batch],
            'image_ids': [item['image_id'] for item in batch],
            'original_sizes': [item['original_size'] for item in batch]
        }


class SynthesizedDataset(Dataset):
    """Dataset of synthesized features for unseen classes"""
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            features: [N, feature_dim]
            labels: [N]
        """
        self.features = features
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> dict:
        return {
            'features': self.features[idx],
            'labels': self.labels[idx]
        }


class SemanticVectorLoader:
    """Loads and processes semantic vectors for classes"""
    
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = None
        self.model = None
        
    def load_bert_model(self):
        """Load BERT model for embedding generation"""
        from transformers import BertTokenizer, BertModel
        
        model_name = self.config['semantic']['bert_model']
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        
    def get_embeddings(self, class_descriptions: Dict[int, str]) -> torch.Tensor:
        """
        Generate BERT embeddings for class descriptions
        
        Args:
            class_descriptions: {class_id: description text}
            
        Returns:
            embeddings: [num_classes, embedding_dim]
        """
        if self.model is None:
            self.load_bert_model()
        
        num_classes = len(class_descriptions)
        embeddings = []
        
        with torch.no_grad():
            for class_id in sorted(class_descriptions.keys()):
                text = class_descriptions[class_id]
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=self.config['semantic']['max_length'],
                    padding='max_length',
                    truncation=True
                )
                
                # Get BERT outputs
                outputs = self.model(**inputs)
                
                # Average last 4 hidden layers
                hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
                if hidden_states is None:
                    # Use [CLS] token from last layer
                    embedding = outputs.last_hidden_state[:, 0, :]
                else:
                    # Average last 4 layers [CLS] token
                    embedding = torch.mean(
                        torch.stack(hidden_states[-4:]), dim=0
                    )[:, 0, :]
                
                embeddings.append(embedding.squeeze(0))
        
        return torch.stack(embeddings)
    
    @staticmethod
    def load_from_file(file_path: str) -> torch.Tensor:
        """Load pre-computed semantic vectors from file"""
        data = torch.load(file_path)
        return data['embeddings']


def get_data_loaders(config: dict, 
                    root_dir: str) -> Tuple[torch.utils.data.DataLoader, 
                                           torch.utils.data.DataLoader]:
    """
    Create train and test data loaders
    
    Returns:
        train_loader, test_loader
    """
    # Training transforms with augmentation
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=config['augmentation']['horizontal_flip']),
        T.ColorJitter(brightness=config['augmentation']['color_jitter'],
                     contrast=config['augmentation']['color_jitter']),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    
    # Test transforms (no augmentation)
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FSVHDataset(root_dir, 'train', train_transform, config)
    test_dataset = FSVHDataset(root_dir, 'test', test_transform, config)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['detector_batch_size'],
        shuffle=True,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory'],
        collate_fn=FSVHDataset.collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['training']['detector_batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=config['system']['pin_memory'],
        collate_fn=FSVHDataset.collate_fn
    )
    
    return train_loader, test_loader
