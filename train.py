"""
Training script for Zero-Shot Food Safety Hazard Detection
Implements Algorithm 1 from the paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import os
from tqdm import tqdm
import numpy as np

from data.dataset import get_data_loaders, SemanticVectorLoader, SynthesizedDataset
from data.knowledge_graph import FoodSafetyKnowledgeGraph, MultiSourceGraphs
from models.kefs import KEFS
from utils.losses import KEFSLoss, DetectionLoss
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_detector(config: dict, args: argparse.Namespace):
    """
    Stage 1: Train Faster R-CNN detector on seen classes
    Corresponds to Line 1 in Algorithm 1
    """
    print("=" * 80)
    print("Stage 1: Training Detector on Seen Classes")
    print("=" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['system']['seed'])
    
    # Data loaders
    train_loader, val_loader = get_data_loaders(config, args.data_dir)
    
    # Create Faster R-CNN model
    num_classes = config['dataset']['num_seen_classes'] + 1  # +1 for background
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['training']['detector_lr'],
                          weight_decay=config['training']['detector_weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training']['detector_lr_decay_epochs'],
        gamma=config['training']['detector_lr_decay_factor']
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(config['training']['detector_epochs']):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["detector_epochs"]}')
        
        for batch in pbar:
            images = batch['images'].to(device)
            
            # Prepare targets
            targets = []
            for i in range(len(images)):
                target = {
                    'boxes': batch['boxes'][i].to(device),
                    'labels': batch['labels'][i].to(device)
                }
                targets.append(target)
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            pbar.set_postfix({'loss': losses.item()})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.output_dir, 'detector_best.pth'))
    
    print(f'\nDetector training completed. Best loss: {best_loss:.4f}')
    return model


def extract_features(model, data_loader, device):
    """
    Extract region features from trained detector
    Corresponds to Line 2 in Algorithm 1
    """
    print("\nExtracting region features...")
    
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch['images'].to(device)
            
            # Get backbone features
            features = model.backbone(images)
            
            # Pool features (simplified)
            pooled = nn.AdaptiveAvgPool2d(1)(features['3'])
            pooled = pooled.view(pooled.size(0), -1)
            
            all_features.append(pooled.cpu())
            
            # Get corresponding labels
            for labels in batch['labels']:
                all_labels.extend(labels.tolist())
    
    features = torch.cat(all_features, dim=0)
    labels = torch.tensor(all_labels)
    
    return features, labels


def train_kefs(config: dict, args: argparse.Namespace):
    """
    Stage 2: Train Knowledge-Enhanced Feature Synthesizer
    Corresponds to Lines 3-4 in Algorithm 1
    """
    print("\n" + "=" * 80)
    print("Stage 2: Training KEFS")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load detector
    print("Loading trained detector...")
    detector = fasterrcnn_resnet50_fpn(pretrained=False)
    checkpoint = torch.load(args.detector_checkpoint, map_location=device)
    detector.load_state_dict(checkpoint['model_state_dict'])
    detector = detector.to(device)
    
    # Extract features from seen classes
    train_loader, _ = get_data_loaders(config, args.data_dir)
    features, labels = extract_features(detector, train_loader, device)
    features = features.to(device)
    labels = labels.to(device)
    
    # Load semantic vectors
    print("Loading semantic vectors...")
    semantic_loader = SemanticVectorLoader(config)
    
    # Load class descriptions (you need to provide these)
    class_descriptions = {}  # Load from file
    semantic_vectors = semantic_loader.get_embeddings(class_descriptions)
    semantic_vectors = semantic_vectors.to(device)
    
    # Load knowledge graphs
    print("Building knowledge graphs...")
    fskg = FoodSafetyKnowledgeGraph(config)
    # Load FSKG from files
    # fskg.load_from_files(...)
    
    multi_graphs = MultiSourceGraphs(
        config['dataset']['num_seen_classes'] + config['dataset']['num_unseen_classes']
    )
    # Build adjacency matrices
    # adj_fskg = multi_graphs.build_fskg_adjacency(...)
    # adj_hyperclass = multi_graphs.build_hyperclass_adjacency(...)
    # adj_cooccur = multi_graphs.build_cooccurrence_adjacency(...)
    
    # For demonstration, create dummy adjacencies
    num_classes = config['dataset']['num_seen_classes'] + config['dataset']['num_unseen_classes']
    adj_fskg = torch.eye(num_classes).to(device)
    adj_hyperclass = torch.eye(num_classes).to(device)
    adj_cooccur = torch.eye(num_classes).to(device)
    
    # Initialize KEFS
    print("Initializing KEFS...")
    kefs = KEFS(config).to(device)
    
    # Optimizer
    optimizer_g = optim.Adam(
        list(kefs.msgf.parameters()) + list(kefs.rfdm.parameters()),
        lr=config['training']['kefs_lr'],
        weight_decay=config['training']['kefs_weight_decay']
    )
    
    optimizer_d = optim.Adam(
        kefs.discriminator.parameters(),
        lr=config['training']['kefs_lr'] * 0.5,
        weight_decay=config['training']['kefs_weight_decay']
    )
    
    # Loss function
    kefs_loss_fn = KEFSLoss(
        lambda1=config['training']['lambda1'],
        lambda2=config['training']['lambda2'],
        alpha=config['training']['alpha']
    )
    
    # Training loop
    print("Training KEFS...")
    best_loss = float('inf')
    
    for epoch in range(config['training']['kefs_epochs']):
        kefs.train()
        
        # Sample batch of features
        batch_size = config['training']['kefs_batch_size']
        indices = torch.randperm(len(features))[:batch_size]
        batch_features = features[indices]
        batch_labels = labels[indices]
        
        # Train discriminator
        optimizer_d.zero_grad()
        
        outputs = kefs.forward_train(
            batch_features,
            semantic_vectors,
            batch_labels,
            adj_fskg,
            adj_hyperclass,
            adj_cooccur
        )
        
        # Generate fake features
        with torch.no_grad():
            knowledge_repr = outputs['knowledge_repr']
            batch_knowledge = knowledge_repr[batch_labels]
            fake_features = kefs.rfdm.sample(batch_knowledge, num_samples=1)
        
        # Compute discriminator loss
        real_validity = kefs.discriminator(batch_features)
        fake_validity = kefs.discriminator(fake_features.detach())
        
        d_loss = kefs_loss_fn(
            batch_features,
            fake_features,
            real_validity,
            fake_validity,
            outputs['rfdm_loss'],
            knowledge_repr,
            [adj_fskg, adj_hyperclass, adj_cooccur],
            torch.eye(num_classes).to(device),
            mode='discriminator'
        )
        
        d_loss['total'].backward()
        optimizer_d.step()
        
        # Train generator
        optimizer_g.zero_grad()
        
        outputs = kefs.forward_train(
            batch_features,
            semantic_vectors,
            batch_labels,
            adj_fskg,
            adj_hyperclass,
            adj_cooccur
        )
        
        # Generate fake features for generator training
        knowledge_repr = outputs['knowledge_repr']
        batch_knowledge = knowledge_repr[batch_labels]
        fake_features = kefs.rfdm.sample(batch_knowledge, num_samples=1)
        fake_validity = kefs.discriminator(fake_features)
        
        g_loss = kefs_loss_fn(
            batch_features,
            fake_features,
            outputs['real_validity'],
            fake_validity,
            outputs['rfdm_loss'],
            knowledge_repr,
            [adj_fskg, adj_hyperclass, adj_cooccur],
            torch.eye(num_classes).to(device),
            mode='generator'
        )
        
        g_loss['total'].backward()
        optimizer_g.step()
        
        # Logging
        if epoch % config['system']['log_frequency'] == 0:
            print(f'Epoch {epoch}: D_Loss={d_loss["total"]:.4f}, '
                  f'G_Loss={g_loss["total"]:.4f}, '
                  f'RFDM={g_loss["region_diffusion"]:.4f}')
        
        # Save checkpoint
        if epoch % config['system']['save_frequency'] == 0:
            torch.save({
                'epoch': epoch,
                'kefs_state_dict': kefs.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }, os.path.join(args.output_dir, f'kefs_epoch_{epoch}.pth'))
    
    # Save final model
    torch.save(kefs.state_dict(), os.path.join(args.output_dir, 'kefs_final.pth'))
    print("\nKEFS training completed.")
    
    return kefs


def train_unseen_classifier(config: dict, args: argparse.Namespace):
    """
    Stage 3: Train classifier on synthesized features for unseen classes
    Corresponds to Lines 5-6 in Algorithm 1
    """
    print("\n" + "=" * 80)
    print("Stage 3: Training Unseen Classifier")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load KEFS
    kefs = KEFS(config).to(device)
    kefs.load_state_dict(torch.load(args.kefs_checkpoint, map_location=device))
    kefs.eval()
    
    # Synthesize features for unseen classes
    print("Synthesizing features for unseen classes...")
    unseen_class_ids = list(range(config['dataset']['num_seen_classes'],
                                  config['dataset']['num_seen_classes'] + 
                                  config['dataset']['num_unseen_classes']))
    
    # Load semantic vectors and adjacencies (same as before)
    # ... (load semantic vectors, adjacencies)
    
    # For demonstration
    semantic_vectors = torch.randn(
        config['dataset']['num_seen_classes'] + config['dataset']['num_unseen_classes'],
        config['semantic']['projection_dim']
    ).to(device)
    
    num_classes = config['dataset']['num_seen_classes'] + config['dataset']['num_unseen_classes']
    adj_fskg = torch.eye(num_classes).to(device)
    adj_hyperclass = torch.eye(num_classes).to(device)
    adj_cooccur = torch.eye(num_classes).to(device)
    
    with torch.no_grad():
        synth_features, synth_labels = kefs.synthesize_dataset(
            semantic_vectors,
            unseen_class_ids,
            adj_fskg,
            adj_hyperclass,
            adj_cooccur,
            num_samples_per_class=config['training']['num_synthesized_features']
        )
    
    # Create dataset and loader
    synth_dataset = SynthesizedDataset(synth_features, synth_labels)
    synth_loader = DataLoader(synth_dataset, 
                              batch_size=config['training']['unseen_batch_size'],
                              shuffle=True)
    
    # Train classifier
    classifier = nn.Linear(config['detector']['feature_dim'],
                          config['dataset']['num_unseen_classes']).to(device)
    
    optimizer = optim.Adam(classifier.parameters(), 
                          lr=config['training']['unseen_lr'])
    criterion = nn.CrossEntropyLoss()
    
    print("Training unseen classifier...")
    for epoch in range(config['training']['unseen_epochs']):
        classifier.train()
        epoch_loss = 0
        
        for batch in synth_loader:
            features = batch['features'].to(device)
            labels = batch['labels'].to(device) - config['dataset']['num_seen_classes']
            
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss={epoch_loss/len(synth_loader):.4f}')
    
    # Save classifier
    torch.save(classifier.state_dict(), 
              os.path.join(args.output_dir, 'unseen_classifier.pth'))
    
    print("\nUnseen classifier training completed.")


def main():
    parser = argparse.ArgumentParser(description='Train Zero-Shot Food Safety Detection')
    parser.add_argument('--stage', type=str, required=True,
                       choices=['detector', 'kefs', 'unseen_classifier', 'all'],
                       help='Training stage')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to FSVH dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    parser.add_argument('--detector_checkpoint', type=str,
                       help='Path to detector checkpoint (for KEFS stage)')
    parser.add_argument('--kefs_checkpoint', type=str,
                       help='Path to KEFS checkpoint (for unseen classifier stage)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training stages
    if args.stage == 'detector' or args.stage == 'all':
        train_detector(config, args)
    
    if args.stage == 'kefs' or args.stage == 'all':
        if not args.detector_checkpoint and args.stage != 'all':
            raise ValueError("--detector_checkpoint required for KEFS training")
        train_kefs(config, args)
    
    if args.stage == 'unseen_classifier' or args.stage == 'all':
        if not args.kefs_checkpoint and args.stage != 'all':
            raise ValueError("--kefs_checkpoint required for unseen classifier training")
        train_unseen_classifier(config, args)
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
