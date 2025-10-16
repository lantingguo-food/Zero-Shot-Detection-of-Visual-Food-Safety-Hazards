"""
Quickstart Example for Zero-Shot Food Safety Detection
This script demonstrates the complete pipeline from data loading to inference
"""

import torch
import yaml
from pathlib import Path

# Import project modules
from data.dataset import get_data_loaders, SemanticVectorLoader
from data.knowledge_graph import FoodSafetyKnowledgeGraph, MultiSourceGraphs
from models.kefs import KEFS
from models.detector import build_detector
from utils.metrics import DetectionMetrics


def load_configuration(config_path='../config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def example_1_data_loading():
    """Example 1: Load and explore the FSVH dataset"""
    print("=" * 80)
    print("Example 1: Data Loading")
    print("=" * 80)
    
    config = load_configuration()
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(
        config, 
        root_dir='../data/FSVH'
    )
    
    # Examine a batch
    batch = next(iter(train_loader))
    print(f"Batch size: {len(batch['images'])}")
    print(f"Image shape: {batch['images'][0].shape}")
    print(f"Number of boxes in first image: {len(batch['boxes'][0])}")
    print(f"Labels in first image: {batch['labels'][0]}")
    print()


def example_2_knowledge_graph():
    """Example 2: Build and explore the Knowledge Graph"""
    print("=" * 80)
    print("Example 2: Knowledge Graph Construction")
    print("=" * 80)
    
    config = load_configuration()
    
    # Initialize knowledge graph
    fskg = FoodSafetyKnowledgeGraph(config)
    
    # Load from files
    fskg.load_from_files(
        food_file='../data/FSVH/knowledge_graph/food_categories.json',
        attr_file='../data/FSVH/knowledge_graph/visual_attributes.json',
        rel_file='../data/FSVH/knowledge_graph/relationships.json'
    )
    
    print(f"Number of food categories: {len(fskg.food_categories)}")
    print(f"Number of visual attributes: {len(fskg.visual_attributes)}")
    print(f"Food-Attribute edges: {fskg.adj_food_attr.shape}")
    print()
    
    # Get full adjacency matrix
    adj_full = fskg.get_full_adjacency_matrix()
    print(f"Full knowledge graph shape: {adj_full.shape}")
    print()


def example_3_semantic_vectors():
    """Example 3: Generate semantic vectors for classes"""
    print("=" * 80)
    print("Example 3: Semantic Vector Generation")
    print("=" * 80)
    
    config = load_configuration()
    
    # Initialize semantic vector loader
    semantic_loader = SemanticVectorLoader(config)
    
    # Example class descriptions
    class_descriptions = {
        0: "Visible fungal mold contamination growing on bread surface with fuzzy texture",
        1: "Small transparent glass fragments embedded in food products",
        2: "Insect body parts including legs and wings found in food"
    }
    
    # Generate embeddings
    print("Generating BERT embeddings...")
    embeddings = semantic_loader.get_embeddings(class_descriptions)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print()


def example_4_kefs_synthesis():
    """Example 4: Feature synthesis with KEFS"""
    print("=" * 80)
    print("Example 4: Feature Synthesis with KEFS")
    print("=" * 80)
    
    config = load_configuration()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize KEFS
    kefs = KEFS(config).to(device)
    
    # Create dummy inputs for demonstration
    num_classes = config['dataset']['num_seen_classes'] + config['dataset']['num_unseen_classes']
    semantic_vectors = torch.randn(num_classes, config['semantic']['projection_dim']).to(device)
    
    # Create dummy adjacency matrices
    adj_fskg = torch.eye(num_classes).to(device)
    adj_hyperclass = torch.eye(num_classes).to(device)
    adj_cooccur = torch.eye(num_classes).to(device)
    
    # Get knowledge representations
    print("Computing knowledge representations...")
    knowledge_repr = kefs.get_knowledge_representations(
        semantic_vectors, adj_fskg, adj_hyperclass, adj_cooccur
    )
    print(f"Knowledge representation shape: {knowledge_repr.shape}")
    
    # Synthesize features for unseen classes
    print("Synthesizing features for unseen classes...")
    unseen_class_ids = list(range(config['dataset']['num_seen_classes'],
                                  config['dataset']['num_seen_classes'] + 
                                  config['dataset']['num_unseen_classes']))
    
    synth_features, synth_labels = kefs.synthesize_dataset(
        semantic_vectors,
        unseen_class_ids,
        adj_fskg,
        adj_hyperclass,
        adj_cooccur,
        num_samples_per_class=10
    )
    
    print(f"Synthesized features shape: {synth_features.shape}")
    print(f"Synthesized labels shape: {synth_labels.shape}")
    print()


def example_5_detector_training():
    """Example 5: Train detector on seen classes"""
    print("=" * 80)
    print("Example 5: Detector Training (Simplified)")
    print("=" * 80)
    
    config = load_configuration()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build detector
    detector = build_detector(config, mode='seen').to(device)
    print(f"Detector built with {config['dataset']['num_seen_classes']} seen classes")
    
    # Get data
    train_loader, _ = get_data_loaders(config, '../data/FSVH')
    
    # Training loop (simplified - just one batch for demo)
    detector.train()
    batch = next(iter(train_loader))
    
    images = batch['images'].to(device)
    targets = []
    for i in range(len(images)):
        target = {
            'boxes': batch['boxes'][i].to(device),
            'labels': batch['labels'][i].to(device)
        }
        targets.append(target)
    
    # Forward pass
    print("Running forward pass...")
    loss_dict = detector(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    print(f"Total loss: {losses.item():.4f}")
    print()


def example_6_evaluation():
    """Example 6: Evaluate model performance"""
    print("=" * 80)
    print("Example 6: Model Evaluation")
    print("=" * 80)
    
    config = load_configuration()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize metrics
    metrics = DetectionMetrics(
        num_classes=config['dataset']['num_unseen_classes'],
        iou_threshold=config['evaluation']['iou_threshold']
    )
    
    # Simulate some predictions and ground truths
    pred_boxes = torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]])
    pred_labels = torch.tensor([1, 2])
    pred_scores = torch.tensor([0.9, 0.85])
    
    gt_boxes = torch.tensor([[105, 105, 205, 205], [295, 295, 395, 395]])
    gt_labels = torch.tensor([1, 2])
    
    metrics.update(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels)
    
    # Compute mAP
    results = metrics.compute_map()
    print(f"mAP: {results['mAP']:.4f}")
    print(f"Per-class AP: {results['class_aps']}")
    print()


def example_7_inference():
    """Example 7: Run inference on a single image"""
    print("=" * 80)
    print("Example 7: Single Image Inference")
    print("=" * 80)
    
    config = load_configuration()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build detector (in inference mode)
    detector = build_detector(config, mode='gzsd').to(device)
    detector.eval()
    
    # Create dummy image
    dummy_image = torch.randn(1, 3, 800, 800).to(device)
    
    print("Running inference...")
    with torch.no_grad():
        predictions = detector([dummy_image])
    
    pred = predictions[0]
    print(f"Number of detections: {len(pred['boxes'])}")
    print(f"Detection scores: {pred['scores'][:5]}")  # Show first 5
    print(f"Detection labels: {pred['labels'][:5]}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("Zero-Shot Food Safety Detection - Quickstart Examples")
    print("=" * 80 + "\n")
    
    try:
        example_1_data_loading()
    except Exception as e:
        print(f"Example 1 failed (expected if data not available): {e}\n")
    
    try:
        example_2_knowledge_graph()
    except Exception as e:
        print(f"Example 2 failed (expected if data not available): {e}\n")
    
    try:
        example_3_semantic_vectors()
    except Exception as e:
        print(f"Example 3 failed: {e}\n")
    
    try:
        example_4_kefs_synthesis()
    except Exception as e:
        print(f"Example 4 failed: {e}\n")
    
    try:
        example_5_detector_training()
    except Exception as e:
        print(f"Example 5 failed (expected if data not available): {e}\n")
    
    try:
        example_6_evaluation()
    except Exception as e:
        print(f"Example 6 failed: {e}\n")
    
    try:
        example_7_inference()
    except Exception as e:
        print(f"Example 7 failed: {e}\n")
    
    print("=" * 80)
    print("Examples completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()
