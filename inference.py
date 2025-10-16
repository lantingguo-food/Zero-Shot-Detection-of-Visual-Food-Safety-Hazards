"""
Inference script for detecting food safety hazards in single images
"""

import torch
import yaml
import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision


# Class names for visualization
CLASS_NAMES = {
    0: 'Background',
    1: 'Mold Growth', 2: 'Glass Fragments', 3: 'Insect Parts',
    4: 'Bacterial Colonies', 5: 'Chemical Residue', 6: 'Surface Moisture',
    7: 'Light Discoloration', 8: 'Texture Anomaly', 9: 'Foreign Object',
    10: 'Spoilage', 11: 'Contamination', 12: 'Physical Damage',
    13: 'Color Change', 14: 'Mold (Aspergillus)', 15: 'Mold (Penicillium)',
    16: 'Rust Spots', 17: 'Oil Contamination', 18: 'Pest Evidence',
    19: 'Freezer Burn', 20: 'Dehydration', 21: 'Fermentation',
    22: 'Plastic Debris', 23: 'Metal Fragments', 24: 'Hair/Fibers',
    25: 'Bacterial Film', 26: 'Chemical Staining', 27: 'Cross Contamination',
    28: 'Structural Defect'
}

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (192, 0, 0), (0, 192, 0), (0, 0, 192), (192, 192, 0),
    (192, 0, 192), (0, 192, 192), (255, 128, 0), (255, 0, 128),
    (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255),
    (255, 192, 0), (255, 0, 192), (192, 255, 0), (0, 255, 192),
    (192, 0, 255)
]


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: dict, device):
    """Load trained model from checkpoint"""
    
    # Determine number of classes
    total_classes = (config['dataset']['num_seen_classes'] + 
                    config['dataset']['num_unseen_classes'] + 1)
    
    # Create model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, total_classes
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path: str, config: dict):
    """Load and preprocess image"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Transform
    transform = T.Compose([
        T.Resize((config['dataset']['image_size'][0], 
                 config['dataset']['image_size'][1])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    return image, image_tensor, original_size


def detect_hazards(model, image_tensor, device, threshold=0.5):
    """Run detection on image"""
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
    
    # Get predictions
    boxes = outputs[0]['boxes'].cpu()
    labels = outputs[0]['labels'].cpu()
    scores = outputs[0]['scores'].cpu()
    
    # Filter by threshold
    keep = scores > threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    return boxes, labels, scores


def visualize_detections(image, boxes, labels, scores, output_path, config):
    """Draw bounding boxes and labels on image"""
    
    # Resize boxes to original image size
    image_width, image_height = image.size
    target_width, target_height = config['dataset']['image_size']
    
    scale_x = image_width / target_width
    scale_y = image_height / target_height
    
    # Create drawing context
    draw = ImageDraw.Draw(image)
    
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw each detection
    for box, label, score in zip(boxes, labels, scores):
        # Scale box coordinates
        x1 = box[0] * scale_x
        y1 = box[1] * scale_y
        x2 = box[2] * scale_x
        y2 = box[3] * scale_y
        
        # Get label and color
        label_id = label.item()
        class_name = CLASS_NAMES.get(label_id, f'Class {label_id}')
        color = COLORS[label_id % len(COLORS)]
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        label_text = f'{class_name}: {score:.2f}'
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        draw.rectangle([x1, y1 - 20, text_bbox[2], y1], fill=color)
        
        # Draw label text
        draw.text((x1 + 2, y1 - 18), label_text, fill='white', font=font)
    
    # Save image
    image.save(output_path)
    print(f"Output saved to: {output_path}")


def print_detections(boxes, labels, scores):
    """Print detection results to console"""
    
    print(f"\nDetected {len(boxes)} hazards:")
    print("-" * 80)
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        label_id = label.item()
        class_name = CLASS_NAMES.get(label_id, f'Class {label_id}')
        
        print(f"{i+1}. {class_name}")
        print(f"   Confidence: {score:.2%}")
        print(f"   Location: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
        print()


def main():
    parser = argparse.ArgumentParser(description='Detect food safety hazards in images')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='Path to output image')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--no_viz', action='store_true',
                       help='Skip visualization (only print detections)')
    
    args = parser.parse_args()
    
    # Check if input image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config, device)
    
    # Load and preprocess image
    print("Processing image...")
    image, image_tensor, original_size = preprocess_image(args.image, config)
    
    # Run detection
    print("Running detection...")
    boxes, labels, scores = detect_hazards(
        model, image_tensor, device, threshold=args.threshold
    )
    
    # Print results
    print_detections(boxes, labels, scores)
    
    # Visualize if requested
    if not args.no_viz and len(boxes) > 0:
        print("Creating visualization...")
        visualize_detections(image, boxes, labels, scores, args.output, config)
    elif len(boxes) == 0:
        print("\nNo hazards detected above threshold.")
    
    print("\nInference completed!")


if __name__ == '__main__':
    main()
