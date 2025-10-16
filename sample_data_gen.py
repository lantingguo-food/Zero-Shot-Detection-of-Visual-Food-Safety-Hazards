"""
Generate sample data files for FSVH dataset
This creates the JSON structure for annotations and knowledge graph
"""

import json
import os
import argparse
import numpy as np


def generate_food_categories():
    """Generate sample food categories"""
    categories = [
        {"id": 0, "name": "White Bread", "food_type": "Bakery",
         "composition": {"water_content": 0.38, "ph_level": 5.5, "protein": 0.09}},
        {"id": 1, "name": "Whole Wheat Bread", "food_type": "Bakery",
         "composition": {"water_content": 0.36, "ph_level": 5.6, "protein": 0.11}},
        {"id": 2, "name": "Fresh Strawberries", "food_type": "Fruit",
         "composition": {"water_content": 0.91, "ph_level": 3.3, "sugar": 0.05}},
        {"id": 3, "name": "Fresh Raspberries", "food_type": "Fruit",
         "composition": {"water_content": 0.90, "ph_level": 3.4, "sugar": 0.05}},
        {"id": 4, "name": "Cheddar Cheese", "food_type": "Dairy",
         "composition": {"water_content": 0.37, "ph_level": 5.3, "fat": 0.33}},
        {"id": 5, "name": "Fresh Chicken", "food_type": "Meat",
         "composition": {"water_content": 0.74, "ph_level": 6.2, "protein": 0.20}},
    ]
    return {"categories": categories}


def generate_visual_attributes():
    """Generate sample visual attributes"""
    attributes = [
        {"id": 0, "name": "Fuzzy Growth Pattern", "category": "Decomposition",
         "description": "Visible fuzzy or hairy growth on food surface indicating mold"},
        {"id": 1, "name": "Green-Blue Discoloration", "category": "Appearance",
         "description": "Green or blue color patches indicating mold contamination"},
        {"id": 2, "name": "White-Gray Patches", "category": "Appearance",
         "description": "White or gray colored patches on food surface"},
        {"id": 3, "name": "Surface Moisture", "category": "Appearance",
         "description": "Excessive moisture or sliminess on food surface"},
        {"id": 4, "name": "Dark Spots", "category": "Appearance",
         "description": "Dark brown or black spots indicating decay"},
        {"id": 5, "name": "Glass-like Shine", "category": "Contamination",
         "description": "Shiny glass-like fragments in food"},
    ]
    return {"attributes": attributes}


def generate_relationships(num_foods=6, num_attrs=6):
    """Generate sample relationships"""
    
    # Food-Attribute relationships
    food_attr = []
    for food_id in range(num_foods):
        for attr_id in range(num_attrs):
            # Random frequency and importance
            freq = np.random.uniform(0.1, 0.9)
            importance = np.random.uniform(0.5, 1.0)
            food_attr.append({
                "food_id": food_id,
                "attribute_id": attr_id,
                "frequency": float(freq),
                "importance": float(importance)
            })
    
    # Food-Food relationships
    food_food = []
    for i in range(num_foods):
        for j in range(i+1, num_foods):
            comp_sim = np.random.uniform(0.2, 0.9)
            hazard_sim = np.random.uniform(0.2, 0.9)
            proc_sim = np.random.uniform(0.2, 0.9)
            food_food.append({
                "food_id_1": i,
                "food_id_2": j,
                "compositional_similarity": float(comp_sim),
                "hazard_similarity": float(hazard_sim),
                "processing_similarity": float(proc_sim)
            })
    
    # Attribute-Attribute relationships
    attr_attr = []
    for i in range(num_attrs):
        for j in range(i+1, num_attrs):
            cooccur = np.random.uniform(0.1, 0.8)
            attr_attr.append({
                "attribute_id_1": i,
                "attribute_id_2": j,
                "cooccurrence": float(cooccur)
            })
    
    return {
        "food_attribute": food_attr,
        "food_food": food_food,
        "attribute_attribute": attr_attr
    }


def generate_annotations(num_images=100, num_classes=28):
    """Generate sample COCO-format annotations"""
    
    images = []
    annotations = []
    categories = []
    
    # Create categories
    category_names = [
        "Mold Growth", "Glass Fragments", "Insect Parts", "Bacterial Colonies",
        "Chemical Residue", "Surface Moisture", "Light Discoloration", "Texture Anomaly",
        "Foreign Object", "Spoilage", "Contamination", "Physical Damage",
        "Color Change", "Mold (Aspergillus)", "Mold (Penicillium)", "Rust Spots",
        "Oil Contamination", "Pest Evidence", "Freezer Burn", "Dehydration",
        "Fermentation", "Plastic Debris", "Metal Fragments", "Hair/Fibers",
        "Bacterial Film", "Chemical Staining", "Cross Contamination", "Structural Defect"
    ]
    
    for i in range(min(num_classes, len(category_names))):
        categories.append({
            "id": i + 1,
            "name": category_names[i],
            "supercategory": "Food Safety Hazard"
        })
    
    # Create images and annotations
    ann_id = 1
    for img_id in range(1, num_images + 1):
        # Add image
        images.append({
            "id": img_id,
            "file_name": f"img_{img_id:04d}.jpg",
            "width": 800,
            "height": 600
        })
        
        # Add 1-3 annotations per image
        num_anns = np.random.randint(1, 4)
        for _ in range(num_anns):
            # Random box
            x = np.random.randint(50, 600)
            y = np.random.randint(50, 400)
            w = np.random.randint(50, 200)
            h = np.random.randint(50, 150)
            
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": np.random.randint(1, min(num_classes, len(category_names)) + 1),
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1
    
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }


def generate_class_descriptions(num_classes=28):
    """Generate semantic descriptions for classes"""
    descriptions = {
        "1": "Visible fungal mold contamination growing on bread surface showing fuzzy texture and discoloration ranging from white to green-blue",
        "2": "Small transparent glass fragments embedded in food products, appearing as shiny irregular pieces indicating physical contamination",
        "3": "Insect body parts including legs, wings, or antennae found in food products indicating pest infestation",
        "4": "Visible bacterial colonies forming on food surface with slimy texture and potential discoloration",
        "5": "Chemical residue stains visible on food surface showing unusual coloration or texture patterns",
        "6": "Excessive surface moisture or condensation on food indicating improper storage conditions",
        "7": "Light discoloration or fading of food surface color indicating early quality deterioration",
        "8": "Texture anomalies including unusual softness, hardness, or surface irregularities in food structure"
    }
    
    return {"class_descriptions": descriptions}


def generate_class_splits(num_seen=20, num_unseen=8):
    """Generate seen/unseen class splits"""
    return {
        "seen_classes": list(range(1, num_seen + 1)),
        "unseen_classes": list(range(num_seen + 1, num_seen + num_unseen + 1))
    }


def main():
    parser = argparse.ArgumentParser(description='Generate sample FSVH data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for generated files')
    parser.add_argument('--num_train', type=int, default=100,
                       help='Number of training images')
    parser.add_argument('--num_test', type=int, default=50,
                       help='Number of test images')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'knowledge_graph'), exist_ok=True)
    
    print("Generating food categories...")
    food_cats = generate_food_categories()
    with open(os.path.join(args.output_dir, 'knowledge_graph', 'food_categories.json'), 'w') as f:
        json.dump(food_cats, f, indent=2)
    
    print("Generating visual attributes...")
    vis_attrs = generate_visual_attributes()
    with open(os.path.join(args.output_dir, 'knowledge_graph', 'visual_attributes.json'), 'w') as f:
        json.dump(vis_attrs, f, indent=2)
    
    print("Generating relationships...")
    rels = generate_relationships()
    with open(os.path.join(args.output_dir, 'knowledge_graph', 'relationships.json'), 'w') as f:
        json.dump(rels, f, indent=2)
    
    print(f"Generating training annotations ({args.num_train} images)...")
    train_anns = generate_annotations(args.num_train)
    with open(os.path.join(args.output_dir, 'annotations', 'train.json'), 'w') as f:
        json.dump(train_anns, f, indent=2)
    
    print(f"Generating test annotations ({args.num_test} images)...")
    test_anns = generate_annotations(args.num_test)
    with open(os.path.join(args.output_dir, 'annotations', 'test.json'), 'w') as f:
        json.dump(test_anns, f, indent=2)
    
    print("Generating class descriptions...")
    descriptions = generate_class_descriptions()
    with open(os.path.join(args.output_dir, 'class_descriptions.json'), 'w') as f:
        json.dump(descriptions, f, indent=2)
    
    print("Generating class splits...")
    splits = generate_class_splits()
    with open(os.path.join(args.output_dir, 'class_splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSample data generated successfully in: {args.output_dir}")
    print("\nNote: You still need to add actual images to:")
    print(f"  - {os.path.join(args.output_dir, 'images/train/')}")
    print(f"  - {os.path.join(args.output_dir, 'images/test/')}")


if __name__ == '__main__':
    main()
