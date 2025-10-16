# Zero-Shot Detection of Visual Food Safety Hazards

Official implementation of "Zero-Shot Detection of Visual Food Safety Hazards via Knowledge-Enhanced Feature Synthesis".

## Overview

This repository contains the implementation of ZSFDet, a framework for detecting food safety hazards without requiring training examples for novel hazard types. The system leverages:

- **Food Safety Knowledge Graph (FSKG)**: Encodes relationships between 26 food categories and 48 visual safety attributes
- **Multi-Source Graph Fusion (MSGF)**: Integrates knowledge from multiple sources
- **Region Feature Diffusion Model (RFDM)**: Synthesizes discriminative features for unseen hazard classes

## Installation

```bash
# Clone the repository
git clone https://github.com/XXX/food-safety-zsd.git
cd food-safety-zsd

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA >= 11.1 (for GPU training)

## Dataset Preparation

The FSVH dataset should be organized as follows:

```
data/
├── FSVH/
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   ├── annotations/
│   │   ├── train.json
│   │   └── test.json
│   └── knowledge_graph/
│       ├── food_categories.json
│       ├── visual_attributes.json
│       └── relationships.json
```

## Training

### Stage 1: Train Detector on Seen Classes

```bash
python train.py --stage detector \
                --config config/config.yaml \
                --data_dir data/FSVH \
                --output_dir outputs/detector
```

### Stage 2: Train KEFS

```bash
python train.py --stage kefs \
                --config config/config.yaml \
                --data_dir data/FSVH \
                --detector_checkpoint outputs/detector/best.pth \
                --output_dir outputs/kefs
```

### Stage 3: Train Unseen Classifier

```bash
python train.py --stage unseen_classifier \
                --config config/config.yaml \
                --data_dir data/FSVH \
                --kefs_checkpoint outputs/kefs/best.pth \
                --output_dir outputs/final
```

## Evaluation

```bash
# Zero-Shot Detection (ZSD)
python test.py --mode zsd \
               --config config/config.yaml \
               --checkpoint outputs/final/best.pth \
               --data_dir data/FSVH

# Generalized Zero-Shot Detection (GZSD)
python test.py --mode gzsd \
               --config config/config.yaml \
               --checkpoint outputs/final/best.pth \
               --data_dir data/FSVH \
               --calibration_factor 0.7
```

## Inference

```bash
python inference.py --image path/to/image.jpg \
                    --checkpoint outputs/final/best.pth \
                    --config config/config.yaml \
                    --output output.jpg
```

## Project Structure

```
food-safety-zsd/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── dataset.py               # Dataset loader
│   └── knowledge_graph.py       # FSKG construction
├── models/
│   ├── detector.py              # Faster R-CNN detector
│   ├── gcn.py                   # Graph Convolutional Network
│   ├── msgf.py                  # Multi-Source Graph Fusion
│   ├── rfdm.py                  # Region Feature Diffusion Model
│   └── kefs.py                  # Knowledge-Enhanced Feature Synthesizer
├── utils/
│   ├── losses.py                # Loss functions
│   └── metrics.py               # Evaluation metrics
├── train.py                     # Training script
├── test.py                      # Evaluation script
└── inference.py                 # Inference script
```

## Results

### Zero-Shot Detection (ZSD)

| Method | mAP@50 |
|--------|--------|
| RRFS   | 56.8   |
| **ZSFDet (Ours)** | **63.7** |

### Generalized Zero-Shot Detection (GZSD)

| Method | Seen | Unseen | HM |
|--------|------|--------|-----|
| RRFS   | 68.3 | 52.7   | 59.5 |
| **ZSFDet (Ours)** | **68.9** | **53.5** | **60.2** |

## License

This project is licensed under the MIT License.

## Acknowledgments

- Based on the KEFS framework for zero-shot food detection
- Uses Faster R-CNN with ResNet-101 backbone
- BERT embeddings for semantic representations

