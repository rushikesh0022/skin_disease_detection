# Face Concern Detector - Complete Implementation Guide

## Project Overview

A complete deep learning system for detecting facial skin concerns (acne, dark circles, redness, wrinkles) with explainable AI visualizations, optimized for Mac.

## Complete File Structure

```
face-concern-detector/
│
├── README.md                       # Complete documentation
├── requirements.txt                # Python dependencies
├── setup.sh                        # Setup script
│
├── data/
│   ├── raw/                        # Place downloaded dataset here
│   ├── processed/                  # Auto-generated preprocessed data
│   └── sample_images/              # Test images for demo
│
├── models/
│   ├── __init__.py
│   ├── resnet_model.py             # ResNet18 architecture
│   └── saved_weights/              # Trained model weights
│       └── best_model.pth
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # All configuration settings
│   ├── dataset.py                  # Custom dataset loader
│   ├── preprocessing.py            # MTCNN face detection
│   ├── train.py                    # Complete training pipeline
│   ├── inference.py                # Prediction script
│   └── gradcam.py                  # GradCAM visualization
│
├── outputs/
│   ├── visualizations/             # GradCAM heatmaps
│   ├── logs/                       # Training logs
│   └── results/                    # Prediction results
│
└── app/
    └── flask_api.py                # REST API for deployment
```

## Installation Steps

### 1. Setup Environment

```bash
# Clone or download project files
cd face-concern-detector

# Run setup script
bash setup.sh

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

**Primary Dataset (Recommended):**
- Name: Acne-Wrinkles-Spots Classification
- Source: Kaggle
- Size: 600 images (~96 MB)
- Link: https://www.kaggle.com/datasets/ranvijaybalbir/acne-wrinkles-spots-classification

```bash
# Install Kaggle CLI
pip install kaggle

# Setup Kaggle credentials
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d ranvijaybalbir/acne-wrinkles-spots-classification

# Extract
unzip acne-wrinkles-spots-classification.zip -d data/raw/
```

### 3. Prepare Dataset

```python
from src.dataset import KaggleDatasetAdapter, split_dataset

# Convert Kaggle format to our format
adapter = KaggleDatasetAdapter()
annotations_file = adapter.prepare_dataset(
    kaggle_dir='data/raw/acne-wrinkles-spots',
    output_dir='data/processed'
)

# Split into train/val/test
from src.dataset import split_dataset
split_dataset(annotations_file, train_ratio=0.8, val_ratio=0.1)
```

## Usage Guide

### Training the Model

```bash
# Train with default settings
python src/train.py
```

**Expected output:**
- Training time: ~1-2 hours on Mac M1/M2
- Model saved to: `models/saved_weights/best_model.pth`
- Training curves: `outputs/training_curves.png`

**Customization:**
Edit `src/config.py` to change:
- Batch size (default: 16)
- Learning rate (default: 1e-4)
- Number of epochs (default: 30)
- Image size (default: 224x224)

### Running Inference

**Simple prediction:**
```python
from src.inference import FaceConcernPredictor
from src.config import Config

config = Config()
predictor = FaceConcernPredictor(
    model_path='models/saved_weights/best_model.pth',
    config=config
)

# Predict on single image
results = predictor.predict('path/to/image.jpg')

# Print scores
for concern, score in results['scores'].items():
    print(f"{concern}: {score:.2%}")
```

**With visualization:**
```python
predictor.visualize_predictions(
    image_path='path/to/image.jpg',
    save_path='outputs/visualizations/result.png'
)
```

### Flask API Deployment

```bash
# Start API server
python app/flask_api.py

# API will run at http://localhost:5000
```

**API Endpoints:**

1. **Health Check:**
```bash
curl http://localhost:5000/health
```

2. **Scan Image:**
```bash
curl -X POST http://localhost:5000/scan \
  -F "file=@face_image.jpg" \
  -F "return_visualization=true"
```

3. **Batch Scan:**
```bash
curl -X POST http://localhost:5000/batch-scan \
  -F "files=@face1.jpg" \
  -F "files=@face2.jpg"
```

## Key Features

### 1. Multi-Label Classification
- Detects 4 skin concerns simultaneously
- Binary classification per concern
- Confidence scores from 0-100%

### 2. Face Detection
- Automatic face detection using MTCNN
- Face alignment and cropping
- Handles multiple faces (selects highest confidence)

### 3. Explainable AI (GradCAM)
- Visualizes which regions influence predictions
- Separate heatmaps for each concern
- Color-coded overlays on original image

### 4. Mac Optimization
- Auto-detects MPS (Metal Performance Shaders)
- Optimized batch size for M1/M2 chips
- Efficient memory management
- Small dataset support

## Model Architecture

```
Input: RGB Image (224x224)
    ↓
MTCNN Face Detection
    ↓
ResNet18 Backbone (pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Fully Connected (512)
    ↓
ReLU + Dropout
    ↓
Fully Connected (4)
    ↓
Sigmoid Activation
    ↓
Output: [acne, dark_circles, redness, wrinkles]
```

## Configuration Options

**Training Settings:**
```python
# src/config.py
BATCH_SIZE = 16              # Reduce to 8 if memory issues
NUM_EPOCHS = 30              # Increase for better accuracy
LEARNING_RATE = 1e-4         # Lower for fine-tuning
IMAGE_SIZE = 224             # Standard ResNet input
```

**Device Settings:**
```python
# Automatically detects best device
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
```

**Detection Threshold:**
```python
THRESHOLD = 0.5              # Adjust for sensitivity
```

## Expected Performance

### Training Metrics
- Training Loss: ~0.20-0.25 (after 30 epochs)
- Validation Loss: ~0.22-0.27
- Per-class Accuracy: 75-85%
- Overall Accuracy: ~80%

### Inference Speed
- Face Detection: ~0.2-0.3s
- Model Prediction: ~0.05-0.1s
- GradCAM Generation: ~0.3-0.5s
- **Total per image: <1 second**

## Troubleshooting

### Issue: "No face detected"
**Solution:**
- Ensure face is clearly visible
- Use well-lit images
- Face should be front-facing
- Adjust `MIN_FACE_SIZE` in config

### Issue: Out of memory
**Solution:**
```python
# In src/config.py
BATCH_SIZE = 8  # or even 4
NUM_WORKERS = 0
```

### Issue: Poor accuracy
**Solution:**
1. Train for more epochs (50+)
2. Increase dataset size
3. Unfreeze backbone layers:
```python
from models.resnet_model import freeze_backbone
freeze_backbone(model, freeze=False)
```

### Issue: Slow training
**Solution:**
- Close other applications
- Reduce image size to 160
- Use smaller batch size
- Ensure MPS is detected

## Dataset Format

**Expected CSV format:**
```csv
image_name,acne,dark_circles,redness,wrinkles
face001.jpg,1,0,0,0
face002.jpg,1,0,1,0
face003.jpg,0,1,0,1
```

Where:
- 1 = concern present
- 0 = concern absent

## Extending the Project

### Add New Skin Concerns

1. Update config:
```python
NUM_CLASSES = 5
CONCERN_LABELS = ['acne', 'dark_circles', 'redness', 'wrinkles', 'pigmentation']
```

2. Update dataset annotations (add new column)
3. Retrain model

### Use Different Backbone

```python
# In models/resnet_model.py
from models.resnet_model import EfficientNetDetector

model = EfficientNetDetector(num_classes=4)
```

### Custom Data Augmentation

```python
# In src/preprocessing.py, modify create_transforms()
transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
transforms.GaussianBlur(kernel_size=3),
```

## Important Notes

1. **Dataset Size:** 600 images is minimum viable. 1000+ recommended for production.
2. **Training Time:** Varies by Mac model (1-3 hours typical).
3. **Memory:** 8GB RAM sufficient, 16GB recommended.
4. **Storage:** Need ~5GB for project + dataset.
5. **Python Version:** 3.8+ required.

## Citation & References

**ResNet:**
- He et al., "Deep Residual Learning for Image Recognition" (2015)

**GradCAM:**
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017)

**MTCNN:**
- Zhang et al., "Joint Face Detection and Alignment" (2016)

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review configuration settings
3. Verify dataset format
4. Check system requirements

## License

MIT License - Free to use and modify

## Author Notes

This implementation is specifically optimized for:
- Mac M1/M2 chips with MPS
- Small datasets (600-2000 images)
- Fast inference (<1 second)
- Explainable predictions with GradCAM
- Easy deployment via Flask API

All code is production-ready and includes error handling, logging, and documentation.
