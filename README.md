# 🔍 Face Concern Detector

AI-powered skin concern detection system using deep learning to identify acne, dark circles, redness, and wrinkles from facial images.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features

- **Multi-label Classification**: Detects multiple skin concerns simultaneously
- **Real-time Analysis**: Fast inference using optimized ResNet18 architecture
- **Web Interface**: Beautiful, responsive frontend for easy image upload
- **REST API**: Flask-based API for integration with other applications
- **High Accuracy**: Trained model achieving 90%+ accuracy
- **Mac Optimized**: Uses Metal Performance Shaders (MPS) for acceleration

## 🧠 Model Architecture

- **Base Model**: ResNet18 with ImageNet pre-training
- **Task**: Multi-label binary classification
- **Classes**: 4 skin concerns (acne, dark_circles, redness, wrinkles)
- **Input**: 224x224 RGB images
- **Output**: Probability scores for each concern
- **Threshold**: 70% confidence for detection

## 📊 Dataset

- **Combined Dataset**: 690+ images from multiple Kaggle datasets
- **Sources**:
  - Acne-Wrinkles-Spots Classification
  - Skin Defects (Acne, Redness, and Bags Under Eyes)
- **Split**: 80% train, 10% validation, 10% test
- **Preprocessing**: Face detection, cropping, and augmentation

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install required packages
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**:

```bash
git clone https://github.com/rushikesh0022/skin_disease_detection.git
cd skin_disease_detection
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Download and prepare datasets**:

```bash
python -c "
from src.dataset import KaggleDatasetAdapter
adapter = KaggleDatasetAdapter()
adapter.prepare_combined_dataset('./data/processed')
"
```

4. **Train the model** (optional - pre-trained weights included):

```bash
python src/train.py
```

### Usage

#### 🌐 Web Interface

1. **Start the Flask API**:

```bash
python app/flask_api.py
```

2. **Start the frontend server**:

```bash
python -m http.server 8000
```

3. **Open in browser**:

```
http://localhost:8000/frontend.html
```

#### 🔧 Command Line Interface

**Analyze a single image**:

```bash
python inference.py path/to/your/image.jpg
```

**Test the model**:

```bash
python test_model.py
```

#### 🔌 API Usage

**Health Check**:

```bash
curl http://localhost:5001/health
```

**Image Analysis**:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5001/scan
```

## 📁 Project Structure

```
skin_disease_detection/
├── src/                          # Source code
│   ├── config.py                # Configuration settings
│   ├── dataset.py               # Dataset handling and preprocessing
│   ├── preprocessing.py         # Image preprocessing utilities
│   └── train.py                 # Training script
├── models/                       # Model architecture and weights
│   ├── resnet_model.py          # ResNet18 model definition
│   └── saved_weights/           # Trained model weights
├── app/                         # Flask API application
│   └── flask_api.py             # REST API endpoints
├── frontend.html                # Web interface
├── inference.py                 # Single image inference script
├── test_model.py               # Model testing utilities
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🎛️ Configuration

Edit `src/config.py` to customize:

- **Model settings**: Architecture, classes, pretrained weights
- **Training parameters**: Batch size, learning rate, epochs
- **Detection threshold**: Confidence threshold for positive detection
- **Paths**: Data directories, model save locations

## 🧪 Testing

**Run comprehensive tests**:

```bash
python test_model.py
```

**Test with sample images**:

```bash
python inference.py data/processed/sample_image.jpg
```

**Check model metrics**:

- View training curves: `outputs/training_curves.png`
- Test results: `outputs/test_results/`

## 📈 Results

- **Training Accuracy**: 90%+
- **Validation Loss**: 0.186 (best)
- **Detection Performance**:
  - Acne: High precision with 70%+ confidence
  - Redness: Excellent detection for inflammatory conditions
  - Dark Circles: Good performance on under-eye concerns
  - Wrinkles: Reliable detection of aging signs

## 🔧 API Endpoints

### `GET /health`

Check API health status

### `POST /scan`

Analyze uploaded image

- **Input**: Form data with 'file' field
- **Output**: JSON with concern scores and detections

**Example Response**:

```json
{
  "scores": {
    "acne": 0.72,
    "dark_circles": 0.45,
    "redness": 0.68,
    "wrinkles": 0.38
  },
  "detected_concerns": ["acne", "redness"],
  "threshold": 0.7
}
```

## 🛠️ Development

### Training Custom Model

1. **Prepare your dataset**:

```bash
python -c "
from src.dataset import create_annotation_csv
create_annotation_csv('data/your_images', 'annotations.csv')
"
```

2. **Adjust configuration**:
   Edit `src/config.py` for your dataset

3. **Train**:

```bash
python src/train.py
```

### Extending the Model

- Add new concern classes in `src/config.py`
- Modify model architecture in `models/resnet_model.py`
- Update preprocessing in `src/preprocessing.py`

## 📱 Web Interface Features

- **Drag & Drop**: Easy image upload
- **Real-time Preview**: Image preview before analysis
- **Visual Results**: Color-coded concern cards
- **Confidence Scores**: Percentage confidence for each detection
- **Responsive Design**: Works on desktop and mobile

## 🔬 Technical Details

- **Framework**: PyTorch 2.0+
- **Backend**: Flask with CORS support
- **Frontend**: Pure HTML/CSS/JavaScript
- **Optimization**: MPS acceleration on Apple Silicon
- **Image Processing**: PIL with automatic resizing
- **Model Format**: PyTorch checkpoint (.pth)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Datasets**: Kaggle contributors for skin concern datasets
- **Architecture**: ResNet paper authors
- **Framework**: PyTorch team for the excellent deep learning framework
- **Inspiration**: Dermatology research and AI applications in healthcare

---

**Made with ❤️ for advancing AI in dermatology**
