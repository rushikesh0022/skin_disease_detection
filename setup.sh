#!/bin/bash
# setup.sh - Setup script for Face Concern Detector

echo "========================================="
echo "Face Concern Detector - Setup Script"
echo "========================================="

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/sample_images
mkdir -p models/saved_weights
mkdir -p outputs/visualizations
mkdir -p outputs/logs
mkdir -p outputs/results
mkdir -p notebooks
mkdir -p src
mkdir -p models
mkdir -p app

echo "✓ Directories created"

# Create __init__.py files
touch models/__init__.py
touch src/__init__.py
touch app/__init__.py

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

echo "✓ Virtual environment created"

# Activate virtual environment
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"

echo ""
echo "After activation, install dependencies with:"
echo "  pip install -r requirements.txt"

# Dataset instructions
echo ""
echo "========================================="
echo "DATASET SETUP"
echo "========================================="
echo "Download the Kaggle dataset:"
echo "  1. Install Kaggle CLI: pip install kaggle"
echo "  2. Configure API key: ~/.kaggle/kaggle.json"
echo "  3. Download dataset:"
echo "     kaggle datasets download -d ranvijaybalbir/acne-wrinkles-spots-classification"
echo "  4. Extract to data/raw/"
echo ""

echo "========================================="
echo "Setup complete! Next steps:"
echo "========================================="
echo "1. Activate environment: source venv/bin/activate"
echo "2. Install packages: pip install -r requirements.txt"
echo "3. Download dataset (see instructions above)"
echo "4. Run: python -m src.train"
echo "========================================="
