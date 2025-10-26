# app/flask_api.py
"""
Flask API for Face Concern Detector
"""

import os
import sys
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import io
import base64

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import FaceConcernInference
from src.config import Config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize predictor
config = Config()
try:
    predictor = FaceConcernInference()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    predictor = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """API information"""
    return jsonify({
        'name': 'Face Concern Detector API',
        'version': '1.0',
        'endpoints': {
            '/scan': 'POST - Upload image for analysis',
            '/health': 'GET - Check API health',
            '/concerns': 'GET - List supported concerns'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if predictor is not None else 'error',
        'device': str(config.DEVICE),
        'model_loaded': predictor is not None
    })


@app.route('/concerns')
def concerns():
    """List supported skin concerns"""
    return jsonify({
        'concerns': config.CONCERN_LABELS,
        'threshold': config.THRESHOLD
    })


@app.route('/scan', methods=['POST'])
def scan():
    """
    Scan uploaded image for skin concerns
    
    Request:
        - file: Image file (multipart/form-data)
        - return_visualization: Optional, boolean (default: false)
    
    Response:
        - scores: Dictionary of concern scores
        - detected_concerns: List of detected concerns
        - visualization: Base64 encoded image (if requested)
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Invalid file type. Allowed: {ALLOWED_EXTENSIONS}'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Get visualization flag
        return_viz = request.form.get('return_visualization', 'false').lower() == 'true'
        
        # Predict
        results = predictor.predict(filepath)
        
        # Format results for API response
        detected_concerns = [concern for concern, data in results.items() if data['predicted']]
        scores = {concern: data['probability'] for concern, data in results.items()}
        
        # Prepare response
        response = {
            'scores': scores,
            'detected_concerns': detected_concerns,
            'threshold': config.THRESHOLD
        }
        
        # Add visualization if requested (simplified for now)
        if return_viz:
            # For now, we'll just indicate visualization was requested
            # You can enhance this later with GradCAM integration
            response['visualization_requested'] = True
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response), 200
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'error': str(e)}), 500


@app.route('/batch-scan', methods=['POST'])
def batch_scan():
    """
    Scan multiple images
    
    Request:
        - files: Multiple image files
    
    Response:
        - results: List of prediction results
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'No files provided'}), 400
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                # Predict
                result = predictor.predict(filepath)
                result['filename'] = filename
                results.append(result)
                
                # Clean up
                os.remove(filepath)
            
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
    
    return jsonify({'results': results}), 200


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Face Concern Detector API")
    print("="*50)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Concerns: {', '.join(config.CONCERN_LABELS)}")
    print("="*50 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
