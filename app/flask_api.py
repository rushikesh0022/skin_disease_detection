# -*- coding: utf-8 -*-
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
from src.advanced_visualization import create_complete_analysis
from models.resnet_model import SkinConcernDetector

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize predictor and model for advanced visualization
config = Config()
try:
    predictor = FaceConcernInference()
    
    # Load model for advanced visualization
    model = SkinConcernDetector(num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    
    print("Model and advanced visualization loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    predictor = None
    model = None


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
    Enhanced scan with advanced overlays for each concern type
    
    Request:
        - file: Image file (multipart/form-data)
        - return_visualization: Optional, boolean (default: true)
    
    Response:
        - scores: Dictionary of concern scores
        - detected_concerns: List of detected concerns
        - visualization: Base64 encoded image with overlays
        - detailed_analysis: Precise localization data
    """
    if predictor is None or model is None:
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
        
        # Get visualization flag (default to true now)
        return_viz = request.form.get('return_visualization', 'true').lower() == 'true'
        
        # Predict basic scores
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
        
        # Add advanced visualization with overlays
        if return_viz:
            try:
                # Run advanced analysis with precise localization
                analysis_results = create_complete_analysis(model, filepath, config)
                
                # Convert detailed visualization to base64
                buffer = io.BytesIO()
                analysis_results['detailed_visualization'].save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                response['visualization'] = f"data:image/png;base64,{img_str}"
                
                # Add detailed analysis data
                detailed_analysis = {}
                for concern, data in analysis_results['localization_data'].items():
                    locations_summary = []
                    
                    if concern == 'acne' and data['locations']:
                        locations_summary = [
                            {
                                'type': 'acne_spot',
                                'position': {'x': loc['x'], 'y': loc['y']},
                                'severity': loc['severity'],
                                'radius': loc['radius']
                            } for loc in data['locations']
                        ]
                    elif concern == 'dark_circles' and data['locations']:
                        locations_summary = [
                            {
                                'type': 'dark_circle_heatmap',
                                'region': loc['name'],
                                'severity': loc['severity'],
                                'bbox': loc['bbox']
                            } for loc in data['locations']
                        ]
                    elif concern == 'redness' and data['locations']:
                        locations_summary = [
                            {
                                'type': 'redness_heatmap', 
                                'region': loc['name'],
                                'severity': loc['severity'],
                                'bbox': loc['bbox']
                            } for loc in data['locations']
                        ]
                    elif concern == 'wrinkles' and data['locations']:
                        locations_summary = [
                            {
                                'type': 'wrinkle_line',
                                'area': loc['area'],
                                'start': loc['start'],
                                'end': loc['end'],
                                'severity': loc['severity'],
                                'length': loc['length']
                            } for loc in data['locations']
                        ]
                    
                    detailed_analysis[concern] = {
                        'confidence': f"{data['score']:.1%}",
                        'severity': data['severity'],
                        'locations_count': len(data['locations']),
                        'locations': locations_summary
                    }
                
                response['detailed_analysis'] = detailed_analysis
                
            except Exception as viz_error:
                print(f"Visualization error: {viz_error}")
                response['visualization_error'] = str(viz_error)
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response), 200
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'error': str(e)}), 500


@app.route('/advanced-scan', methods=['POST'])
def advanced_scan():
    """
    Advanced scan with precise localization and overlays
    
    Request:
        - file: Image file (multipart/form-data)
    
    Response:
        - analysis: Detailed analysis with locations
        - visualization: Base64 encoded detailed image
        - report: Base64 encoded analysis report
    """
    if predictor is None or model is None:
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
        
        # Run advanced analysis
        analysis_results = create_complete_analysis(model, filepath, config)
        
        # Convert images to base64
        def image_to_base64(pil_image):
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
        
        # Format detailed response
        detailed_analysis = {}
        
        for concern, data in analysis_results['localization_data'].items():
            locations_summary = []
            
            if concern == 'acne':
                locations_summary = [
                    {
                        'type': 'spot',
                        'position': {'x': loc['x'], 'y': loc['y']},
                        'severity': loc['severity'],
                        'radius': loc['radius']
                    } for loc in data['locations']
                ]
            elif concern == 'dark_circles':
                locations_summary = [
                    {
                        'type': 'heatmap_area',
                        'region': loc['name'],
                        'severity': loc['severity'],
                        'bbox': loc['bbox']
                    } for loc in data['locations']
                ]
            elif concern == 'redness':
                locations_summary = [
                    {
                        'type': 'redness_area', 
                        'region': loc['name'],
                        'severity': loc['severity'],
                        'bbox': loc['bbox']
                    } for loc in data['locations']
                ]
            elif concern == 'wrinkles':
                locations_summary = [
                    {
                        'type': 'line',
                        'area': loc['area'],
                        'start': loc['start'],
                        'end': loc['end'],
                        'severity': loc['severity'],
                        'length': loc['length']
                    } for loc in data['locations']
                ]
            
            detailed_analysis[concern] = {
                'confidence': f"{data['score']:.1%}",
                'severity': data['severity'],
                'locations_count': len(data['locations']),
                'locations': locations_summary
            }
        
        response = {
            'analysis': detailed_analysis,
            'visualization': image_to_base64(analysis_results['detailed_visualization']),
            'report': image_to_base64(analysis_results['analysis_panel']),
            'summary': {
                'total_concerns': len(detailed_analysis),
                'detected_concerns': list(detailed_analysis.keys())
            }
        }
        
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
        port=5000,
        debug=True
    )
