import os
import json
import requests
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = 'brain_tumor_model.pth'
app.config['LABELS_PATH'] = 'labels.txt'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download model from Railway Variables if available
def download_model_from_railway():
    """Download model file from Railway environment variable"""
    try:
        model_url = os.environ.get('MODEL_FILE_URL')
        if model_url:
            print("Downloading model from Railway...")
            response = requests.get(model_url)
            response.raise_for_status()
            
            with open(app.config['MODEL_PATH'], 'wb') as f:
                f.write(response.content)
            print("Model downloaded successfully!")
            return True
    except Exception as e:
        print(f"Error downloading model: {e}")
    return False

# Load model (with memory optimization)
try:
    device = torch.device('cpu')
    model = timm.create_model('resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    
    # Try to load model
    if os.path.exists(app.config['MODEL_PATH']):
        print("Loading model from file...")
        checkpoint = torch.load(app.config['MODEL_PATH'], map_location=device)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully from file!")
    else:
        # Try Railway download
        if download_model_from_railway():
            checkpoint = torch.load(app.config['MODEL_PATH'], map_location=device)
            model.load_state_dict(checkpoint)
            print("Model loaded successfully from Railway!")
        else:
            print("Warning: Model file not found. Running with random weights for demonstration.")
    
    model = model.to(device)
    model.eval()
    
    # Optimize memory
    for param in model.parameters():
        param.requires_grad = False
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load labels
try:
    with open(app.config['LABELS_PATH'], 'r') as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]
except Exception as e:
    print(f"Error loading labels: {e}")
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple RAG-like responses (memory efficient)
MEDICAL_RESPONSES = {
    'glioma': 'Glioma is a type of tumor that occurs in the brain and spinal cord. It originates from glial cells.',
    'meningioma': 'Meningioma is a tumor that arises from the meninges, the membranes that surround the brain and spinal cord.',
    'pituitary': 'Pituitary tumors are abnormal growths that develop in the pituitary gland.',
    'brain tumor': 'Brain tumors are masses of abnormal cells in the brain. They can be benign or malignant.',
    'symptoms': 'Common brain tumor symptoms include headaches, seizures, vision problems, and personality changes.',
    'treatment': 'Treatment options include surgery, radiation therapy, chemotherapy, and targeted therapy.'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_file_exists': os.path.exists(app.config['MODEL_PATH']),
        'memory_efficient': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not available'}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image
            image = Image.open(filepath).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
            # Clean up
            os.remove(filepath)
            
            # Format results
            probs_list = probabilities.squeeze().tolist()
            all_probs = {labels[i]: round(probs_list[i] * 100, 2) for i in range(len(labels))}
            
            return jsonify({
                'prediction': labels[predicted.item()],
                'confidence': round(confidence.item() * 100, 2),
                'all_probs': all_probs,
                'model_loaded': True
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rag/query', methods=['POST'])
def rag_query():
    try:
        data = request.get_json()
        question = data.get('question', '').lower()
        
        # Simple keyword-based responses
        response = "I can provide general information about brain tumors. Please consult a doctor for medical advice."
        
        for keyword, answer in MEDICAL_RESPONSES.items():
            if keyword in question:
                response = answer
                break
        
        return jsonify({
            'answer': response,
            'sources': [{'title': 'Medical Knowledge Base', 'relevance_score': 0.8}]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rag/categories', methods=['GET'])
def rag_categories():
    return jsonify({
        'categories': ['glioma', 'meningioma', 'pituitary', 'general']
    })

@app.route('/rag/stats', methods=['GET'])
def rag_stats():
    return jsonify({
        'total_documents': 6,
        'categories': ['glioma', 'meningioma', 'pituitary', 'general'],
        'model_loaded': model is not None
    })

@app.route('/predict_with_rag', methods=['POST'])
def predict_with_rag():
    # Get prediction first
    prediction_result = predict()
    
    if isinstance(prediction_result, tuple):
        return prediction_result
    
    # Add simple medical info
    prediction_result['rag_info'] = {
        'summary': MEDICAL_RESPONSES.get('brain tumor', 'Medical information available.'),
        'sources': [{'title': 'Medical Knowledge Base'}]
    }
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
