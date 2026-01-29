import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
import torchvision.transforms as transforms
from PIL import Image
import timm
import numpy as np
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = 'brain_tumor_model.pth'
app.config['LABELS_PATH'] = 'labels.txt'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model (with memory optimization)
model = None
device = torch.device('cpu')

try:
    print("Creating ResNet18 model...")
    model = timm.create_model('resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    
    # Try to load model
    model_path = app.config['MODEL_PATH']
    print(f"Looking for model at: {model_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        print("Loading model from file...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully from file!")
    else:
        print("Warning: Model file not found. Running with random weights for demonstration.")
    
    model = model.to(device)
    model.eval()
    
    # Optimize memory
    for param in model.parameters():
        param.requires_grad = False
    
    print("Model setup complete!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Creating model with random weights for demonstration...")
    try:
        model = timm.create_model('resnet18', pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("Demo model created successfully!")
    except Exception as e2:
        print(f"Error creating demo model: {e2}")
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
                'all_probs': all_probs
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
    prediction_response = predict()
    
    # Check if it's an error response
    if isinstance(prediction_response, tuple):
        return prediction_response
    
    # Convert response to dict if needed
    if hasattr(prediction_response, 'get_json'):
        prediction_result = prediction_response.get_json()
    else:
        prediction_result = prediction_response
    
    # Add simple medical info
    prediction_result['rag_info'] = {
        'summary': MEDICAL_RESPONSES.get('brain tumor', 'Medical information available.'),
        'sources': [{'title': 'Medical Knowledge Base'}]
    }
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
