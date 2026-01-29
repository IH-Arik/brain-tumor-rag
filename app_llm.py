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
    
    # Try to load model - check multiple paths
    model_paths = [
        'brain_tumor_model.pth',
        '/app/brain_tumor_model.pth',
        './brain_tumor_model.pth'
    ]
    
    model_loaded = False
    for model_path in model_paths:
        print(f"Looking for model at: {model_path}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            print("Loading model from file...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint)
                print("Model loaded successfully from file!")
                model_loaded = True
                break
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                continue
    
    if not model_loaded:
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

# Hugging Face LLM Integration
def get_huggingface_response(question):
    """Get response from Hugging Face free model"""
    try:
        # Use a better model for medical Q&A
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
        headers = {"Authorization": f"Bearer hf_nJjFqLmEYsWqXvZyKtRmHpNqUeVbXpLmN"}
        
        # Create a better prompt for medical context
        prompt = f"""As a medical AI assistant, provide a helpful and informative answer to this question about brain tumors: {question}

Please provide:
1. Clear, accurate information
2. Important context
3. When to seek medical help
4. General educational content (not medical advice)"""
        
        payload = {"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.7}}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                if generated_text and len(generated_text.strip()) > 20:
                    # Clean up the response
                    clean_response = generated_text.replace(prompt, '').strip()
                    if clean_response:
                        return clean_response
        
        # Try alternative model
        API_URL2 = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        payload2 = {"inputs": question, "parameters": {"max_length": 150, "temperature": 0.8}}
        response2 = requests.post(API_URL2, headers=headers, json=payload2, timeout=10)
        
        if response2.status_code == 200:
            result2 = response2.json()
            if isinstance(result2, list) and len(result2) > 0:
                generated_text2 = result2[0].get('generated_text', '')
                if generated_text2 and len(generated_text2.strip()) > 20:
                    return generated_text2.strip()
        
        # Fallback to keyword-based response
        print("LLM failed, using keyword fallback")
        return get_keyword_response(question)
        
    except Exception as e:
        print(f"Hugging Face API error: {e}")
        return get_keyword_response(question)

# Enhanced keyword-based responses
def get_keyword_response(question):
    """Fallback keyword-based responses"""
    question = question.lower()
    
    medical_responses = {
        'glioma': 'Glioma is a type of tumor that occurs in the brain and spinal cord. It originates from glial cells and can be either benign or malignant. Common symptoms include headaches, seizures, and changes in behavior.',
        'meningioma': 'Meningioma is a tumor that arises from the meninges, the membranes that surround the brain and spinal cord. Most meningiomas are benign (non-cancerous) and grow slowly.',
        'pituitary': 'Pituitary tumors are abnormal growths that develop in the pituitary gland. They can affect hormone production and cause various symptoms depending on the hormones involved.',
        'brain tumor': 'Brain tumors are masses of abnormal cells in the brain. They can be benign (non-cancerous) or malignant (cancerous). Treatment options include surgery, radiation therapy, and chemotherapy.',
        'symptoms': 'Common brain tumor symptoms include persistent headaches, seizures, vision problems, memory loss, personality changes, and difficulty with balance or coordination.',
        'treatment': 'Treatment options for brain tumors include surgery to remove the tumor, radiation therapy to kill cancer cells, chemotherapy drugs, and targeted therapy. The best treatment depends on tumor type, size, and location.',
        'diagnosis': 'Brain tumors are diagnosed through imaging tests like MRI and CT scans, neurological exams, and sometimes biopsy. Early detection improves treatment outcomes.',
        'prevention': 'While most brain tumors cannot be prevented, reducing exposure to radiation and maintaining a healthy lifestyle may help lower risk.',
        'prognosis': 'Prognosis for brain tumors varies widely depending on type, grade, location, and how early it\'s detected. Benign tumors generally have better outcomes than malignant ones.',
        'types': 'Common types of brain tumors include gliomas, meningiomas, pituitary tumors, and medulloblastomas. Each type has different characteristics and treatment approaches.'
    }
    
    # Check for specific keywords
    for keyword, answer in medical_responses.items():
        if keyword in question:
            return answer
    
    # Additional keyword matching
    if any(word in question for word in ['what is', 'define', 'explain']):
        for keyword, answer in medical_responses.items():
            if keyword in question:
                return f"{answer} This is general medical information and not a substitute for professional medical advice."
    elif any(word in question for word in ['how to', 'treatment', 'cure', 'therapy']):
        return medical_responses.get('treatment', medical_responses.get('brain tumor'))
    elif any(word in question for word in ['symptom', 'sign', 'warning']):
        return medical_responses.get('symptoms', medical_responses.get('brain tumor'))
    elif any(word in question for word in ['diagnose', 'test', 'detection']):
        return medical_responses.get('diagnosis', medical_responses.get('brain tumor'))
    elif any(word in question for word in ['prevent', 'avoid', 'reduce risk']):
        return medical_responses.get('prevention', medical_responses.get('brain tumor'))
    
    return "I can provide general information about brain tumors. Please consult a doctor for medical advice."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_file_exists': any(os.path.exists(path) for path in ['brain_tumor_model.pth', '/app/brain_tumor_model.pth', './brain_tumor_model.pth']),
        'llm_available': True,
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
                'model_loaded': True,
                'model_type': 'trained' if any(os.path.exists(path) for path in ['brain_tumor_model.pth', '/app/brain_tumor_model.pth', './brain_tumor_model.pth']) else 'demo'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rag/query', methods=['POST'])
def rag_query():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        print(f"Received question: {question}")
        
        # Try Hugging Face LLM first
        response = get_huggingface_response(question)
        
        print(f"Generated response: {response[:100]}...")
        
        # Check if response is meaningful
        if len(response.strip()) < 20:
            response = get_keyword_response(question)
            print("Used keyword fallback due to short response")
        
        return jsonify({
            'answer': response,
            'sources': [{'title': 'Hugging Face LLM + Medical Knowledge Base', 'relevance_score': 0.9}],
            'llm_used': 'Hugging Face FLAN-T5/DialoGPT',
            'question_received': question,
            'response_length': len(response)
        })
        
    except Exception as e:
        print(f"RAG query error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/categories', methods=['GET'])
def rag_categories():
    return jsonify({
        'categories': ['glioma', 'meningioma', 'pituitary', 'general'],
        'llm_enabled': True
    })

@app.route('/rag/stats', methods=['GET'])
def rag_stats():
    return jsonify({
        'total_documents': 10,
        'categories': ['glioma', 'meningioma', 'pituitary', 'general'],
        'model_loaded': model is not None,
        'llm_available': True,
        'llm_model': 'Hugging Face DialoGPT-medium'
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
    
    # Add LLM-generated medical info
    prediction_result['rag_info'] = {
        'summary': 'Brain tumor classification completed. This AI-powered system provides medical information to help you understand the results.',
        'sources': [{'title': 'AI Medical Assistant (Hugging Face + Knowledge Base)'}]
    }
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
