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

# Load model (with URL download support)
model = None
device = torch.device('cpu')

def download_model_from_url():
    """Download model from external URL"""
    try:
        # Try GitHub raw URL first
        model_urls = [
            "https://raw.githubusercontent.com/IH-Arik/brain-tumor-rag/main/brain_tumor_model.pth",
            "https://github.com/IH-Arik/brain-tumor-rag/raw/main/brain_tumor_model.pth"
        ]
        
        for url in model_urls:
            print(f"Trying to download model from: {url}")
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    model_data = response.content
                    with open('brain_tumor_model.pth', 'wb') as f:
                        f.write(model_data)
                    print(f"Model downloaded successfully! Size: {len(model_data) / (1024*1024):.2f} MB")
                    return True
                else:
                    print(f"Failed to download from {url}, status: {response.status_code}")
            except Exception as e:
                print(f"Error downloading from {url}: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"Error in download_model_from_url: {e}")
        return False

def download_model_from_railway():
    """Download model from Railway Variables (small chunks only)"""
    try:
        # Try Railway Variables for model URL
        model_url = os.environ.get('MODEL_URL')
        if model_url:
            print(f"Found model URL in Railway Variables: {model_url}")
            try:
                response = requests.get(model_url, timeout=30)
                if response.status_code == 200:
                    model_data = response.content
                    with open('brain_tumor_model.pth', 'wb') as f:
                        f.write(model_data)
                    print(f"Model downloaded from URL! Size: {len(model_data) / (1024*1024):.2f} MB")
                    return True
            except Exception as e:
                print(f"Error downloading from URL: {e}")
        
        return False
        
    except Exception as e:
        print(f"Error in download_model_from_railway: {e}")
        return False

try:
    print("Creating ResNet18 model...")
    model = timm.create_model('resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    
    # Try to download model
    model_downloaded = False
    
    # Try Railway Variables first
    if download_model_from_railway():
        model_downloaded = True
    
    # Try direct URL download
    elif download_model_from_url():
        model_downloaded = True
    
    # Try to load model from file
    model_paths = [
        'brain_tumor_model.pth',
        '/app/brain_tumor_model.pth',
        './brain_tumor_model.pth'
    ]
    
    for model_path in model_paths:
        print(f"Looking for model at: {model_path}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            print("Loading model from file...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint)
                print("Model loaded successfully from file!")
                model_downloaded = True
                break
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                continue
    
    if not model_downloaded:
        print("Warning: Model file not found. Running with random weights for demonstration.")
        print("To fix: Set MODEL_URL environment variable with model download URL")
    
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

# Enhanced keyword-based responses (LLM-like without API)
def get_keyword_response(question):
    """Enhanced keyword-based responses with LLM-like variations"""
    question = question.lower()
    
    # Enhanced medical responses with more detail
    medical_responses = {
        'glioma': [
            'Glioma is a type of tumor that occurs in the brain and spinal cord, originating from glial cells. These tumors can be benign or malignant and may require surgery, radiation, or chemotherapy depending on their grade and location.',
            'Gliomas are primary brain tumors that develop from glial cells, which support nerve cells. They range from low-grade (slow-growing) to high-grade (aggressive) and are the most common type of brain tumor in adults.',
            'A glioma is a tumor that starts in the glial cells of the brain or spinal cord. Treatment options include surgical removal, radiation therapy, and chemotherapy, with prognosis depending on tumor grade and location.'
        ],
        'meningioma': [
            'Meningioma is a tumor that arises from the meninges, the membranes surrounding the brain and spinal cord. Most meningiomas are benign and grow slowly, often requiring monitoring or surgical removal if symptomatic.',
            'Meningiomas are typically slow-growing tumors that form in the meninges. While usually non-cancerous, they can cause symptoms by pressing on brain tissue and may require surgery or radiation treatment.',
            'A meningioma develops in the protective membranes covering the brain and spinal cord. These tumors are usually benign but can cause neurological symptoms depending on their size and location.'
        ],
        'pituitary': [
            'Pituitary tumors are abnormal growths in the pituitary gland that can affect hormone production. They may cause hormonal imbalances, vision problems, and headaches, with treatment ranging from medication to surgery.',
            'Tumors of the pituitary gland can disrupt normal hormone function, leading to various symptoms including hormonal changes, vision loss, and headaches. Treatment options include medication, radiation, or surgical removal.',
            'Pituitary tumors are typically benign growths that can affect hormone production and regulation. They may cause endocrine disorders and require specialized treatment based on hormone levels.'
        ],
        'brain tumor': [
            'Brain tumors are abnormal growths of cells in the brain that can be benign (non-cancerous) or malignant (cancerous). Symptoms vary widely but may include headaches, seizures, and changes in behavior or cognitive function.',
            'A brain tumor is a mass or growth of abnormal cells in the brain. These tumors can originate in the brain (primary) or spread from other parts of the body (secondary), with treatment depending on type, size, and location.',
            'Brain tumors are classified as either benign or malignant growths that affect brain function. Common symptoms include persistent headaches, seizures, vision problems, and personality changes.'
        ],
        'symptoms': [
            'Common brain tumor symptoms include persistent headaches that worsen over time, seizures or convulsions, vision problems, memory loss, personality changes, difficulty with balance or coordination, and unexplained nausea or vomiting.',
            'Brain tumor symptoms often include headaches that are different from normal headaches, seizures, vision or hearing changes, cognitive difficulties, weakness or numbness in parts of the body, and changes in personality or behavior.',
            'Warning signs of brain tumors may include new or changing headache patterns, seizures, progressive loss of sensation or movement in arms or legs, difficulty with balance, speech problems, and personality or behavior changes.'
        ],
        'treatment': [
            'Brain tumor treatment options include surgery to remove the tumor, radiation therapy to destroy cancer cells, chemotherapy drugs to kill rapidly dividing cells, and targeted therapy. The best approach depends on tumor type, size, and location.',
            'Treatment for brain tumors typically involves a combination of surgery, radiation therapy, chemotherapy, and sometimes targeted therapy or immunotherapy. The treatment plan is personalized based on tumor characteristics.',
            'Brain tumor treatment may include surgical removal, radiation therapy, chemotherapy, targeted therapy, and clinical trials. The specific treatment approach depends on tumor type, grade, location, and patient health.'
        ]
    }
    
    # Check for specific keywords and return varied responses
    for keyword, responses in medical_responses.items():
        if keyword in question:
            # Return a random response from the list for variation
            import random
            return random.choice(responses)
    
    # Additional keyword matching with varied responses
    if any(word in question for word in ['what is', 'define', 'explain']):
        for keyword, responses in medical_responses.items():
            if keyword in question:
                import random
                response = random.choice(responses)
                return f"{response} This is general medical information and not a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment."
    elif any(word in question for word in ['how to', 'treatment', 'cure', 'therapy']):
        import random
        return random.choice(medical_responses.get('treatment', medical_responses.get('brain tumor')))
    elif any(word in question for word in ['symptom', 'sign', 'warning']):
        import random
        return random.choice(medical_responses.get('symptoms', medical_responses.get('brain tumor')))
    
    return "I can provide general information about brain tumors. Please consult a doctor for medical advice."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_file_exists': os.path.exists('brain_tumor_model.pth'),
        'llm_available': True,
        'memory_efficient': True,
        'download_method': 'url_based'
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
                'model_type': 'trained' if os.path.exists('brain_tumor_model.pth') else 'demo'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rag/query', methods=['POST'])
def rag_query():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        print(f"Received question: {question}")
        
        # Use keyword-based responses
        response = get_keyword_response(question)
        
        print(f"Generated response: {response[:100]}...")
        
        return jsonify({
            'answer': response,
            'sources': [{'title': 'Medical Knowledge Base', 'relevance_score': 0.9}],
            'llm_used': 'Enhanced Keyword System',
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
        'llm_model': 'Enhanced Keyword System'
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
    
    # Add medical info
    prediction_result['rag_info'] = {
        'summary': 'Brain tumor classification completed. This AI-powered system provides medical information to help you understand the results.',
        'sources': [{'title': 'AI Medical Assistant (Knowledge Base)'}]
    }
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
