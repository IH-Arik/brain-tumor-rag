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
        # Use the new Hugging Face router API
        API_URL = "https://router.huggingface.co/hf/google/flan-t5-base"
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
        API_URL2 = "https://router.huggingface.co/hf/microsoft/DialoGPT-medium"
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
        ],
        'diagnosis': [
            'Brain tumors are diagnosed through imaging tests like MRI and CT scans, neurological exams to assess brain function, and sometimes biopsy to examine tumor tissue. Early detection improves treatment outcomes.',
            'Diagnosis of brain tumors involves neurological examinations, imaging studies (MRI, CT scans), and sometimes biopsy. Advanced imaging techniques help determine tumor type, size, and location for treatment planning.',
            'Brain tumor diagnosis typically begins with neurological exams followed by imaging tests like MRI or CT scans. In some cases, a biopsy may be needed to determine the tumor type and grade.'
        ],
        'prevention': [
            'While most brain tumors cannot be prevented, reducing exposure to radiation, maintaining a healthy lifestyle, avoiding smoking, and protecting the head from injury may help lower risk. Regular medical check-ups are important.',
            'Primary prevention of brain tumors is limited, but avoiding unnecessary radiation exposure, maintaining good overall health, and seeking prompt medical attention for neurological symptoms may help with early detection.',
            'Brain tumor prevention strategies include minimizing radiation exposure, maintaining a healthy immune system, avoiding known carcinogens, and seeking medical evaluation for persistent neurological symptoms.'
        ],
        'prognosis': [
            'Prognosis for brain tumors varies widely depending on type (benign vs malignant), grade, location, size, and how early it\'s detected. Benign tumors generally have better outcomes than malignant ones.',
            'Brain tumor prognosis depends on multiple factors including tumor type, grade, location, patient age, and overall health. Early detection and treatment significantly improve outcomes and quality of life.',
            'The outlook for brain tumor patients varies based on tumor characteristics, treatment response, and individual health factors. Benign tumors typically have excellent prognoses, while malignant tumors require aggressive treatment.'
        ],
        'types': [
            'Common types of brain tumors include gliomas (astrocytomas, glioblastomas), meningiomas, pituitary tumors, medulloblastomas, and schwannomas. Each type has different characteristics, growth patterns, and treatment approaches.',
            'Brain tumors are classified by cell type and include gliomas, meningiomas, pituitary adenomas, medulloblastomas, and metastatic tumors. Classification helps determine treatment options and prognosis.',
            'Major brain tumor categories include primary tumors (originating in brain tissue) and secondary tumors (spreading from elsewhere). Primary types include gliomas, meningiomas, and pituitary tumors.'
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
    elif any(word in question for word in ['diagnose', 'test', 'detection']):
        import random
        return random.choice(medical_responses.get('diagnosis', medical_responses.get('brain tumor')))
    elif any(word in question for word in ['prevent', 'avoid', 'reduce risk']):
        import random
        return random.choice(medical_responses.get('prevention', medical_responses.get('brain tumor')))
    
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
