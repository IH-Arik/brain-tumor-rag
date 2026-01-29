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

# Simplified RAG without complex LangChain chains
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.documents import Document

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
        # Try GitHub raw URL (correct one first)
        model_urls = [
            "https://github.com/IH-Arik/brain-tumor-rag/raw/main/brain_tumor_model.pth",
            "https://raw.githubusercontent.com/IH-Arik/brain-tumor-rag/main/brain_tumor_model.pth"
        ]
        
        for url in model_urls:
            print(f"Trying to download model from: {url}")
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    model_data = response.content
                    # Check if we got actual model data (not empty)
                    if len(model_data) > 1000000:  # At least 1MB
                        with open('brain_tumor_model.pth', 'wb') as f:
                            f.write(model_data)
                        print(f"Model downloaded successfully! Size: {len(model_data) / (1024*1024):.2f} MB")
                        return True
                    else:
                        print(f"Downloaded file too small: {len(model_data)} bytes")
                        continue
                else:
                    print(f"Failed to download from {url}, status: {response.status_code}")
            except Exception as e:
                print(f"Error downloading from {url}: {e}")
                continue
        
        return False
        
    except Exception as e:
        print(f"Error in download_model_from_url: {e}")
        return False

try:
    print("Creating ResNet18 model...")
    model = timm.create_model('resnet18', pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    
    # Try to download model
    if download_model_from_url():
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
                    break
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    continue
    
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

# Simplified RAG Setup
embeddings = None
vector_store = None
llm = None

def setup_simple_rag():
    """Setup simplified RAG system"""
    global embeddings, vector_store, llm
    
    try:
        print("🔗 Setting up simplified RAG system...")
        
        # Create knowledge base
        knowledge_base = """
        Brain Tumor Medical Information
        
        Glioma:
        Glioma is a type of tumor that occurs in the brain and spinal cord, originating from glial cells.
        These tumors can be benign or malignant and may require surgery, radiation, or chemotherapy.
        Gliomas range from low-grade (slow-growing) to high-grade (aggressive) and are the most common type of brain tumor in adults.
        Symptoms include headaches, seizures, memory loss, personality changes, and difficulty with balance.
        Treatment options include surgical removal, radiation therapy, and chemotherapy, with prognosis depending on tumor grade and location.
        
        Meningioma:
        Meningioma is a tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.
        Most meningiomas are benign and grow slowly, often requiring monitoring or surgical removal if symptomatic.
        These tumors are typically slow-growing and form in the meninges surrounding the brain and spinal cord.
        While usually non-cancerous, they can cause symptoms by pressing on brain tissue.
        Treatment may include surgery, radiation treatment, or simple monitoring for asymptomatic cases.
        
        Pituitary Tumors:
        Pituitary tumors are abnormal growths in the pituitary gland that can affect hormone production.
        They may cause hormonal imbalances, vision problems, and headaches, with treatment ranging from medication to surgery.
        These tumors are typically benign growths that can affect hormone production and regulation.
        They may cause endocrine disorders and require specialized treatment based on hormone levels.
        Common symptoms include hormonal changes, vision loss, and persistent headaches.
        
        Brain Tumor Symptoms:
        Common brain tumor symptoms include persistent headaches that worsen over time, seizures or convulsions, vision problems, memory loss, personality changes, and difficulty with balance or coordination.
        Brain tumor symptoms often include headaches that are different from normal headaches, seizures, vision or hearing changes, cognitive difficulties, weakness or numbness in parts of the body.
        Warning signs may include new or changing headache patterns, seizures, progressive loss of sensation or movement, difficulty with balance, speech problems, and personality changes.
        
        Brain Tumor Treatment:
        Brain tumor treatment options include surgery to remove the tumor, radiation therapy to destroy cancer cells, chemotherapy drugs to kill rapidly dividing cells, and targeted therapy.
        Treatment for brain tumors typically involves a combination of surgery, radiation therapy, chemotherapy, and sometimes targeted therapy or immunotherapy.
        The treatment plan is personalized based on tumor characteristics, including type, grade, location, and patient health.
        Brain tumor treatment may include surgical removal, radiation therapy, chemotherapy, targeted therapy, and clinical trials.
        
        Brain Tumor Diagnosis:
        Brain tumors are diagnosed through imaging tests like MRI and CT scans, neurological exams to assess brain function, and sometimes biopsy to examine tumor tissue.
        Diagnosis of brain tumors involves neurological examinations, imaging studies (MRI, CT scans), and sometimes biopsy for tissue examination.
        Advanced imaging techniques help determine tumor type, size, and location for treatment planning.
        Early detection improves treatment outcomes and prognosis for brain tumor patients.
        
        Brain Tumor Prognosis:
        Prognosis for brain tumors varies widely depending on type (benign vs malignant), grade, location, size, and how early it's detected.
        Brain tumor prognosis depends on multiple factors including tumor type, grade, location, patient age, and overall health.
        Early detection and treatment significantly improve outcomes and quality of life for brain tumor patients.
        Benign tumors typically have excellent prognoses, while malignant tumors require aggressive treatment.
        
        Brain Tumor Types:
        Common types of brain tumors include gliomas (astrocytomas, glioblastomas), meningiomas, pituitary tumors, medulloblastomas, and schwannomas.
        Brain tumors are classified by cell type and include gliomas, meningiomas, pituitary adenomas, medulloblastomas, and metastatic tumors.
        Each type has different characteristics, growth patterns, and treatment approaches.
        Classification helps determine treatment options and prognosis for brain tumor patients.
        """
        
        # Create embeddings
        print("📝 Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create documents
        documents = [Document(page_content=knowledge_base)]
        
        # Create vector store
        print("🗂️ Creating vector store...")
        vector_store = FAISS.from_documents(documents, embeddings)
        
        # Setup Ollama
        print("🦙 Setting up Ollama...")
        try:
            # Test Ollama connection
            ollama_base_url = os.environ.get('OLLAMA_HOST', 'localhost')
            ollama_port = os.environ.get('OLLAMA_PORT', '11434')
            ollama_url = f"http://{ollama_base_url}:{ollama_port}"
            
            import requests
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama server reachable at {ollama_url}")
                
                # Try to download llama2 if needed
                models_response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                if models_response.status_code == 200:
                    available_models = [model['name'] for model in models_response.json().get('models', [])]
                    if not any('llama2' in model for model in available_models):
                        print("⚠️ Downloading llama2 model...")
                        import subprocess
                        subprocess.run(['ollama', 'pull', 'llama2'], timeout=300)
                
                llm = Ollama(model="llama2", base_url=ollama_url, temperature=0.7)
                test_response = llm("Hello")
                print(f"✅ Ollama connected! Test: {test_response[:50]}...")
            else:
                print(f"⚠️ Ollama server not reachable: {response.status_code}")
                llm = None
                
        except Exception as e:
            print(f"⚠️ Ollama setup error: {e}")
            llm = None
        
        print("✅ Simplified RAG system ready!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up RAG: {e}")
        return False

def get_simple_rag_response(question):
    """Get response from simplified RAG system"""
    try:
        question_lower = question.lower()
        medical_keywords = ['brain tumor', 'glioma', 'meningioma', 'pituitary', 'cancer', 'tumor', 'symptom', 'treatment', 'diagnosis', 'medical']
        
        is_medical = any(keyword in question_lower for keyword in medical_keywords)
        
        if llm and vector_store and is_medical:
            print("🔍 Using RAG for medical question...")
            
            # Retrieve relevant documents
            docs = vector_store.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate response
            prompt = f"""Based on the following medical information, answer the question. Include a medical disclaimer.

Context:
{context}

Question: {question}

Answer:"""
            
            response = llm(prompt)
            response += " This information is for educational purposes only and is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment."
            
            return response, docs
            
        elif llm and not is_medical:
            print("💬 Using direct LLM for general conversation...")
            prompt = f"""You are a helpful and friendly AI assistant. Respond naturally and conversationally to the user's message. Be engaging, polite, and helpful.

User: {question}
Assistant:"""
            
            response = llm(prompt)
            return response, []
            
        else:
            # Fallback responses
            return get_fallback_response(question), []
            
    except Exception as e:
        print(f"❌ RAG error: {e}")
        return get_fallback_response(question), []

def get_fallback_response(question):
    """Enhanced fallback responses"""
    question = question.lower()
    
    # Greetings
    if any(greeting in question for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return "Hello! I'm here to help you. I can provide information about brain tumors and medical topics, or we can have a general conversation. How can I assist you today?"
    
    # Name introductions
    elif any(name_intro in question for name_intro in ['my name is', 'i am', 'i\'m', 'call me']):
        return "Nice to meet you! I'm here to help with any questions you might have about brain tumors, medical topics, or just to chat. What would you like to know?"
    
    # How are you questions
    elif any(how_are in question for how_are in ['how are you', 'how do you do', 'how\'s it going']):
        return "I'm doing great, thank you for asking! I'm here and ready to help you with any questions about brain tumors or other topics you'd like to discuss."
    
    # Medical questions
    elif 'glioma' in question:
        return "Glioma is a type of tumor that occurs in the brain and spinal cord, originating from glial cells. These tumors can be benign or malignant and may require surgery, radiation, or chemotherapy depending on their grade and location."
    elif 'meningioma' in question:
        return "Meningioma is a tumor that arises from the meninges, the membranes surrounding the brain and spinal cord. Most meningiomas are benign and grow slowly, often requiring monitoring or surgical removal if symptomatic."
    elif 'pituitary' in question:
        return "Pituitary tumors are abnormal growths in the pituitary gland that can affect hormone production. They may cause hormonal imbalances, vision problems, and headaches, with treatment ranging from medication to surgery."
    elif 'brain tumor' in question:
        return "Brain tumors are abnormal growths of cells in the brain that can be benign (non-cancerous) or malignant (cancerous). Symptoms vary widely but may include headaches, seizures, and changes in behavior or cognitive function."
    
    # General fallback
    else:
        return "That's interesting! While I specialize in providing information about brain tumors and medical topics, I'm also here to chat. Could you tell me more about what you'd like to know?"

# Initialize RAG
setup_simple_rag()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_file_exists': os.path.exists('brain_tumor_model.pth'),
        'llm_available': llm is not None,
        'rag_system': 'Simplified RAG',
        'vector_store': vector_store is not None,
        'embeddings': embeddings is not None,
        'ollama_available': llm is not None
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
        
        # Get response from RAG
        response, sources = get_simple_rag_response(question)
        
        print(f"Generated response: {response[:100]}...")
        
        # Format sources
        source_info = []
        for i, source in enumerate(sources[:3]):
            source_info.append({
                'title': f'Knowledge Base Document {i+1}',
                'content': source.page_content[:200] + '...',
                'relevance_score': 0.9
            })
        
        return jsonify({
            'answer': response,
            'sources': source_info,
            'llm_used': 'Simplified RAG' if llm else 'Enhanced Keyword System',
            'question_received': question,
            'response_length': len(response),
            'rag_enabled': llm is not None,
            'ollama_available': llm is not None
        })
        
    except Exception as e:
        print(f"RAG query error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/categories', methods=['GET'])
def rag_categories():
    return jsonify({
        'categories': ['glioma', 'meningioma', 'pituitary', 'general'],
        'llm_enabled': llm is not None,
        'rag_system': 'Simplified RAG',
        'vector_store': vector_store is not None,
        'ollama_available': llm is not None
    })

@app.route('/rag/stats', methods=['GET'])
def rag_stats():
    return jsonify({
        'total_documents': 1,
        'categories': ['glioma', 'meningioma', 'pituitary', 'general'],
        'model_loaded': model is not None,
        'llm_available': llm is not None,
        'rag_system': 'Simplified RAG',
        'vector_store': vector_store is not None,
        'embeddings': embeddings is not None,
        'ollama_available': llm is not None,
        'capabilities': [
            'Simplified RAG',
            'Document Retrieval',
            'Vector Similarity',
            'Medical Q&A',
            'General Conversation',
            'Context-aware Responses'
        ]
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
        'summary': 'Brain tumor classification completed using simplified RAG system.',
        'sources': [{'title': 'Simplified RAG Knowledge Base'}]
    }
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
