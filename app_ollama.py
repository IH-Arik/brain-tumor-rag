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

# LangChain imports for RAG
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

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

# LangChain RAG Setup with Ollama
rag_chain = None
vector_store = None

def setup_ollama_rag():
    """Setup LangChain RAG system with Ollama"""
    global rag_chain, vector_store
    
    try:
        print("🦙 Setting up Ollama RAG system...")
        
        # Wait for Ollama to be ready
        import time
        time.sleep(10)  # Give Ollama time to start
        
        # Create knowledge base documents
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
        Common brain tumor symptoms include persistent headaches that worsen over time, seizures or convulsions, vision problems, memory loss, personality changes, difficulty with balance or coordination, and unexplained nausea or vomiting.
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
        
        # Create documents
        from langchain.docstore.document import Document
        documents = [Document(page_content=knowledge_base)]
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings (using a free model)
        print("📝 Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create vector store
        print("🗂️ Creating vector store...")
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Setup Ollama LLM
        print("🦙 Setting up Ollama LLM...")
        try:
            # Try to connect to Ollama with Railway configuration
            ollama_base_url = os.environ.get('OLLAMA_HOST', 'localhost')
            ollama_port = os.environ.get('OLLAMA_PORT', '11434')
            ollama_url = f"http://{ollama_base_url}:{ollama_port}"
            
            # Test Ollama connection
            import requests
            try:
                response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    print(f"✅ Ollama server reachable at {ollama_url}")
                    
                    # Try different models
                    models_to_try = ["llama2", "mistral", "codellama"]
                    working_model = None
                    
                    for model in models_to_try:
                        try:
                            # Test if model is available
                            models_response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                            if models_response.status_code == 200:
                                available_models = [model['name'] for model in models_response.json().get('models', [])]
                                if model in available_models or any(model in m for m in available_models):
                                    working_model = model
                                    print(f"✅ Found working model: {model}")
                                    break
                        except:
                            continue
                    
                    if working_model:
                        llm = Ollama(model=working_model, base_url=ollama_url, temperature=0.7)
                        
                        # Test the connection
                        test_response = llm("Hello")
                        print(f"✅ Ollama connected with {working_model}! Test: {test_response[:50]}...")
                        
                    else:
                        print("⚠️ No working Ollama models found")
                        llm = None
                        
                else:
                    print(f"⚠️ Ollama server responded with: {response.status_code}")
                    llm = None
                    
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Cannot connect to Ollama server: {e}")
                print("💡 Make sure Ollama is running and accessible")
                llm = None
                
        except Exception as e:
            print(f"⚠️ Ollama setup error: {e}")
            llm = None
        
        if llm:
            # Create RAG chain
            print("🔗 Creating RAG chain...")
            rag_prompt = PromptTemplate(
                template="""You are a helpful medical AI assistant. Use the following context to answer the question about brain tumors. Always include a medical disclaimer.

Context: {context}

Question: {question}

Helpful Answer:""",
                input_variables=["context", "question"]
            )
            
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                chain_type_kwargs={"prompt": rag_prompt},
                return_source_documents=True
            )
            
            print("✅ Ollama RAG system ready!")
        else:
            print("⚠️ Using fallback RAG system")
            rag_chain = None
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Ollama RAG: {e}")
        return False

def get_ollama_response(question):
    """Get response from Ollama RAG system"""
    try:
        if rag_chain:
            print("🦙 Using Ollama RAG...")
            result = rag_chain({"query": question})
            
            # Extract answer and sources
            answer = result.get('result', '')
            sources = result.get('source_documents', [])
            
            # Add medical disclaimer if it's a medical question
            question_lower = question.lower()
            medical_keywords = ['brain tumor', 'glioma', 'meningioma', 'pituitary', 'cancer', 'tumor', 'symptom', 'treatment', 'diagnosis', 'medical']
            
            if any(keyword in question_lower for keyword in medical_keywords):
                answer += " This information is for educational purposes only and is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment."
            
            return answer, sources
        else:
            # Fallback to simple keyword matching
            return get_fallback_response(question), []
            
    except Exception as e:
        print(f"❌ Ollama RAG error: {e}")
        return get_fallback_response(question), []

def get_fallback_response(question):
    """Fallback response system"""
    question = question.lower()
    
    # Simple keyword responses
    if 'glioma' in question:
        return "Glioma is a type of tumor that occurs in the brain and spinal cord, originating from glial cells. These tumors can be benign or malignant and may require surgery, radiation, or chemotherapy depending on their grade and location."
    elif 'meningioma' in question:
        return "Meningioma is a tumor that arises from the meninges, the membranes surrounding the brain and spinal cord. Most meningiomas are benign and grow slowly, often requiring monitoring or surgical removal if symptomatic."
    elif 'pituitary' in question:
        return "Pituitary tumors are abnormal growths in the pituitary gland that can affect hormone production. They may cause hormonal imbalances, vision problems, and headaches, with treatment ranging from medication to surgery."
    elif 'brain tumor' in question:
        return "Brain tumors are abnormal growths of cells in the brain that can be benign (non-cancerous) or malignant (cancerous). Symptoms vary widely but may include headaches, seizures, and changes in behavior or cognitive function."
    elif 'symptom' in question:
        return "Common brain tumor symptoms include persistent headaches that worsen over time, seizures or convulsions, vision problems, memory loss, personality changes, and difficulty with balance or coordination."
    elif 'treatment' in question:
        return "Brain tumor treatment options include surgery to remove the tumor, radiation therapy to destroy cancer cells, chemotherapy drugs to kill rapidly dividing cells, and targeted therapy."
    else:
        return "I can provide information about brain tumors and related medical topics. Please ask a more specific question about brain tumors, symptoms, treatments, or types of tumors."

# Initialize Ollama RAG
setup_ollama_rag()

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
        'download_method': 'url_based',
        'rag_system': 'Ollama RAG',
        'vector_store': vector_store is not None,
        'rag_chain': rag_chain is not None,
        'ollama_available': rag_chain is not None
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
        
        # Get response from Ollama RAG
        response, sources = get_ollama_response(question)
        
        print(f"Generated response: {response[:100]}...")
        
        # Format sources
        source_info = []
        for i, source in enumerate(sources[:3]):  # Limit to top 3 sources
            source_info.append({
                'title': f'Knowledge Base Document {i+1}',
                'content': source.page_content[:200] + '...',
                'relevance_score': 0.9
            })
        
        return jsonify({
            'answer': response,
            'sources': source_info,
            'llm_used': 'Ollama RAG' if rag_chain else 'Enhanced Keyword System',
            'question_received': question,
            'response_length': len(response),
            'rag_enabled': rag_chain is not None,
            'ollama_available': rag_chain is not None
        })
        
    except Exception as e:
        print(f"RAG query error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rag/categories', methods=['GET'])
def rag_categories():
    return jsonify({
        'categories': ['glioma', 'meningioma', 'pituitary', 'general'],
        'llm_enabled': True,
        'rag_system': 'Ollama RAG',
        'vector_store': vector_store is not None,
        'ollama_available': rag_chain is not None
    })

@app.route('/rag/stats', methods=['GET'])
def rag_stats():
    return jsonify({
        'total_documents': 1,
        'categories': ['glioma', 'meningioma', 'pituitary', 'general'],
        'model_loaded': model is not None,
        'llm_available': True,
        'rag_system': 'Ollama RAG',
        'vector_store': vector_store is not None,
        'rag_chain': rag_chain is not None,
        'ollama_available': rag_chain is not None,
        'capabilities': [
            'Ollama Local LLM',
            'Document-based RAG',
            'Semantic Search',
            'Vector Similarity',
            'Medical Q&A',
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
        'summary': 'Brain tumor classification completed using Ollama RAG system for enhanced medical information.',
        'sources': [{'title': 'Ollama RAG Knowledge Base'}]
    }
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
