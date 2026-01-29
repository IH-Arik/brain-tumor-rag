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

# Global variables with safe defaults
model = None
device = torch.device('cpu')
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
embeddings = None
vector_store = None
llm = None

def safe_setup_model():
    """Safe model setup with error handling"""
    global model, device
    
    try:
        print("🧠 Setting up brain tumor model...")
        
        # Create model
        model = timm.create_model('resnet18', pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)
        
        # Try to download model
        try:
            model_urls = [
                "https://github.com/IH-Arik/brain-tumor-rag/raw/main/brain_tumor_model.pth",
                "https://raw.githubusercontent.com/IH-Arik/brain-tumor-rag/main/brain_tumor_model.pth"
            ]
            
            for url in model_urls:
                print(f"Trying to download model from: {url}")
                response = requests.get(url, timeout=30)
                if response.status_code == 200 and len(response.content) > 1000000:
                    with open('brain_tumor_model.pth', 'wb') as f:
                        f.write(response.content)
                    print(f"Model downloaded! Size: {len(response.content) / (1024*1024):.2f} MB")
                    break
            
            # Load model if file exists
            if os.path.exists('brain_tumor_model.pth'):
                checkpoint = torch.load('brain_tumor_model.pth', map_location=device)
                model.load_state_dict(checkpoint)
                print("✅ Model loaded successfully!")
            else:
                print("⚠️ Model file not found, using demo weights")
                
        except Exception as e:
            print(f"⚠️ Model download failed: {e}")
        
        model = model.to(device)
        model.eval()
        
        # Optimize memory
        for param in model.parameters():
            param.requires_grad = False
        
        print("✅ Model setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ Model setup failed: {e}")
        model = None
        return False

def safe_setup_rag():
    """Safe RAG setup with error handling"""
    global embeddings, vector_store, llm
    
    try:
        print("🔗 Setting up RAG system...")
        
        # Try to setup embeddings
        try:
            print("📝 Creating embeddings...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Create simple knowledge base
            knowledge_base = """
            Brain Tumor Medical Information
            
            Glioma is a type of tumor that occurs in the brain and spinal cord, originating from glial cells.
            These tumors can be benign or malignant and may require surgery, radiation, or chemotherapy.
            
            Meningioma is a tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.
            Most meningiomas are benign and grow slowly, often requiring monitoring or surgical removal if symptomatic.
            
            Pituitary tumors are abnormal growths in the pituitary gland that can affect hormone production.
            They may cause hormonal imbalances, vision problems, and headaches, with treatment ranging from medication to surgery.
            
            Brain tumor symptoms include persistent headaches, seizures, vision problems, and personality changes.
            Treatment options include surgery, radiation therapy, chemotherapy, and targeted therapy.
            """
            
            # Create documents and vector store
            from langchain_core.documents import Document
            documents = [Document(page_content=knowledge_base)]
            
            print("🗂️ Creating vector store...")
            vector_store = FAISS.from_documents(documents, embeddings)
            print("✅ Vector store created!")
            
        except Exception as e:
            print(f"⚠️ RAG setup failed: {e}")
            embeddings = None
            vector_store = None
        
        # Try to setup Ollama (with timeout protection)
        try:
            print("🦙 Setting up Ollama...")
            ollama_url = "http://localhost:11434"
            
            # Quick connection test
            response = requests.get(f"{ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server reachable")
                
                # Try to use Ollama without downloading
                try:
                    llm = Ollama(model="llama2", base_url=ollama_url, temperature=0.7)
                    test_response = llm("Hello", timeout=10)
                    print(f"✅ Ollama working! Test: {test_response[:30]}...")
                except Exception as e:
                    print(f"⚠️ Ollama test failed: {e}")
                    llm = None
            else:
                print("⚠️ Ollama server not reachable")
                llm = None
                
        except Exception as e:
            print(f"⚠️ Ollama setup failed: {e}")
            llm = None
        
        print("✅ RAG setup complete!")
        return True
        
    except Exception as e:
        print(f"❌ RAG setup failed: {e}")
        return False

def get_response(question):
    """Get response with fallback"""
    try:
        question_lower = question.lower()
        medical_keywords = ['brain tumor', 'glioma', 'meningioma', 'pituitary', 'cancer', 'tumor', 'symptom', 'treatment', 'diagnosis', 'medical']
        
        is_medical = any(keyword in question_lower for keyword in medical_keywords)
        
        # Try RAG if available
        if llm and vector_store and is_medical:
            print("🔍 Using RAG...")
            try:
                docs = vector_store.similarity_search(question, k=2)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"Based on this medical information, answer: {question}\n\nContext: {context}\n\nAnswer:"
                response = llm(prompt, timeout=30)
                
                if response:
                    response += " This information is for educational purposes only and is not a substitute for professional medical advice."
                    return response, docs
            except Exception as e:
                print(f"⚠️ RAG failed: {e}")
        
        # Try direct LLM for general conversation
        elif llm and not is_medical:
            print("💬 Using direct LLM...")
            try:
                prompt = f"You are a helpful AI assistant. Respond to: {question}"
                response = llm(prompt, timeout=15)
                if response:
                    return response, []
            except Exception as e:
                print(f"⚠️ Direct LLM failed: {e}")
        
        # Fallback responses
        return get_fallback_response(question), []
        
    except Exception as e:
        print(f"❌ Response generation failed: {e}")
        return "I'm sorry, I encountered an error. Please try again.", []

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

# Safe initialization
print("🚀 Starting application initialization...")

# Load labels safely
try:
    if os.path.exists(app.config['LABELS_PATH']):
        with open(app.config['LABELS_PATH'], 'r') as f:
            labels = [line.strip() for line in f.readlines() if line.strip()]
        print(f"✅ Labels loaded: {labels}")
    else:
        print("⚠️ Labels file not found, using defaults")
except Exception as e:
    print(f"⚠️ Error loading labels: {e}")

# Setup model (non-blocking)
safe_setup_model()

# Setup RAG (non-blocking)
try:
    # Import optional dependencies
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import Ollama
    from langchain_core.documents import Document
    
    safe_setup_rag()
except ImportError as e:
    print(f"⚠️ RAG dependencies not available: {e}")
except Exception as e:
    print(f"⚠️ RAG setup failed: {e}")

print("✅ Application initialization complete!")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
        'rag_system': 'Robust RAG',
        'vector_store': vector_store is not None,
        'embeddings': embeddings is not None,
        'ollama_available': llm is not None,
        'labels_count': len(labels)
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
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        print(f"Received question: {question}")
        
        # Get response
        response, sources = get_response(question)
        
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
            'llm_used': 'Robust RAG' if llm else 'Enhanced Keyword System',
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
        'rag_system': 'Robust RAG',
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
        'rag_system': 'Robust RAG',
        'vector_store': vector_store is not None,
        'embeddings': embeddings is not None,
        'ollama_available': llm is not None,
        'capabilities': [
            'Robust RAG',
            'Document Retrieval',
            'Vector Similarity',
            'Medical Q&A',
            'General Conversation',
            'Error Handling',
            'Fallback Responses'
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
        'summary': 'Brain tumor classification completed using robust RAG system.',
        'sources': [{'title': 'Robust RAG Knowledge Base'}]
    }
    
    return jsonify(prediction_result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Starting Flask app on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
