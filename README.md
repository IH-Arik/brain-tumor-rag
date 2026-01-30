#  Brain Tumor Classification with RAG System

A comprehensive medical AI system that combines deep learning-based brain tumor classification with Retrieval-Augmented Generation (RAG) for intelligent medical information retrieval.

##  Features

###  Core Classification
- **4-Class Classification**: Glioma, Meningioma, Pituitary, No Tumor
- **Deep Learning Model**: ResNet18-based CNN
- **High Accuracy**: Trained on medical MRI images
- **Web Interface**: Modern, responsive UI

###  RAG Capabilities
- **Medical Knowledge Base**: 10+ comprehensive medical documents
- **Intelligent Q&A**: Ask questions about brain tumors
- **Context-Aware Responses**: AI-powered medical information
- **Source Attribution**: Transparent information sourcing
- **Category-Specific Queries**: Targeted information retrieval

###  Deployment Options
- **Local Deployment**: Run on your machine
- **GitHub Pages**: Frontend deployment
- **Cloud Platforms**: AWS, Azure, GCP ready
- **Docker Support**: Containerized deployment

##  Quick Start

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/brain-tumor-rag.git
cd brain-tumor-rag

# Install dependencies
pip install -r requirements.txt

# Download the model (place in project root)
# [Model file should be named: brain_tumor_model.pth]

# Run the application
python app.py
```

Visit `http://localhost:5000` to access the web interface.

### Test the RAG System
```bash
python test_rag.py
```

##  Requirements

```txt
flask>=2.3.0
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pillow>=10.0.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
faiss-cpu>=1.7.0
transformers>=4.30.0
numpy>=1.24.0
```

##  Architecture

```
brain-tumor-rag/
├── app.py                 # Main Flask application
├── knowledge_base.py      # Medical knowledge repository
├── vector_store.py        # FAISS vector database
├── rag_engine.py         # RAG orchestration
├── test_rag.py           # Testing script
├── requirements.txt       # Dependencies
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── templates/
│   └── index.html       # Web interface
├── static/              # Static assets
├── uploads/             # Image uploads
└── brain_tumor_model.pth # Trained model
```

##  Usage

### 1. Image Classification
1. Upload an MRI image (JPG/PNG)
2. Click "Predict" for basic classification
3. Click "Predict + RAG Info" for classification with medical information

### 2. Medical Q&A
1. Select a tumor category (optional)
2. Ask a question about brain tumors
3. Get AI-powered medical answers with sources

### 3. API Usage

#### Basic Prediction
```python
import requests

with open('mri_scan.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f}
    )
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}%")
```

#### RAG-Enhanced Prediction
```python
with open('mri_scan.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict_with_rag',
        files={'image': f}
    )
    result = response.json()
    print(f"Medical Info: {result['rag_info']['summary']}")
```

#### Medical Q&A
```python
response = requests.post(
    'http://localhost:5000/rag/query',
    json={
        'question': 'What are glioma treatment options?',
        'category': 'glioma'
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
```

##  Knowledge Base Content

The system includes comprehensive information about:

- **Glioma Tumors**: Overview, classification, treatment options
- **Meningioma Tumors**: Benign/atypical types, surgical approaches
- **Pituitary Tumors**: Functioning/non-functioning adenomas
- **Diagnostic Methods**: MRI, CT scans, biopsy procedures
- **Treatment Options**: Surgery, radiation, chemotherapy
- **Prognosis**: Survival rates, prognostic factors
- **Emerging Therapies**: Immunotherapy, targeted treatments

##  Configuration

### Environment Variables
```bash
# Optional: For enhanced responses (currently using Hugging Face)
# OPENAI_API_KEY=your_api_key_here
```

### Model Configuration
- **Default Model**: `microsoft/DialoGPT-medium` (Hugging Face)
- **Vector Store**: FAISS with cosine similarity
- **Embedding Model**: `all-MiniLM-L6-v2`

##  Deployment Options

### 1. GitHub Pages (Frontend Only)
```bash
# Build static files
# Deploy to gh-pages branch
```

### 2. Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

### 3. Heroku
```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

### 4. Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

##  Model Performance

- **Accuracy**: ~95% on test set
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Input Size**: 224x224 RGB images
- **Model Architecture**: ResNet18 (modified)

##  Medical Disclaimer

**Important**: This system is for educational and research purposes only. 
- Not a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals
- Use as supplementary information tool only

##  Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 for Python code
- Add tests for new features
- Update documentation
- Ensure medical information accuracy

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Medical knowledge from reputable sources
- Hugging Face for NLP models
- FAISS for vector similarity search
- Flask for web framework

##  Contact

- **Developer**: Md Ittesaf Hossain
- **Email**: [ittesafarik@gmail.com]


---

**⚡ Ready to deploy? Check out the deployment guide above!**
