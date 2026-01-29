# Brain Tumor Classification with RAG System

This project enhances the existing brain tumor classification system with a Retrieval-Augmented Generation (RAG) capability, providing medical information and insights alongside image classification.

## Features

### Original Features
- Brain tumor image classification using ResNet18
- Support for 4 tumor types: glioma, meningioma, pituitary, and no tumor
- Web-based interface with Flask

### New RAG Features
- **Medical Knowledge Base**: Comprehensive information about brain tumors
- **Intelligent Q&A**: Ask questions about brain tumors and get AI-powered answers
- **Context-Aware Responses**: Answers based on retrieved medical documents
- **Enhanced Predictions**: Classification results include relevant medical information
- **Multiple API Endpoints**: RESTful APIs for different functionalities

## Architecture

### Core Components

1. **Knowledge Base** (`knowledge_base.py`)
   - Contains 10 comprehensive medical documents
   - Covers glioma, meningioma, pituitary tumors
   - Includes treatment, diagnosis, prognosis information

2. **Vector Store** (`vector_store.py`)
   - FAISS-based vector database
   - Sentence transformer embeddings
   - Efficient similarity search

3. **RAG Engine** (`rag_engine.py`)
   - Orchestrates retrieval and generation
   - OpenAI integration for answer generation
   - Fallback responses without API

4. **Enhanced Flask App** (`app.py`)
   - Original prediction endpoints
   - New RAG endpoints
   - Integrated prediction with medical info

## API Endpoints

### Original Endpoints
- `POST /predict` - Classify brain tumor image
- `GET /` - Web interface

### New RAG Endpoints

#### 1. General Q&A
```http
POST /rag/query
Content-Type: application/json

{
    "question": "What are the treatment options for glioma?",
    "top_k": 3
}
```

#### 2. Category-Specific Q&A
```http
POST /rag/query
Content-Type: application/json

{
    "question": "What are the symptoms?",
    "category": "glioma",
    "top_k": 2
}
```

#### 3. Get Available Categories
```http
GET /rag/categories
```

#### 4. Knowledge Base Statistics
```http
GET /rag/stats
```

#### 5. Enhanced Prediction with RAG
```http
POST /predict_with_rag
Content-Type: multipart/form-data

image: [file]
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API (Optional)
For enhanced responses, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage Examples

### 1. Basic Classification with RAG Info
```python
import requests

# Upload image for classification with medical info
with open('brain_scan.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict_with_rag',
        files={'image': f}
    )

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")
if result['rag_info']:
    print(f"Medical Info: {result['rag_info']['summary']}")
```

### 2. Ask Medical Questions
```python
import requests

# General question
response = requests.post(
    'http://localhost:5000/rag/query',
    json={'question': 'What are the early symptoms of brain tumors?'}
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### 3. Category-Specific Questions
```python
import requests

# Ask about specific tumor type
response = requests.post(
    'http://localhost:5000/rag/query',
    json={
        'question': 'What are the treatment options?',
        'category': 'meningioma'
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
```

## Knowledge Base Content

The system includes comprehensive information about:

### Glioma Tumors
- Overview and classification
- Treatment options (surgery, radiation, chemotherapy)
- Prognostic factors

### Meningioma Tumors
- Benign and atypical types
- Surgical and radiation treatments
- Observation strategies

### Pituitary Tumors
- Functioning vs non-functioning adenomas
- Hormonal implications
- Treatment approaches

### General Information
- Diagnostic methods (MRI, CT, biopsy)
- Prognostic factors and molecular markers
- Postoperative care and recovery
- Emerging therapies and research

## Technical Details

### Vector Database
- **Embedding Model**: all-MiniLM-L6-v2
- **Similarity Metric**: Cosine similarity
- **Index Type**: FAISS IndexFlatIP
- **Storage**: Local file system with persistence

### RAG Pipeline
1. **Query Processing**: Text preprocessing and embedding
2. **Document Retrieval**: Vector similarity search
3. **Context Building**: Relevant document aggregation
4. **Answer Generation**: OpenAI API or fallback responses

### Performance Features
- **Fast Retrieval**: Sub-millisecond vector search
- **Scalable Storage**: Efficient FAISS indexing
- **Fallback Mechanism**: Works without OpenAI API
- **Context Management**: Intelligent document selection

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for enhanced responses
- Default model: `gpt-3.5-turbo`

### Customization
- Add new documents to `knowledge_base.py`
- Modify embedding model in `vector_store.py`
- Adjust response generation in `rag_engine.py`

## Limitations and Considerations

### Medical Disclaimer
- This system provides educational information only
- Not a substitute for professional medical advice
- Always consult healthcare professionals

### Technical Limitations
- Requires internet connection for OpenAI API
- Knowledge base is static (requires code updates for new info)
- Fallback responses are less sophisticated

## Future Enhancements

1. **Dynamic Knowledge Base**: Web scraping for latest research
2. **Multi-modal RAG**: Include image analysis in responses
3. **Citation System**: Proper medical literature references
4. **User Personalization**: Adaptive responses based on user history
5. **Multi-language Support**: Support for different languages

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **OpenAI API Errors**: Check API key and internet connection
3. **Memory Issues**: Reduce `top_k` parameter for large documents
4. **Slow Responses**: Consider using smaller embedding models

### Performance Optimization
- Use GPU for embedding generation if available
- Cache frequent queries
- Implement document chunking for large texts

## Contributing

To extend the system:
1. Add new medical documents to `knowledge_base.py`
2. Update categories and keywords as needed
3. Test new endpoints thoroughly
4. Update documentation

## License

This project is for educational and research purposes. Please ensure compliance with medical information guidelines and API terms of service.
