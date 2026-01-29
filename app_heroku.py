import os
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from rag_engine import BrainTumorRAGEngine

# Configuration
MODEL_PATH = "brain_tumor_model.pth"
LABELS_PATH = "labels.txt"
IMAGE_SAVE_DIR = "uploads"

# Create uploads directory if it doesn't exist
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Load labels
try:
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
except FileNotFoundError:
    class_names = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build and load model
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))

# Load model weights if available
if os.path.exists(MODEL_PATH):
    try:
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state, strict=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model weights: {e}")
        print("Running with random weights for demonstration.")
else:
    print("Warning: Model file not found. Running with random weights for demonstration.")

model.to(device)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize RAG engine
try:
    rag_engine = BrainTumorRAGEngine()
    print("RAG engine initialized successfully!")
except Exception as e:
    print(f"Warning: RAG engine initialization failed: {e}")
    rag_engine = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_bytes = file.read()
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image file", "detail": str(e)}), 400

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).tolist()
        best_idx = int(torch.argmax(logits, dim=1).item())
        best_prob = probs[best_idx]

    response = {
        "prediction": class_names[best_idx],
        "confidence": round(best_prob * 100, 2),
        "all_probs": {class_names[i]: round(p * 100, 2) for i, p in enumerate(probs)},
        "filename": file.filename
    }
    return jsonify(response)

@app.route("/predict_with_rag", methods=["POST"])
def predict_with_rag():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    img_bytes = file.read()
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Invalid image file", "detail": str(e)}), 400

    # Get prediction
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).tolist()
        best_idx = int(torch.argmax(logits, dim=1).item())
        best_prob = probs[best_idx]

    prediction = class_names[best_idx]
    
    # Get RAG information for the predicted tumor type
    rag_info = None
    if rag_engine and prediction != "no_tumor":
        try:
            rag_response = rag_engine.query_by_category(
                prediction.replace("_tumor", ""), 
                f"What is {prediction.replace('_', ' ')} and what are the key information about it?"
            )
            rag_info = {
                "summary": rag_response.answer[:500] + "..." if len(rag_response.answer) > 500 else rag_response.answer,
                "sources": rag_response.sources[:2],
                "confidence": rag_response.confidence
            }
        except Exception as e:
            print(f"RAG info error: {e}")
            rag_info = {"error": "Could not retrieve additional information"}

    response = {
        "prediction": prediction,
        "confidence": round(best_prob * 100, 2),
        "all_probs": {class_names[i]: round(p * 100, 2) for i, p in enumerate(probs)},
        "filename": file.filename,
        "rag_info": rag_info
    }
    return jsonify(response)

@app.route("/rag/query", methods=["POST"])
def rag_query():
    """RAG query endpoint"""
    if not rag_engine:
        return jsonify({"error": "RAG engine not available"}), 503
        
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    question = data['question']
    category = data.get('category', None)
    top_k = data.get('top_k', 3)
    
    try:
        if category:
            response = rag_engine.query_by_category(category, question)
        else:
            response = rag_engine.query(question, top_k)
        
        return jsonify({
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "query": response.query
        })
    except Exception as e:
        return jsonify({"error": f"RAG processing error: {str(e)}"}), 500

@app.route("/rag/categories", methods=["GET"])
def rag_categories():
    """Get available categories"""
    if not rag_engine:
        return jsonify({"error": "RAG engine not available"}), 503
        
    try:
        categories = rag_engine.get_available_categories()
        return jsonify({"categories": categories})
    except Exception as e:
        return jsonify({"error": f"Error fetching categories: {str(e)}"}), 500

@app.route("/rag/stats", methods=["GET"])
def rag_stats():
    """Get knowledge base statistics"""
    if not rag_engine:
        return jsonify({"error": "RAG engine not available"}), 503
        
    try:
        stats = rag_engine.get_knowledge_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": f"Error fetching stats: {str(e)}"}), 500

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(IMAGE_SAVE_DIR, filename)

@app.route("/health")
def health_check():
    """Health check endpoint for deployment"""
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "rag_available": rag_engine is not None
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
