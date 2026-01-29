#!/usr/bin/env python3
"""
Debug script to test app startup without Railway
"""

import sys
import os

print("🔍 Debugging app startup...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

try:
    print("📦 Testing imports...")
    
    # Test basic imports
    import flask
    print(f"✅ Flask: {flask.__version__}")
    
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    
    import timm
    print(f"✅ TIMM: {timm.__version__}")
    
    # Test LangChain imports
    try:
        from langchain.document_loaders import TextLoader
        print("✅ LangChain document_loaders")
    except ImportError as e:
        print(f"❌ LangChain document_loaders: {e}")
    
    try:
        from langchain.text_splitter import CharacterTextSplitter
        print("✅ LangChain text_splitter")
    except ImportError as e:
        print(f"❌ LangChain text_splitter: {e}")
    
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        print("✅ LangChain embeddings")
    except ImportError as e:
        print(f"❌ LangChain embeddings: {e}")
    
    try:
        from langchain.vectorstores import FAISS
        print("✅ LangChain FAISS")
    except ImportError as e:
        print(f"❌ LangChain FAISS: {e}")
    
    try:
        from langchain.chains import RetrievalQA
        print("✅ LangChain RetrievalQA")
    except ImportError as e:
        print(f"❌ LangChain RetrievalQA: {e}")
    
    try:
        from langchain.llms import Ollama
        print("✅ LangChain Ollama")
    except ImportError as e:
        print(f"❌ LangChain Ollama: {e}")
    
    # Test app import
    print("🚀 Testing app import...")
    from app_railway import app
    print("✅ App imported successfully")
    
    # Test health endpoint
    print("🏥 Testing health endpoint...")
    with app.test_client() as client:
        response = client.get('/health')
        print(f"✅ Health endpoint status: {response.status_code}")
        print(f"✅ Health response: {response.get_json()}")
    
    print("🎉 All tests passed!")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
