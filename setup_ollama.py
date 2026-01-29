#!/usr/bin/env python3
"""
Ollama Setup Script for Railway
This script helps set up Ollama in Railway environment
"""

import os
import subprocess
import requests
import time

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    try:
        # Check if Ollama command exists
        result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
        else:
            print("❌ Ollama is not installed")
            return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def start_ollama_server():
    """Start Ollama server"""
    try:
        print("🦙 Starting Ollama server...")
        # Start Ollama in background
        subprocess.Popen(['ollama', 'serve'], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        time.sleep(5)
        
        # Check if server is running
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server is running")
                return True
        except:
            pass
        
        print("❌ Ollama server failed to start")
        return False
        
    except Exception as e:
        print(f"❌ Error starting Ollama: {e}")
        return False

def pull_ollama_model(model_name="llama2"):
    """Pull Ollama model"""
    try:
        print(f"📦 Pulling {model_name} model...")
        result = subprocess.run(['ollama', 'pull', model_name], 
                               capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {model_name} model pulled successfully")
            return True
        else:
            print(f"❌ Failed to pull {model_name}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout pulling {model_name}")
        return False
    except Exception as e:
        print(f"❌ Error pulling {model_name}: {e}")
        return False

def test_ollama_model(model_name="llama2"):
    """Test Ollama model"""
    try:
        print(f"🧪 Testing {model_name} model...")
        
        # Test with a simple question
        response = requests.post('http://localhost:11434/api/generate', 
                                json={
                                    "model": model_name,
                                    "prompt": "What is a brain tumor?",
                                    "stream": False
                                }, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {model_name} model working!")
            print(f"📝 Response: {result.get('response', '')[:100]}...")
            return True
        else:
            print(f"❌ {model_name} model test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False

def setup_ollama_complete():
    """Complete Ollama setup"""
    print("🦙 Ollama Setup Starting...")
    print("=" * 50)
    
    # Step 1: Check installation
    if not check_ollama_installation():
        print("❌ Ollama not installed. Please install Ollama first.")
        return False
    
    # Step 2: Start server
    if not start_ollama_server():
        print("❌ Failed to start Ollama server.")
        return False
    
    # Step 3: Pull model
    models_to_pull = ["llama2", "mistral"]
    pulled_models = []
    
    for model in models_to_pull:
        if pull_ollama_model(model):
            pulled_models.append(model)
            # Test the model
            test_ollama_model(model)
        else:
            print(f"⚠️ Failed to pull {model}, continuing with available models...")
    
    if pulled_models:
        print(f"✅ Setup complete! Available models: {', '.join(pulled_models)}")
        return True
    else:
        print("❌ No models successfully pulled")
        return False

if __name__ == "__main__":
    success = setup_ollama_complete()
    
    if success:
        print("\n🎉 Ollama setup completed successfully!")
        print("🦙 You can now use Ollama for RAG responses!")
    else:
        print("\n❌ Ollama setup failed!")
        print("💡 Please check the logs and try again.")
