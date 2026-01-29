import requests
import json

def test_qwen_api():
    """Test Qwen API directly"""
    
    print("🧪 Testing Qwen API Directly...")
    print("=" * 50)
    
    # Test API
    API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-1.5B-Instruct"
    headers = {"Authorization": f"Bearer hf_nJjFqLmEYsWqXvZyKtRmHpNqUeVbXpLmN"}
    
    # Test general question
    prompt = """You are a helpful AI assistant. Please provide accurate, informative, and engaging answers to any question.

Question: What is artificial intelligence?

Answer:"""
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    try:
        print("📡 Sending request to Qwen API...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response:")
            print(json.dumps(result, indent=2))
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                if generated_text and len(generated_text.strip()) > 20:
                    clean_response = generated_text.replace(prompt, '').strip()
                    print(f"\n🤖 Qwen Response: {clean_response}")
                    return True
                else:
                    print(f"❌ Empty or short response: {generated_text}")
            else:
                print(f"❌ Unexpected response format: {result}")
        else:
            print(f"❌ API Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    return False

def test_alternative_models():
    """Test alternative free models"""
    
    print("\n🔄 Testing Alternative Models...")
    print("=" * 50)
    
    # Alternative models that might work better
    models = [
        "microsoft/DialoGPT-medium",
        "google/flan-t5-base",
        "facebook/blenderbot-400M-distill",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ]
    
    for model in models:
        print(f"\n🤖 Testing: {model}")
        
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer hf_nJjFqLmEYsWqXvZyKtRmHpNqUeVbXpLmN"}
        
        prompt = "What is artificial intelligence?"
        payload = {"inputs": prompt, "parameters": {"max_length": 100}}
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    if generated_text and len(generated_text.strip()) > 20:
                        print(f"✅ {model}: Working!")
                        print(f"📝 Response: {generated_text[:100]}...")
                        return model
                    else:
                        print(f"❌ {model}: Empty response")
                else:
                    print(f"❌ {model}: Bad format")
            else:
                print(f"❌ {model}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {model}: Error - {e}")
    
    return None

if __name__ == "__main__":
    # Test main Qwen model
    qwen_works = test_qwen_api()
    
    if not qwen_works:
        print("\n🚨 Qwen API not working, testing alternatives...")
        working_model = test_alternative_models()
        
        if working_model:
            print(f"\n✅ Found working model: {working_model}")
        else:
            print("\n❌ No working models found")
    else:
        print("\n✅ Qwen API working!")
