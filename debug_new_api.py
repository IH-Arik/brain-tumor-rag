import requests
import json

def test_new_api():
    """Test new Hugging Face router API"""
    
    print("🧪 Testing New Hugging Face Router API...")
    print("=" * 50)
    
    # Test new API URL
    API_URL = "https://router.huggingface.co/hf/Qwen/Qwen2.5-1.5B-Instruct"
    headers = {"Authorization": f"Bearer hf_nJjFqLmEYsWqXvZyKtRmHpNqUeVbXpLmN"}
    
    # Test general question
    prompt = """You are a helpful AI assistant.

Question: What is artificial intelligence?

Answer:"""
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 150,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "return_full_text": False
        }
    }
    
    try:
        print("📡 Sending request to new API...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=15)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ New API Working!")
            print("📄 Response:")
            print(json.dumps(result, indent=2))
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                if generated_text and len(generated_text.strip()) > 20:
                    clean_response = generated_text.replace(prompt, '').strip()
                    print(f"\n🤖 Clean Response: {clean_response}")
                    return True
                else:
                    print(f"❌ Empty response: {generated_text}")
            else:
                print(f"❌ Unexpected format: {result}")
        else:
            print(f"❌ API Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    return False

def test_simple_models():
    """Test simpler models that might work"""
    
    print("\n🔄 Testing Simple Models...")
    print("=" * 50)
    
    # Simple models that might work with new API
    models = [
        "hf/Qwen/Qwen2.5-1.5B-Instruct",
        "hf/microsoft/DialoGPT-medium",
        "hf/google/flan-t5-base"
    ]
    
    for model in models:
        print(f"\n🤖 Testing: {model}")
        
        API_URL = f"https://router.huggingface.co/{model}"
        headers = {"Authorization": f"Bearer hf_nJjFqLmEYsWqXvZyKtRmHpNqUeVbXpLmN"}
        
        payload = {"inputs": "What is AI?", "parameters": {"max_length": 50}}
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {model}: Working!")
                print(f"📝 Response: {str(result)[:100]}...")
                return model
            else:
                print(f"❌ {model}: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {model}: Error - {e}")
    
    return None

if __name__ == "__main__":
    # Test new API format
    new_api_works = test_new_api()
    
    if not new_api_works:
        print("\n🚨 New API not working, testing simple models...")
        working_model = test_simple_models()
        
        if working_model:
            print(f"\n✅ Found working model: {working_model}")
        else:
            print("\n❌ No working models found - need alternative solution")
    else:
        print("\n✅ New API working!")
