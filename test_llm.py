import requests
import json

def test_huggingface_api():
    """Test if Hugging Face API is working"""
    
    # Test 1: FLAN-T5 Model
    print("🧪 Testing FLAN-T5 Model...")
    API_URL = "https://router.huggingface.co/hf/google/flan-t5-base"
    headers = {"Authorization": f"Bearer hf_nJjFqLmEYsWqXvZyKtRmHpNqUeVbXpLmN"}
    
    prompt = "What is glioma? Give a brief medical explanation."
    payload = {"inputs": prompt, "parameters": {"max_length": 100, "temperature": 0.7}}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ FLAN-T5 Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"❌ FLAN-T5 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ FLAN-T5 Exception: {e}")
        return False

def test_dialogpt_model():
    """Test DialoGPT Model"""
    print("\n🧪 Testing DialoGPT Model...")
    API_URL = "https://router.huggingface.co/hf/microsoft/DialoGPT-medium"
    headers = {"Authorization": f"Bearer hf_nJjFqLmEYsWqXvZyKtRmHpNqUeVbXpLmN"}
    
    payload = {"inputs": "What is brain tumor?", "parameters": {"max_length": 100}}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ DialoGPT Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"❌ DialoGPT Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ DialoGPT Exception: {e}")
        return False

def test_local_fallback():
    """Test local keyword responses"""
    print("\n🧪 Testing Local Fallback...")
    
    medical_responses = {
        'glioma': 'Glioma is a type of tumor that occurs in the brain and spinal cord.',
        'brain tumor': 'Brain tumors are masses of abnormal cells in the brain.',
        'symptoms': 'Common brain tumor symptoms include headaches, seizures, vision problems.'
    }
    
    test_questions = [
        "What is glioma?",
        "What are brain tumor symptoms?",
        "Tell me about brain tumors"
    ]
    
    for question in test_questions:
        question_lower = question.lower()
        response = "I can provide general information about brain tumors."
        
        for keyword, answer in medical_responses.items():
            if keyword in question_lower:
                response = answer
                break
        
        print(f"Q: {question}")
        print(f"A: {response}")
        print()
    
    return True

if __name__ == "__main__":
    print("🤖 Testing LLM Integration...")
    print("=" * 50)
    
    # Test Hugging Face APIs
    flan_working = test_huggingface_api()
    dialogpt_working = test_dialogpt_model()
    
    # Test fallback
    fallback_working = test_local_fallback()
    
    print("=" * 50)
    print("📊 Test Results:")
    print(f"FLAN-T5: {'✅ Working' if flan_working else '❌ Not Working'}")
    print(f"DialoGPT: {'✅ Working' if dialogpt_working else '❌ Not Working'}")
    print(f"Local Fallback: {'✅ Working' if fallback_working else '❌ Not Working'}")
    
    if flan_working or dialogpt_working:
        print("\n🎉 LLM Integration is WORKING!")
    else:
        print("\n⚠️ LLM Integration using Local Fallback Only")
