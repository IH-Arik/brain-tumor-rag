import base64

def create_test_model():
    """Create a smaller test model with first few chunks"""
    
    try:
        with open('model_base64.txt', 'r') as f:
            full_base64 = f.read()
        
        # Take only first 5 million characters (about 3.5 MB)
        test_base64 = full_base64[:5000000]
        
        # Save test chunk
        with open('model_test_chunk.txt', 'w') as f:
            f.write(test_base64)
        
        print(f"✅ Test model chunk created!")
        print(f"📏 Size: {len(test_base64):,} characters")
        print(f"💾 Saved to: model_test_chunk.txt")
        print(f"\n📋 Railway Variable:")
        print(f"   Name: MODEL_TEST_CHUNK")
        print(f"   Type: String")
        print(f"   Value: Copy from model_test_chunk.txt")
        
        return test_base64
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    create_test_model()
