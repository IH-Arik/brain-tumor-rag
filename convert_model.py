import base64

# Convert model file to base64
def convert_model_to_base64():
    model_path = 'brain_tumor_model.pth'
    
    try:
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # Convert to base64
        base64_data = base64.b64encode(model_data).decode('utf-8')
        
        # Save to file
        with open('model_base64.txt', 'w') as f:
            f.write(base64_data)
        
        print(f"✅ Model converted to base64!")
        print(f"📁 Original size: {len(model_data) / (1024*1024):.2f} MB")
        print(f"📝 Base64 size: {len(base64_data)} characters")
        print(f"💾 Saved to: model_base64.txt")
        print(f"\n📋 Copy the content of model_base64.txt and paste it in Railway Variables")
        
        return base64_data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    convert_model_to_base64()
