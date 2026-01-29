import torch
import os
import hashlib

def check_model_integrity():
    """Check if downloaded model matches original"""
    
    print("🔍 Model Integrity Check")
    print("=" * 50)
    
    # Check original model
    original_path = 'brain_tumor_model.pth'
    if os.path.exists(original_path):
        print(f"📁 Original model: {original_path}")
        
        # Get file info
        file_size = os.path.getsize(original_path)
        print(f"📏 Size: {file_size / (1024*1024):.2f} MB")
        
        # Calculate checksum
        with open(original_path, 'rb') as f:
            content = f.read()
            checksum = hashlib.md5(content).hexdigest()
        print(f"🔑 MD5: {checksum}")
        
        # Try to load model
        try:
            device = torch.device('cpu')
            model = torch.load(original_path, map_location=device)
            print(f"✅ Model loads successfully")
            print(f"📋 Model keys: {list(model.keys())[:5]}...")
            
            # Check if it's a state dict or full checkpoint
            if 'state_dict' in model:
                print("📋 Format: Full checkpoint")
                state_dict = model['state_dict']
            else:
                print("📋 Format: State dict only")
                state_dict = model
            
            print(f"🧠 Parameters: {len(state_dict)}")
            
            # Check specific layers
            key_layers = ['conv1.weight', 'fc.weight', 'bn1.weight']
            for layer in key_layers:
                if layer in state_dict:
                    tensor = state_dict[layer]
                    print(f"🔍 {layer}: {tensor.shape} | mean: {tensor.mean().item():.6f} | std: {tensor.std().item():.6f}")
                else:
                    print(f"❌ {layer}: Not found")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"❌ Original model not found: {original_path}")
    
    print("\n" + "=" * 50)

def compare_with_railway():
    """Compare local model with Railway version"""
    
    print("🚂 Railway Model Comparison")
    print("=" * 50)
    
    # Download from GitHub URL
    import requests
    
    urls = [
        "https://raw.githubusercontent.com/IH-Arik/brain-tumor-rag/main/brain_tumor_model.pth",
        "https://github.com/IH-Arik/brain-tumor-rag/raw/main/brain_tumor_model.pth"
    ]
    
    for i, url in enumerate(urls, 1):
        print(f"\n🌐 Testing URL {i}: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                content = response.content
                checksum = hashlib.md5(content).hexdigest()
                size = len(content)
                
                print(f"✅ Download successful")
                print(f"📏 Size: {size / (1024*1024):.2f} MB")
                print(f"🔑 MD5: {checksum}")
                
                # Compare with original
                if os.path.exists('brain_tumor_model.pth'):
                    with open('brain_tumor_model.pth', 'rb') as f:
                        original_content = f.read()
                    original_checksum = hashlib.md5(original_content).hexdigest()
                    
                    if checksum == original_checksum:
                        print(f"✅ Identical to original")
                    else:
                        print(f"❌ Different from original")
                        print(f"   Original MD5: {original_checksum}")
                        print(f"   Download MD5: {checksum}")
                
                # Try to load downloaded model
                try:
                    device = torch.device('cpu')
                    model = torch.load(content, map_location=device)
                    print(f"✅ Downloaded model loads successfully")
                    
                    # Check fc layer weights
                    if 'state_dict' in model:
                        state_dict = model['state_dict']
                    else:
                        state_dict = model
                    
                    if 'fc.weight' in state_dict:
                        fc_weights = state_dict['fc.weight']
                        print(f"🧠 FC weights shape: {fc_weights.shape}")
                        print(f"📊 FC weights mean: {fc_weights.mean().item():.6f}")
                        print(f"📊 FC weights std: {fc_weights.std().item():.6f}")
                    
                except Exception as e:
                    print(f"❌ Error loading downloaded model: {e}")
                
            else:
                print(f"❌ Download failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error downloading: {e}")

if __name__ == "__main__":
    check_model_integrity()
    compare_with_railway()
