import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import os

def test_model_loading():
    """Test model loading from different paths"""
    print("🧪 Testing Model Loading...")
    print("=" * 50)
    
    # Test paths
    model_paths = [
        'brain_tumor_model.pth',
        './brain_tumor_model.pth',
        'd:/GitHub/Brain tumor/ML Project main/ML Project/brain_tumor_model.pth'
    ]
    
    device = torch.device('cpu')
    model = None
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n📍 Path {i}: {model_path}")
        print(f"   File exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            try:
                print("   Creating model...")
                model = timm.create_model('resnet18', pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 4)
                
                print("   Loading weights...")
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint)
                
                print("   Moving to device...")
                model = model.to(device)
                model.eval()
                
                print("   ✅ Model loaded successfully!")
                return model, model_path
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        else:
            print("   ❌ File not found")
    
    print("\n⚠️ No valid model file found!")
    return None, None

def test_prediction(model, test_image_path=None):
    """Test model prediction"""
    print("\n🧪 Testing Model Prediction...")
    print("=" * 50)
    
    if model is None:
        print("❌ No model available for testing")
        return
    
    # Create a dummy image if no test image provided
    if test_image_path is None or not os.path.exists(test_image_path):
        print("📝 Creating dummy test image...")
        dummy_image = Image.new('RGB', (224, 224), color='red')
        test_image_path = 'dummy_test.jpg'
        dummy_image.save(test_image_path)
        print(f"   Saved: {test_image_path}")
    
    try:
        # Load and preprocess image
        print("📸 Loading test image...")
        image = Image.open(test_image_path).convert('RGB')
        print(f"   Image size: {image.size}")
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        print(f"   Tensor shape: {image_tensor.shape}")
        
        # Predict
        print("🔮 Making prediction...")
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Results
        labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
        
        print("\n📊 Prediction Results:")
        print(f"   Predicted: {labels[predicted.item()]}")
        print(f"   Confidence: {confidence.item() * 100:.2f}%")
        
        print("\n📈 All Probabilities:")
        probs_list = probabilities.squeeze().tolist()
        for i, (label, prob) in enumerate(zip(labels, probs_list)):
            print(f"   {label}: {prob * 100:.2f}%")
        
        # Check if it's random weights
        max_prob = confidence.item()
        if max_prob < 0.35:  # If max confidence is very low, likely random weights
            print(f"\n⚠️ Low confidence ({max_prob:.2f}) - might be random weights!")
        elif max_prob > 0.90:
            print(f"\n✅ High confidence ({max_prob:.2f}) - likely trained model!")
        else:
            print(f"\n🤔 Medium confidence ({max_prob:.2f}) - uncertain")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    finally:
        # Clean up dummy image
        if test_image_path == 'dummy_test.jpg' and os.path.exists(test_image_path):
            os.remove(test_image_path)

def test_with_real_images():
    """Test with real medical images if available"""
    print("\n🧪 Testing with Real Images...")
    print("=" * 50)
    
    # Look for test images in uploads folder
    uploads_folder = 'uploads'
    if os.path.exists(uploads_folder):
        image_files = [f for f in os.listdir(uploads_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if image_files:
            print(f"📁 Found {len(image_files)} test images:")
            for img_file in image_files[:3]:  # Test first 3 images
                img_path = os.path.join(uploads_folder, img_file)
                print(f"\n🖼️ Testing: {img_file}")
                
                # Load model first
                model, model_path = test_model_loading()
                if model:
                    test_prediction(model, img_path)
                else:
                    print("❌ Cannot test without loaded model")
                break  # Test just one image for now
        else:
            print("❌ No test images found in uploads folder")
    else:
        print("❌ Uploads folder not found")

def check_model_file_info():
    """Check model file details"""
    print("\n🔍 Model File Information...")
    print("=" * 50)
    
    model_path = 'brain_tumor_model.pth'
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"📁 File: {model_path}")
        print(f"📏 Size: {file_size / (1024*1024):.2f} MB")
        
        try:
            # Check if it's a valid PyTorch checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"🔑 Keys in checkpoint: {list(checkpoint.keys())}")
            
            if 'state_dict' in checkpoint:
                print("📋 Format: Full checkpoint with state_dict")
                state_dict = checkpoint['state_dict']
            else:
                print("📋 Format: State dict only")
                state_dict = checkpoint
            
            print(f"🧠 Model parameters: {len(state_dict)} layers")
            
            # Check for typical ResNet layers
            resnet_layers = ['conv1.weight', 'layer1.0.conv1.weight', 'fc.weight']
            found_layers = [layer for layer in resnet_layers if layer in state_dict]
            print(f"🔍 Found ResNet layers: {len(found_layers)}/{len(resnet_layers)}")
            
            if len(found_layers) >= 2:
                print("✅ Looks like a valid ResNet model!")
            else:
                print("⚠️ Might not be a standard ResNet model")
                
        except Exception as e:
            print(f"❌ Error reading model file: {e}")
    else:
        print(f"❌ Model file not found: {model_path}")

if __name__ == "__main__":
    print("🧠 Brain Tumor Model Testing Suite")
    print("=" * 60)
    
    # Check model file info
    check_model_file_info()
    
    # Test model loading
    model, model_path = test_model_loading()
    
    # Test prediction
    if model:
        test_prediction(model)
        test_with_real_images()
    
    print("\n" + "=" * 60)
    print("🏁 Testing Complete!")
