import base64

def split_base64_model():
    """Split base64 model into smaller chunks for Railway Variables"""
    
    try:
        with open('model_base64.txt', 'r') as f:
            base64_data = f.read()
        
        # Split into chunks of 1 million characters each
        chunk_size = 1000000
        chunks = [base64_data[i:i+chunk_size] for i in range(0, len(base64_data), chunk_size)]
        
        print(f"📊 Total chunks: {len(chunks)}")
        print(f"📏 Chunk size: {chunk_size:,} characters")
        
        # Save chunks to separate files
        for i, chunk in enumerate(chunks):
            filename = f'model_chunk_{i+1}.txt'
            with open(filename, 'w') as f:
                f.write(chunk)
            print(f"💾 Saved: {filename} ({len(chunk):,} chars)")
        
        print(f"\n📋 Railway Variables to create:")
        for i in range(len(chunks)):
            print(f"   MODEL_CHUNK_{i+1}")
        
        print(f"\n🔧 Code will automatically combine these chunks!")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    split_base64_model()
