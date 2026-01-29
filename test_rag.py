#!/usr/bin/env python3
"""
Test script for RAG system with Hugging Face model
"""

from rag_engine import BrainTumorRAGEngine

def test_rag_system():
    print("🧠 Testing Brain Tumor RAG System with Hugging Face")
    print("=" * 50)
    
    # Initialize RAG engine
    print("Initializing RAG engine...")
    rag_engine = BrainTumorRAGEngine()
    
    # Test questions
    test_questions = [
        "What are glioma tumors?",
        "What are the treatment options for meningioma?",
        "How are pituitary tumors diagnosed?",
        "What is the prognosis for brain tumors?",
        "What are the symptoms of brain tumors?"
    ]
    
    print("\n📝 Testing Questions:")
    print("-" * 30)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        try:
            response = rag_engine.query(question, top_k=2)
            
            print(f"Answer: {response.answer[:200]}...")
            print(f"Confidence: {response.confidence:.3f}")
            print(f"Sources: {len(response.sources)} documents")
            
            for source in response.sources:
                print(f"  - {source['title']} (Score: {source['relevance_score']:.3f})")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 50)
    
    # Test category-specific queries
    print("\n🎯 Testing Category-Specific Queries:")
    print("-" * 40)
    
    category_tests = [
        ("glioma", "What are the treatment options?"),
        ("meningioma", "Are these tumors cancerous?"),
        ("pituitary", "Do these affect hormones?")
    ]
    
    for category, question in category_tests:
        print(f"\nCategory: {category}")
        print(f"Question: {question}")
        
        try:
            response = rag_engine.query_by_category(category, question)
            print(f"Answer: {response.answer[:200]}...")
            print(f"Confidence: {response.confidence:.3f}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 30)
    
    # Show stats
    print("\n📊 Knowledge Base Stats:")
    print("-" * 25)
    stats = rag_engine.get_knowledge_stats()
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Categories: {', '.join(stats['categories'])}")
    print(f"Vector Store Size: {stats['vector_store_stats']['index_size']}")
    print(f"Embedding Model: {stats['vector_store_stats']['embedding_model']}")

if __name__ == "__main__":
    test_rag_system()
