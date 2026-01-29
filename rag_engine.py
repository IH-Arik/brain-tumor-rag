import os
import json
from typing import List, Dict, Optional, Tuple
from knowledge_base import BrainTumorKnowledgeBase
from vector_store import BrainTumorVectorStore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dataclasses import dataclass

@dataclass
class RAGResponse:
    """Data class for RAG response"""
    answer: str
    sources: List[Dict]
    confidence: float
    query: str
    context_used: List[str]

class BrainTumorRAGEngine:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.knowledge_base = BrainTumorKnowledgeBase()
        self.vector_store = BrainTumorVectorStore()
        self.model_name = model_name
        self.generator = None
        self.tokenizer = None
        
        # Initialize vector store with knowledge base
        self._initialize_vector_store()
        
        # Initialize Hugging Face model
        self._load_huggingface_model()
    
    def _load_huggingface_model(self):
        """Load Hugging Face model for text generation"""
        try:
            print(f"Loading Hugging Face model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                max_length=500,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading Hugging Face model: {e}")
            print("Falling back to rule-based responses only")
            self.generator = None
    
    def _initialize_vector_store(self):
        """Initialize the vector store with documents from knowledge base"""
        documents = self.knowledge_base.get_all_documents()
        self.vector_store.add_documents(documents)
    
    def _retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Retrieve relevant documents for the query"""
        return self.vector_store.search(query, top_k)
    
    def _build_context(self, retrieved_docs: List[Tuple[Dict, float]]) -> str:
        """Build context from retrieved documents"""
        context_parts = []
        
        for doc, score in retrieved_docs:
            context_part = f"""
Document: {doc['title']}
Category: {doc['category']}
Content: {doc['content']}
Relevance Score: {score:.3f}
---"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_answer_with_huggingface(self, query: str, context: str) -> str:
        """Generate answer using Hugging Face model"""
        if self.generator is None:
            return self._generate_fallback_answer(query, context)
        
        try:
            prompt = f"""
Medical Context: {context}

Question: {query}

Please provide a helpful medical answer based on the context above. Keep it concise and informative.

Answer:"""
            
            # Generate response
            responses = self.generator(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            answer = responses[0]['generated_text'].replace(prompt, "").strip()
            
            # Add medical disclaimer
            if answer and not answer.lower().startswith("i don't have"):
                answer += "\n\nNote: This information is for educational purposes only and should not replace professional medical advice."
            
            return answer if answer else self._generate_fallback_answer(query, context)
            
        except Exception as e:
            print(f"Hugging Face generation error: {e}")
            return self._generate_fallback_answer(query, context)
    
    def _generate_fallback_answer(self, query: str, context: str) -> str:
        """Generate a fallback answer without OpenAI"""
        # Simple keyword-based answer generation
        query_lower = query.lower()
        
        # Extract relevant information from context
        relevant_info = []
        if "glioma" in query_lower:
            relevant_info.append("Gliomas are tumors that arise from glial cells and are the most common primary brain tumors in adults.")
        if "meningioma" in query_lower:
            relevant_info.append("Meningiomas arise from the meninges and are typically benign tumors.")
        if "pituitary" in query_lower:
            relevant_info.append("Pituitary tumors are usually benign adenomas that can affect hormone production.")
        if "treatment" in query_lower:
            relevant_info.append("Treatment options typically include surgery, radiation therapy, and chemotherapy depending on tumor type and grade.")
        if "diagnosis" in query_lower:
            relevant_info.append("Diagnosis involves MRI imaging, CT scans, and histopathological examination of tissue samples.")
        
        if not relevant_info:
            relevant_info.append("Based on the available medical information, please consult with a healthcare professional for specific medical advice.")
        
        answer = "\n".join(f"• {info}" for info in relevant_info)
        answer += "\n\nNote: This information is for educational purposes only and should not replace professional medical advice."
        
        return answer
    
    def query(self, question: str, top_k: int = 3) -> RAGResponse:
        """Process a query and return RAG response"""
        # Retrieve relevant documents
        retrieved_docs = self._retrieve_relevant_docs(question, top_k)
        
        if not retrieved_docs:
            return RAGResponse(
                answer="I couldn't find relevant information in the knowledge base to answer your question. Please try rephrasing or consult with a healthcare professional.",
                sources=[],
                confidence=0.0,
                query=question,
                context_used=[]
            )
        
        # Build context
        context = self._build_context(retrieved_docs)
        
        # Generate answer
        answer = self._generate_answer_with_huggingface(question, context)
        
        # Calculate confidence based on retrieval scores
        confidence = sum(score for _, score in retrieved_docs) / len(retrieved_docs)
        
        # Prepare sources
        sources = []
        context_used = []
        for doc, score in retrieved_docs:
            sources.append({
                'title': doc['title'],
                'category': doc['category'],
                'relevance_score': score
            })
            context_used.append(doc['content'])
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            query=question,
            context_used=context_used
        )
    
    def query_by_category(self, category: str, question: str) -> RAGResponse:
        """Query within a specific tumor category"""
        # Filter documents by category
        category_docs = self.knowledge_base.get_documents_by_category(category)
        
        if not category_docs:
            return RAGResponse(
                answer=f"No information found for category: {category}",
                sources=[],
                confidence=0.0,
                query=question,
                context_used=[]
            )
        
        # Create temporary vector store with category documents
        temp_vector_store = BrainTumorVectorStore()
        temp_vector_store.add_documents(category_docs)
        
        # Search within category
        results = temp_vector_store.search(question, top_k=3)
        
        if not results:
            return RAGResponse(
                answer=f"No specific information found for your question within {category} category.",
                sources=[],
                confidence=0.0,
                query=question,
                context_used=[]
            )
        
        # Build context and generate answer
        context = self._build_context(results)
        answer = self._generate_answer_with_huggingface(question, context)
        
        confidence = sum(score for _, score in results) / len(results)
        
        sources = []
        context_used = []
        for doc, score in results:
            sources.append({
                'title': doc['title'],
                'category': doc['category'],
                'relevance_score': score
            })
            context_used.append(doc['content'])
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            query=question,
            context_used=context_used
        )
    
    def get_available_categories(self) -> List[str]:
        """Get all available categories in the knowledge base"""
        return list(set(doc['category'] for doc in self.knowledge_base.get_all_documents()))
    
    def get_knowledge_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        return {
            'total_documents': len(self.knowledge_base.get_all_documents()),
            'categories': self.get_available_categories(),
            'vector_store_stats': self.vector_store.get_stats()
        }
