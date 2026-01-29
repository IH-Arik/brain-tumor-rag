import os
import json
from typing import List, Dict
import numpy as np

class BrainTumorKnowledgeBase:
    def __init__(self):
        self.documents = self._load_medical_documents()
    
    def _load_medical_documents(self) -> List[Dict]:
        """Load comprehensive brain tumor medical information"""
        return [
            {
                "id": "glioma_overview",
                "title": "Glioma Tumor Overview",
                "content": """
                Gliomas are tumors that arise from glial cells in the brain or spinal cord. 
                They are the most common type of primary brain tumor in adults. Gliomas are 
                classified into grades I-IV based on their aggressiveness, with grade IV 
                (glioblastoma) being the most malignant. Symptoms include headaches, seizures, 
                cognitive changes, and neurological deficits depending on tumor location.
                """,
                "category": "glioma",
                "keywords": ["glioma", "glial cells", "brain tumor", "primary tumor"]
            },
            {
                "id": "glioma_treatment",
                "title": "Glioma Treatment Options",
                "content": """
                Treatment for gliomas typically involves a combination of surgical resection, 
                radiation therapy, and chemotherapy. The standard of care for high-grade gliomas 
                includes maximal safe surgical removal followed by concurrent temozolomide 
                chemotherapy and radiation therapy. Targeted therapies and immunotherapy are 
                emerging treatment options for specific molecular subtypes.
                """,
                "category": "glioma",
                "keywords": ["glioma treatment", "surgery", "radiation", "chemotherapy", "temozolomide"]
            },
            {
                "id": "meningioma_overview",
                "title": "Meningioma Tumor Overview",
                "content": """
                Meningiomas are tumors that arise from the meninges, the membranes surrounding 
                the brain and spinal cord. They are typically benign (grade I) but can be 
                atypical (grade II) or malignant (grade III). Meningiomas are more common in 
                women and older adults. Symptoms depend on location but may include headaches, 
                seizures, vision problems, and focal neurological deficits.
                """,
                "category": "meningioma",
                "keywords": ["meningioma", "meninges", "benign tumor", "brain membrane"]
            },
            {
                "id": "meningioma_treatment",
                "title": "Meningioma Treatment Options",
                "content": """
                Treatment for meningiomas depends on tumor grade, size, location, and patient 
                factors. Surgical resection is the primary treatment for symptomatic tumors. 
                For grade I tumors, complete surgical removal may be curative. Radiation therapy 
                is used for incompletely resected tumors, recurrent tumors, or when surgery 
                is not feasible. Observation may be appropriate for small, asymptomatic tumors.
                """,
                "category": "meningioma",
                "keywords": ["meningioma treatment", "surgery", "radiation", "observation"]
            },
            {
                "id": "pituitary_overview",
                "title": "Pituitary Tumor Overview",
                "content": """
                Pituitary tumors are abnormal growths that develop in the pituitary gland. 
                Most pituitary tumors are benign adenomas. They can be functioning (producing 
                hormones) or non-functioning. Functioning tumors can cause hormonal imbalances 
                leading to conditions like acromegaly, Cushing's disease, or prolactinoma. 
                Symptoms include headaches, vision problems, and hormonal abnormalities.
                """,
                "category": "pituitary",
                "keywords": ["pituitary tumor", "pituitary gland", "adenoma", "hormone"]
            },
            {
                "id": "pituitary_treatment",
                "title": "Pituitary Tumor Treatment Options",
                "content": """
                Treatment for pituitary tumors includes medication, surgery, and radiation. 
                Medication is often first-line for functioning tumors (e.g., dopamine agonists 
                for prolactinomas). Transsphenoidal surgery is the standard surgical approach. 
                Radiation therapy is used for residual or recurrent tumors. Regular monitoring 
                of hormone levels and tumor size is essential for long-term management.
                """,
                "category": "pituitary",
                "keywords": ["pituitary treatment", "medication", "transsphenoidal surgery", "radiation"]
            },
            {
                "id": "diagnostic_methods",
                "title": "Brain Tumor Diagnostic Methods",
                "content": """
                Brain tumor diagnosis involves multiple imaging modalities and histopathological 
                analysis. MRI with contrast is the primary imaging modality, providing detailed 
                anatomical information. CT scans are useful for detecting calcifications and 
                acute hemorrhage. Advanced techniques include functional MRI, diffusion tensor 
                imaging, and PET scans. Definitive diagnosis requires tissue biopsy and 
                histopathological examination.
                """,
                "category": "diagnosis",
                "keywords": ["diagnosis", "MRI", "CT scan", "biopsy", "histopathology"]
            },
            {
                "id": "prognosis_factors",
                "title": "Brain Tumor Prognostic Factors",
                "content": """
                Prognosis for brain tumor patients depends on multiple factors including tumor 
                type, grade, location, molecular markers, patient age, and performance status. 
                Key molecular markers include IDH mutation, MGMT methylation, 1p/19q 
                co-deletion, and EGFR amplification. Lower grade tumors, younger age, and 
                favorable molecular profiles generally indicate better prognosis. Extent of 
                surgical resection is also a critical prognostic factor.
                """,
                "category": "prognosis",
                "keywords": ["prognosis", "survival", "molecular markers", "IDH", "MGMT"]
            },
            {
                "id": "postoperative_care",
                "title": "Postoperative Care and Recovery",
                "content": """
                Postoperative care following brain tumor surgery involves close neurological 
                monitoring, management of complications, and rehabilitation. Common postoperative 
                issues include edema, seizures, and hormonal deficiencies. Patients may require 
                physical therapy, occupational therapy, and speech therapy depending on 
                tumor location and treatment effects. Regular follow-up imaging is essential 
                to monitor for recurrence.
                """,
                "category": "recovery",
                "keywords": ["recovery", "rehabilitation", "postoperative care", "complications"]
            },
            {
                "id": "emerging_therapies",
                "title": "Emerging Therapies in Brain Tumor Treatment",
                "content": """
                Emerging therapies for brain tumors include immunotherapy (checkpoint inhibitors, 
                CAR T-cells), targeted molecular therapies, tumor-treating fields, and convection 
                enhanced delivery. Precision medicine approaches based on molecular profiling 
                are becoming increasingly important. Clinical trials offer access to novel 
                treatments and should be considered when appropriate. Liquid biopsies and 
                advanced imaging techniques are improving early detection and monitoring.
                """,
                "category": "research",
                "keywords": ["immunotherapy", "targeted therapy", "clinical trials", "precision medicine"]
            }
        ]
    
    def get_documents_by_category(self, category: str) -> List[Dict]:
        """Get all documents for a specific tumor category"""
        return [doc for doc in self.documents if doc["category"] == category]
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple keyword-based search for documents"""
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.documents:
            score = 0
            # Check title match
            if query_lower in doc["title"].lower():
                score += 3
            # Check content match
            score += doc["content"].lower().count(query_lower) * 0.1
            # Check keywords match
            for keyword in doc["keywords"]:
                if query_lower in keyword.lower():
                    score += 2
            # Check category match
            if query_lower in doc["category"].lower():
                score += 1
            
            scored_docs.append((doc, score))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k] if score > 0]
    
    def get_all_documents(self) -> List[Dict]:
        """Return all documents in the knowledge base"""
        return self.documents
