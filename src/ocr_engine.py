import easyocr
import numpy as np
from PIL import Image
import streamlit as st
import re

class OCREngine:
    def __init__(self):
        """Initialize EasyOCR with CPU (no GPU needed)"""
        if 'reader' not in st.session_state:
            with st.spinner("🚀 Loading OCR model (one-time setup)..."):
                st.session_state.reader = easyocr.Reader(
                    ['en'],
                    gpu=False,  # CPU only!
                    model_storage_directory='./ocr_models',
                    download_enabled=True
                )
        self.reader = st.session_state.reader
    
    def extract(self, image: Image.Image) -> str:
        """Extract text from image"""
        try:
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Run OCR with paragraph grouping
            results = self.reader.readtext(
                img_array, 
                detail=0, 
                paragraph=True,
                contrast_ths=0.1  # Better for medical reports
            )
            
            text = '\n'.join(results)
            return self._post_process(text)
            
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return ""
    
    def _post_process(self, text: str) -> str:
        """Clean up OCR output"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix common medical abbreviations
        fixes = {
            r'\bHb\b': 'Hemoglobin',
            r'\bWBC\b': 'White Blood Cells',
            r'\bRBC\b': 'Red Blood Cells',
            r'\bPLT\b': 'Platelets',
            r'\bBP\b': 'Blood Pressure',
            r'\bHR\b': 'Heart Rate',
            r'\bSpO2\b': 'Oxygen Saturation',
            r'\bBMI\b': 'Body Mass Index',
            r'\bBS\b': 'Blood Sugar',
            r'\bHBA1c\b': 'HbA1c'
        }
        for pattern, replacement in fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text.strip()