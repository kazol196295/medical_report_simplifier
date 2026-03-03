import pytesseract
from PIL import Image
import streamlit as st
import numpy as np


class TesseractOCR:
    """OCR using Tesseract - faster on CPU than EasyOCR"""
    
    def __init__(self):
        """Initialize Tesseract (no model download needed)"""
        # No initialization needed - Tesseract is ready immediately
        st.info("✅ Tesseract OCR ready!")
    
    def extract_text(self, image: Image.Image) -> str:
        """Extract text from image using Tesseract"""
        try:
            # SPEED UP: Resize large images first
            max_dim = 1000
            if max(image.size) > max_dim:
                ratio = max_dim / max(image.size)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            # Extract text with Tesseract
            text = pytesseract.image_to_string(image)
            
            return text.strip()
            
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return ""
    
    def extract_with_confidence(self, image: Image.Image):
        """Extract text with confidence scores"""
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        text_parts = []
        for i, text in enumerate(data['text']):
            if int(data['conf'][i]) > 60:  # Only keep high confidence
                text_parts.append(text)
        
        return ' '.join(text_parts)