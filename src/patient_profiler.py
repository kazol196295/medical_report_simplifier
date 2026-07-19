# src/patient_profiler.py
# Extracts structured patient profile from OCR text using Groq LLM

import json
import re
from langchain_groq import ChatGroq
from src.prompts import EXTRACT_PROFILE_PROMPT


def clean_json_response(text: str) -> str:
    """Strip markdown code fences and whitespace from LLM response."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return text.strip()


def extract_patient_profile(ocr_text: str, llm: ChatGroq) -> dict:
    """
    Call Groq LLM to extract a structured patient profile from OCR text.
    
    Returns dict with keys: age, gender, diagnosis, stage, biomarkers,
    medications, conditions, lab_findings, summary
    """
    prompt = EXTRACT_PROFILE_PROMPT.format(ocr_text=ocr_text)
    
    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    
    cleaned = clean_json_response(raw)
    
    try:
        profile = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(
            f"Failed to parse patient profile JSON. Raw response:\n{raw[:500]}"
        )
    
    required_keys = ["age", "gender", "diagnosis", "conditions"]
    for key in required_keys:
        if key not in profile:
            profile[key] = None if key != "conditions" else []
    
    if profile.get("biomarkers") is None:
        profile["biomarkers"] = []
    if profile.get("medications") is None:
        profile["medications"] = []
    if profile.get("lab_findings") is None:
        profile["lab_findings"] = []
    
    return profile
