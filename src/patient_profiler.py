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
    text = text.strip()

    # If the cleaned text doesn't start with a JSON object/array opener,
    # try to extract the JSON substring from the response.
    if text and text[0] not in ('{', '['):
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end > start:
                return text[start:end + 1]
            # Opening brace missing but closing brace exists — try prepending it
            if start == -1 and end != -1:
                candidate = start_char + text[:end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except (json.JSONDecodeError, ValueError):
                    pass
    return text


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
        # Fallback: try to extract JSON object directly from raw response
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start = raw.find(start_char)
            end = raw.rfind(end_char)
            if start != -1 and end > start:
                try:
                    profile = json.loads(raw[start:end + 1])
                    break
                except json.JSONDecodeError:
                    continue
            # Opening brace missing but closing brace exists — try prepending it
            if start == -1 and end != -1:
                try:
                    profile = json.loads(start_char + raw[:end + 1])
                    break
                except json.JSONDecodeError:
                    continue
        else:
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
