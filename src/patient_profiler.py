# src/patient_profiler.py
# Extracts structured patient profile from OCR text using Groq LLM

import json
import re
import time
from langchain_groq import ChatGroq
from src.prompts import EXTRACT_PROFILE_PROMPT


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before } or ] so json.loads doesn't choke."""
    return re.sub(r',\s*([}\]])', r'\1', text)


def _extract_json_substring(text: str) -> str | None:
    """Extract the first valid JSON object/array from text, if any."""
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            candidate = _strip_trailing_commas(text[start:end + 1])
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass
        # Closing brace exists but opening brace is missing — try prepending it
        if start == -1 and end != -1:
            candidate = _strip_trailing_commas(start_char + text[:end + 1])
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def clean_json_response(text: str) -> str:
    """Strip markdown code fences and whitespace from LLM response."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Try to extract JSON FIRST (before destroying newlines)
    extracted = _extract_json_substring(text)
    if extracted is not None:
        return extracted

    # If no braces found at all, the response might be bare key-value pairs
    if '{' not in text and '}' not in text:
        # Check if it looks like key-value pairs (starts with a quoted key)
        if re.match(r'\s*"[^"]+"\s*:', text):
            candidate = _strip_trailing_commas('{' + text + '}')
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass

    # Last resort: flatten whitespace and try again
    flat = text.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
    flat = flat.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    flat = re.sub(r'\s+', ' ', flat).strip()
    
    extracted = _extract_json_substring(flat)
    if extracted is not None:
        return extracted

    return flat


def _build_fallback_profile(ocr_text: str) -> dict:
    """Build a basic profile by parsing OCR text directly (no LLM)."""
    profile = {
        "age": None,
        "gender": None,
        "diagnosis": None,
        "stage": None,
        "biomarkers": [],
        "medications": [],
        "conditions": [],
        "lab_findings": [],
        "summary": None
    }
    
    text_lower = ocr_text.lower()
    
    # Extract age
    age_match = re.search(r'age[:\s]+(\d{1,3})', ocr_text, re.IGNORECASE)
    if age_match:
        profile["age"] = int(age_match.group(1))
    
    # Extract gender
    if re.search(r'\b(male|man|he|his)\b', text_lower):
        profile["gender"] = "Male"
    elif re.search(r'\b(female|woman|she|her)\b', text_lower):
        profile["gender"] = "Female"
    
    # Extract common conditions from text
    condition_keywords = [
        "hypertension", "diabetes", "hyperlipidemia", "cholesterol",
        "hypothyroidism", "hyperthyroidism", "anemia", "infection",
        "asthma", "copd", "heart disease", "coronary", "arrhythmia",
        "prediabetes", "pre-diabetes", "obesity", "kidney disease"
    ]
    for cond in condition_keywords:
        if cond in text_lower:
            profile["conditions"].append(cond.title())
    
    # Extract abnormal lab findings
    lab_pattern = r'([A-Za-z\s]+)[:\s]+([\d.,]+)\s*(?:mg/dL|ng/dL|mEq/L|U/L|mU/L|g/dL|%)?\s*\(?(High|Low|Normal)\)?'
    for match in re.finditer(lab_pattern, ocr_text, re.IGNORECASE):
        name = match.group(1).strip()
        value = match.group(2)
        status = match.group(3)
        if status and status.lower() != "normal":
            profile["lab_findings"].append(f"{name}: {value} ({status})")
    
    if profile["conditions"]:
        profile["diagnosis"] = profile["conditions"][0]
        profile["summary"] = f"Patient with {', '.join(profile['conditions'][:3])}"
    
    return profile


def extract_patient_profile(ocr_text: str, llm: ChatGroq) -> dict:
    """
    Call Groq LLM to extract a structured patient profile from OCR text.
    Falls back to regex-based extraction if LLM fails.

    Returns dict with keys: age, gender, diagnosis, stage, biomarkers,
    medications, conditions, lab_findings, summary
    """
    prompt = EXTRACT_PROFILE_PROMPT.format(ocr_text=ocr_text)

    max_retries = 3
    last_raw = ""

    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
            last_raw = raw

            # Skip empty or clearly incomplete responses
            if not raw or len(raw.strip()) < 20:
                if attempt < max_retries - 1:
                    time.sleep(1.5)
                continue

            cleaned = clean_json_response(raw)

            try:
                profile = json.loads(_strip_trailing_commas(cleaned))
                if isinstance(profile, dict) and "age" in profile:
                    break
            except json.JSONDecodeError:
                pass

            # Fallback: try raw response
            raw_flat = raw.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ').replace('\n', ' ')
            raw_flat = re.sub(r'\s+', ' ', raw_flat).strip()

            extracted = _extract_json_substring(raw_flat)
            if extracted is not None:
                try:
                    profile = json.loads(extracted)
                    if isinstance(profile, dict):
                        break
                except json.JSONDecodeError:
                    pass

            if attempt < max_retries - 1:
                time.sleep(1.5)

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.5)
            continue
    else:
        # All retries failed - use fallback regex extraction
        profile = _build_fallback_profile(ocr_text)
        if not profile.get("conditions"):
            raise ValueError(
                f"LLM failed after {max_retries} attempts. Last response: {last_raw[:300]}"
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
