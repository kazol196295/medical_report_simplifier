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


def extract_patient_profile(ocr_text: str, llm: ChatGroq) -> dict:
    """
    Call Groq LLM to extract a structured patient profile from OCR text.

    Returns dict with keys: age, gender, diagnosis, stage, biomarkers,
    medications, conditions, lab_findings, summary
    """
    prompt = EXTRACT_PROFILE_PROMPT.format(ocr_text=ocr_text)

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)

        # Skip empty or clearly incomplete responses
        if not raw or len(raw.strip()) < 10:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise ValueError(f"LLM returned empty/incomplete response after {max_retries} attempts: '{raw}'")

        cleaned = clean_json_response(raw)

        try:
            profile = json.loads(_strip_trailing_commas(cleaned))
            break  # Success
        except json.JSONDecodeError:
            # Fallback: try to extract JSON object directly from raw response
            raw_flat = raw.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ').replace('\n', ' ')
            raw_flat = re.sub(r'\s+', ' ', raw_flat).strip()

            extracted = _extract_json_substring(raw_flat)
            if extracted is not None:
                try:
                    profile = json.loads(extracted)
                    break  # Success
                except json.JSONDecodeError:
                    last_error = f"Failed to parse patient profile JSON. Raw response:\n{raw[:800]}"
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    raise ValueError(last_error)
            else:
                # Last resort: try wrapping bare key-value pairs in braces
                if re.match(r'\s*"[^"]+"\s*:', raw_flat):
                    candidate = _strip_trailing_commas('{' + raw_flat + '}')
                    try:
                        profile = json.loads(candidate)
                        break  # Success
                    except (json.JSONDecodeError, ValueError):
                        last_error = f"Failed to parse patient profile JSON. Raw response:\n{raw[:800]}"
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        raise ValueError(last_error)
                else:
                    last_error = f"Failed to parse patient profile JSON. Raw response:\n{raw[:800]}"
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    raise ValueError(last_error)
    else:
        raise ValueError(last_error or f"Failed after {max_retries} attempts")

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
