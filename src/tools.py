# src/tools.py
# LangChain tool definitions for the LangGraph agent

import json
import os
import streamlit as st
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from src.prompts import MEDICAL_ANALYZER_PROMPT, HEALTH_ADVISOR_PROMPT, ELIGIBILITY_AGENT_PROMPT
from src.clinical_trials_api import fetch_clinical_trials, format_trial_for_rag
from src.rag_engine import MedicalRAG


def _get_llm():
    """Get or create a ChatGroq instance cached in session state."""
    if "tool_llm" not in st.session_state:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found!")
        st.session_state.tool_llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=4096,
        )
    return st.session_state.tool_llm


@tool
def medical_analyzer(ocr_text: str) -> str:
    """Analyze a medical report and explain it in plain, easy-to-understand language. Input should be the full OCR text from the medical report."""
    llm = _get_llm()
    prompt = MEDICAL_ANALYZER_PROMPT.format(ocr_text=ocr_text)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


@tool
def health_advisor(ocr_text: str) -> str:
    """Provide practical, actionable health tips and lifestyle recommendations based on a medical report. Input should be the full OCR text from the medical report."""
    llm = _get_llm()
    prompt = HEALTH_ADVISOR_PROMPT.format(ocr_text=ocr_text)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


@tool
def clinical_trial_matcher(patient_profile_json: str) -> str:
    """
    Match a patient to eligible clinical trials based on their profile.
    Input should be a JSON string with patient profile containing: age, gender,
    diagnosis, stage, biomarkers, medications, conditions.
    Returns a JSON array of ranked trials with eligibility scores.
    """
    try:
        profile = json.loads(patient_profile_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid patient profile JSON"})

    diagnosis = profile.get("diagnosis", "")
    if not diagnosis:
        conditions = profile.get("conditions", [])
        diagnosis = conditions[0] if conditions else ""

    if not diagnosis:
        return json.dumps({"error": "No diagnosis found in patient profile"})

    biomarkers = " ".join(profile.get("biomarkers", []))
    conditions = profile.get("conditions", [])

    try:
        trials = fetch_clinical_trials(
            diagnosis=diagnosis,
            biomarkers=biomarkers,
            conditions=conditions,
            max_results=20,
            status="RECRUITING",
        )
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch trials: {str(e)}"})

    if not trials:
        return json.dumps({"trials": [], "message": "No recruiting trials found for this condition."})

    trial_texts = [format_trial_for_rag(t) for t in trials]

    try:
        rag = MedicalRAG()
        num_chunks = rag.create_index(trial_texts)
    except Exception as e:
        # If RAG fails, use first 5 trials directly
        top_trials = trials[:5]
        rag = None

    if rag is not None:
        query_parts = [diagnosis]
        if profile.get("stage"):
            query_parts.append(f"stage {profile['stage']}")
        if biomarkers:
            query_parts.append(biomarkers)
        query = " ".join(query_parts)

        retrieved = rag.retrieve(query, k=5)

        retrieved_nct_ids = set()
        for chunk in retrieved:
            for text in trial_texts:
                if chunk in text:
                    for t in trials:
                        if t["nct_id"] in text:
                            retrieved_nct_ids.add(t["nct_id"])
                    break

        top_trials = [t for t in trials if t["nct_id"] in retrieved_nct_ids]
        if not top_trials:
            top_trials = trials[:5]

    llm = _get_llm()
    all_results = []
    import time

    for trial in top_trials[:3]:  # Reduced from 5 to 3 to save tokens
        try:
            prompt = ELIGIBILITY_AGENT_PROMPT.format(
                patient_profile=json.dumps(profile, indent=2),
                nct_id=trial["nct_id"],
                title=trial["title"],
                status=trial["status"],
                phase=trial["phase"],
                conditions=", ".join(trial["conditions"]),
                interventions=", ".join([iv["name"] for iv in trial["interventions"]]),
                min_age=trial["min_age"],
                max_age=trial["max_age"],
                sex=trial["sex"],
                eligibility_criteria=trial["criteria"][:3000],
            )
            
            # Retry logic for rate limits
            max_retries = 3
            raw = ""
            for attempt in range(max_retries):
                try:
                    response = llm.invoke(prompt)
                    raw = response.content if hasattr(response, "content") else str(response)
                    break
                except Exception as api_err:
                    if "429" in str(api_err) and attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))  # Exponential backoff
                        continue
                    raise

            import re
            cleaned = raw.strip()
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            cleaned = cleaned.strip()

            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict):
                            all_results.append(item)
                elif isinstance(parsed, dict):
                    all_results.append(parsed)
            except json.JSONDecodeError:
                all_results.append({
                    "nct_id": trial["nct_id"],
                    "title": trial["title"],
                    "status": trial["status"],
                    "phase": trial["phase"],
                    "eligibility_score": 0.0,
                    "verdict": "Parse Error",
                    "matched_criteria": [],
                    "unmet_criteria": [],
                    "uncertain_criteria": [],
                    "explanation": f"Could not parse LLM response. Raw: {raw[:200]}",
                })
            
            time.sleep(0.5)  # Small delay between trials
            
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = "429" in error_msg or "rate" in error_msg.lower()
            
            if is_rate_limit:
                # Fallback: basic keyword matching without LLM
                matched = []
                trial_conds = [c.lower() for c in trial.get("conditions", [])]
                patient_conds = [c.lower() for c in profile.get("conditions", [])]
                
                for pc in patient_conds:
                    for tc in trial_conds:
                        if pc in tc or tc in pc:
                            matched.append(f"Condition match: {pc}")
                
                # Check age eligibility
                min_age = trial.get("min_age", "")
                max_age = trial.get("max_age", "")
                patient_age = profile.get("age")
                
                age_match = True
                if patient_age and min_age:
                    try:
                        min_val = int(''.join(filter(str.isdigit, min_age)))
                        if patient_age < min_val:
                            age_match = False
                    except:
                        pass
                if patient_age and max_age:
                    try:
                        max_val = int(''.join(filter(str.isdigit, max_age)))
                        if patient_age > max_val:
                            age_match = False
                    except:
                        pass
                
                score = 0.3 if matched else 0.1
                if age_match:
                    score += 0.2
                if matched:
                    verdict = "Possibly Eligible (Basic Match)"
                else:
                    verdict = "Needs Review (Rate Limited)"
                
                all_results.append({
                    "nct_id": trial.get("nct_id", "Unknown"),
                    "title": trial.get("title", "Unknown"),
                    "status": trial.get("status", ""),
                    "phase": trial.get("phase", ""),
                    "eligibility_score": min(score, 0.5),
                    "verdict": verdict,
                    "matched_criteria": matched,
                    "unmet_criteria": [],
                    "uncertain_criteria": ["Full eligibility analysis requires LLM (rate limit reached)"],
                    "explanation": "Basic matching only. Groq API rate limit reached. Wait for reset or upgrade plan for full analysis.",
                })
            else:
                all_results.append({
                    "nct_id": trial.get("nct_id", "Unknown"),
                    "title": trial.get("title", "Unknown"),
                    "status": trial.get("status", ""),
                    "phase": trial.get("phase", ""),
                    "eligibility_score": 0.0,
                    "verdict": "Error",
                    "matched_criteria": [],
                    "unmet_criteria": [],
                    "uncertain_criteria": [],
                    "explanation": error_msg[:200],
                })

    all_results.sort(key=lambda x: x.get("eligibility_score", 0), reverse=True)

    return json.dumps(all_results, indent=2)
