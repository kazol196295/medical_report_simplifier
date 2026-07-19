# Medical Report Simplifier & Clinical Trial Matcher — Implementation Plan

## Overview
Add clinical trial matching to the existing Medical Report Simplifier. The app extracts text from medical report images (OCR), simplifies medical jargon using an LLM, allows users to chat about their report, and automatically matches patients to eligible clinical trials using a free public API and RAG. The agent is built with LangGraph for stateful, complex interactions.

## Tech Stack (All Free)
| Component | Technology | Cost |
|-----------|-----------|------|
| Frontend | Streamlit (Dark theme) | Free |
| OCR | Tesseract (pytesseract) | Free |
| LLM | Groq API (llama-3.3-70b-versatile) | Free tier: 1000 RPD, 100K TPD |
| Orchestration | LangGraph (stateful agent graphs) | Free, MIT |
| RAG | FAISS + HuggingFace (all-MiniLM-L6-v2) | Free |
| External API | ClinicalTrials.gov API v2 | Free, no key required |
| Hosting | Streamlit Community Cloud | Free, 1GB RAM |

## Architecture
```
Medical Report Image
       ↓
OCR (Tesseract)
       ↓
LLM Patient Profiler → Structured JSON Profile
       ↓
{age, gender, diagnosis, stage, biomarkers, medications, conditions}
       ↓
ClinicalTrials.gov API v2 (RECRUITING trials)
       ↓
RAG (FAISS) — index trial eligibility texts, retrieve top-5
       ↓
Eligibility Agent (LLM) — score & rank trials
       ↓
Ranked Trials with explanations
```

## File Structure
```
app.py                          # Main Streamlit UI (4 tabs)
src/
  ├── ocr_tesseract.py          # Image preprocessing + Tesseract OCR (UNCHANGED)
  ├── rag_engine.py             # FAISS vector DB (UNCHANGED)
  ├── patient_profiler.py       # NEW: LLM extracts structured patient profile
  ├── clinical_trials_api.py    # NEW: ClinicalTrials.gov API v2 client
  ├── prompts.py                # NEW: All LLM system prompts and JSON schemas
  ├── tools.py                  # NEW: LangChain tool definitions (3 tools)
  └── langgraph_agent.py        # NEW: LangGraph state graph + node functions
```

## New Files

### 1. src/prompts.py
All prompt templates centralized:
- `EXTRACT_PROFILE_PROMPT` — Forces valid JSON patient profile
- `ELIGIBILITY_AGENT_PROMPT` — Patient profile + trial → eligibility decision
- `MEDICAL_ANALYZER_PROMPT` — Plain language report explanation
- `HEALTH_ADVISOR_PROMPT` — Lifestyle tips

### 2. src/patient_profiler.py
- `extract_patient_profile(ocr_text, llm) -> dict`
- Calls Groq with EXTRACT_PROFILE_PROMPT
- Strips markdown code fences, parses JSON
- Returns: {age, gender, diagnosis, stage, biomarkers, medications, conditions, lab_findings}

### 3. src/clinical_trials_api.py
- `fetch_clinical_trials(diagnosis, biomarkers="", max_results=20) -> list[dict]`
- GET https://clinicaltrials.gov/api/v2/studies
- Params: query.cond, filter.overallStatus=RECRUITING, fields, pageSize
- Returns: [{nct_id, title, status, phase, criteria, conditions, interventions, min_age, max_age, sex}]

### 4. src/tools.py
Three LangChain @tool functions:
- `medical_analyzer(ocr_text: str) -> str` — Explains report
- `health_advisor(ocr_text: str) -> str` — Lifestyle tips
- `clinical_trial_matcher(patient_profile_json: str) -> str` — Full pipeline: parse profile → fetch trials → FAISS index → retrieve top-5 → LLM eligibility scoring → return JSON

### 5. src/langgraph_agent.py
LangGraph state graph replacing deprecated initialize_agent:
- State: AgentState(TypedDict) with messages + ocr_text
- Nodes: agent (call_model), tools (ToolNode)
- Edges: START → agent → conditional(tools|END) → tools → agent
- Uses InMemorySaver for conversation memory via thread_id

## Modified Files

### 6. app.py — MAJOR REWRITE
- 4 tabs: Analysis & Upload, OCR Text, Chat, Clinical Trials
- Clinical Trials tab: extract profile button, editable JSON, find trials button, colored result cards
- Chat tab uses LangGraph app.invoke()
- Updated session state keys
- Updated imports

### 7. requirements.txt — UPGRADE
- langgraph>=1.0.0 (NEW)
- langchain>=0.3.0 (was 0.2.0)
- langchain-core>=0.3.0 (was 0.2.2)
- langchain-groq>=1.0.0 (was 0.1.6)
- requests (NEW)
- Remove langchainhub (unused)

## Deleted Files
- src/groq_agent.py — Replaced by langgraph_agent.py
- src/ocr_engine.py — Dead code, never imported

## Implementation Order
1. requirements.txt — Upgrade dependencies
2. src/prompts.py — All prompt templates
3. src/patient_profiler.py — LLM profile extractor
4. src/clinical_trials_api.py — API client
5. src/tools.py — 3 LangChain tools
6. src/langgraph_agent.py — LangGraph state graph
7. app.py — Full rewrite with 4 tabs
8. Clean up: delete groq_agent.py, ocr_engine.py

## RAG Usage (Two Places)
1. **Chat follow-up** (existing): FAISS indexes OCR text, retrieves relevant chunks for answering questions
2. **Clinical trial matching** (new): FAISS indexes trial eligibility texts, retrieves top-5 most relevant trials for patient profile, then LLM scores eligibility

## Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| LangChain 0.2→0.3 breaking changes | Full rewrite of agent layer planned |
| Streamlit Cloud 1GB RAM | LangGraph adds <5MB; torch is the bottleneck (already handled) |
| Groq 100K tokens/day | Fallback to llama-3.1-8b-instant (500K TPD) |
| ClinicalTrials.gov API down | Graceful error handling with st.error() |
| LLM JSON parsing failures | Strip markdown fences, try/except, retry |
