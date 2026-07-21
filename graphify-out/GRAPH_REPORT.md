# Graph Report - .  (2026-07-21)

## Corpus Check
- 21 files · ~361,844 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 199 nodes · 257 edges · 31 communities (12 shown, 19 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 28 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Clinical Trials API
- Main Application
- OCR & Packages
- ChatGPT Image Analysis
- Sample 1 Report Analysis
- Clinical Trial Matching Plan
- RAG Engine
- Patient Profiler
- CBC Report Analysis
- Patient Data Reports
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29

## God Nodes (most connected - your core abstractions)
1. `MedicalRAG` - 14 edges
2. `Medical Report Simplifier` - 14 edges
3. `build_agent()` - 10 edges
4. `clinical_trial_matcher()` - 10 edges
5. `Clinical Trial Matching` - 10 edges
6. `Complete Blood Count Lab Report - Shree Diagnostic Centre` - 10 edges
7. `City General Hospital Lab Report - John Smith` - 10 edges
8. `main()` - 9 edges
9. `extract_patient_profile()` - 9 edges
10. `run_tool_directly()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Medical Report Simplifier` --references--> `langchain`  [INFERRED]
  README.md → requirements.txt
- `Medical Report Simplifier` --references--> `langgraph`  [INFERRED]
  README.md → requirements.txt
- `Medical Report Simplifier` --references--> `streamlit`  [INFERRED]
  README.md → requirements.txt
- `render_chat_tab()` --calls--> `chat_with_agent()`  [EXTRACTED]
  app.py → src/langgraph_agent.py
- `render_trials_tab()` --calls--> `extract_patient_profile()`  [EXTRACTED]
  app.py → src/patient_profiler.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Technology Stack** — readme_tesseract_ocr, readme_llama_3_3_70b, readme_langgraph, readme_faiss, readme_streamlit [EXTRACTED 1.00]
- **Clinical Trial Matching Pipeline** — readme_patient_profile_extraction, readme_clinicaltrials_gov_integration, readme_eligibility_scoring [EXTRACTED 1.00]
- **Medical Report Assistant UI Workflow (Upload to Analysis)** — sample_1_medical_report_assistant_app, sample_1_upload_report_section, sample_1_tesseract_ocr_integration, sample_1_analysis_categories, sample_1_agent_analysis_prompt, sample_2_medical_report_assistant_analyzed, sample_2_ai_analysis_output, sample_2_extracted_text_tab, sample_2_chat_tab, sample_2_report_action_buttons [EXTRACTED 1.00]
- **John Smith Lab Report - Digital and Photo Versions** — sample_image_chatgpt_image_jul_20_2026_09_17_42_am_city_general_hospital_lab_report, sample_image_chatgpt_image_jul_20_2026_09_17_42_am_patient_john_smith, sample_image_chatgpt_image_jul_20_2026_09_17_42_am_cbc_section, sample_image_chatgpt_image_jul_20_2026_09_17_42_am_lipid_panel, sample_image_chatgpt_image_jul_20_2026_09_17_42_am_metabolic_panel, sample_image_chatgpt_image_jul_20_2026_09_17_42_am_thyroid_panel, sample_image_chatgpt_image_jul_20_2026_09_17_42_am_liver_function_test, sample_image_chatgpt_image_jul_20_2026_09_17_42_am_notes_impression, sample_image_qwen_city_general_hospital_lab_report, sample_image_qwen_patient_john_smith, sample_image_qwen_cbc_section, sample_image_qwen_lipid_panel, sample_image_qwen_metabolic_panel, sample_image_qwen_thyroid_panel, sample_image_qwen_liver_function, sample_image_qwen_notes_impression [EXTRACTED 1.00]
- **Shree Diagnostic Centre CBC Report Sections** — sample_image_ahd_0425_pa_0007719_e_reports_250427_2032_e_pdf_page_7_cbc_report, sample_image_ahd_0425_pa_0007719_e_reports_250427_2032_e_pdf_page_7_haemoglobin_result, sample_image_ahd_0425_pa_0007719_e_reports_250427_2032_e_pdf_page_7_rbc_count_result, sample_image_ahd_0425_pa_0007719_e_reports_250427_2032_e_pdf_page_7_wbc_count_result, sample_image_ahd_0425_pa_0007719_e_reports_250427_2032_e_pdf_page_7_haematocrit_result, sample_image_ahd_0425_pa_0007719_e_reports_250427_2032_e_pdf_page_7_differential_count, sample_image_ahd_0425_pa_0007719_e_reports_250427_2032_e_pdf_page_7_platelets_section, sample_image_ahd_0425_pa_0007719_e_reports_250427_2032_e_pdf_page_7_absolute_counts [EXTRACTED 1.00]

## Communities (31 total, 19 thin omitted)

### Community 0 - "Clinical Trials API"
Cohesion: 0.11
Nodes (27): fetch_clinical_trials(), format_trial_for_rag(), Format a trial dict into a text block suitable for FAISS indexing., Search ClinicalTrials.gov for recruiting trials matching the diagnosis., Search trials with a specific query term., _search_trials(), AgentState, build_agent() (+19 more)

### Community 1 - "Main Application"
Cohesion: 0.12
Nodes (21): _get_profile_llm(), init_state(), main(), Get ChatGroq for profile extraction., render_analysis_tab(), render_chat_tab(), render_ocr_tab(), render_results() (+13 more)

### Community 2 - "OCR & Packages"
Cohesion: 0.09
Nodes (23): libtesseract-dev, tesseract-ocr, tesseract-ocr-eng, AI-Powered Analysis, Dark Theme, FAISS, Groq API, Hugging Face embeddings (+15 more)

### Community 3 - "ChatGPT Image Analysis"
Cohesion: 0.14
Nodes (18): ALT Result (42 U/L - High), Complete Blood Count (CBC) Section, City General Hospital, City General Hospital Lab Report - John Smith, Dr. Michael Anderson MD - Medical Director, Free T4 Result (0.9 ng/dL - Low), Fasting Glucose Result (110 mg/dL - High), HDL Cholesterol Result (42 mg/dL - Low) (+10 more)

### Community 4 - "Sample 1 Report Analysis"
Cohesion: 0.14
Nodes (17): Full Agent Analysis / Quick Analysis Prompt, Analysis Categories (Blood Tests, Lab Reports, Radiology, Prescriptions), Medical Report Assistant - Initial State, Tesseract OCR Ready Indicator, Upload Report Section, AI Analysis Output Panel, Chat Tab, Extracted Text Tab (+9 more)

### Community 5 - "Clinical Trial Matching Plan"
Cohesion: 0.14
Nodes (14): Clinical Trial Matching, ClinicalTrials.gov API v2, Eligibility Agent, FAISS, HuggingFace (all-MiniLM-L6-v2), LangChain tool definitions, LangGraph, LangGraph state graph (+6 more)

### Community 6 - "RAG Engine"
Cohesion: 0.18
Nodes (4): MedicalRAG, Retrieve top-k relevant chunks as a single joined string., Index a list of text blocks (e.g. trial descriptions). Returns chunk count., Retrieve top-k relevant chunks as a list of strings.

### Community 7 - "Patient Profiler"
Cohesion: 0.31
Nodes (10): _build_fallback_profile(), clean_json_response(), _extract_json_substring(), extract_patient_profile(), Remove trailing commas before } or ] so json.loads doesn't choke., Call Groq LLM to extract a structured patient profile from OCR text.     Falls, Extract the first valid JSON object/array from text, if any., Strip markdown code fences and whitespace from LLM response. (+2 more)

### Community 8 - "CBC Report Analysis"
Cohesion: 0.20
Nodes (10): Absolute Counts Section, Complete Blood Count Lab Report - Shree Diagnostic Centre, Differential Count Section, Dr. Bhavesh Chauhan MD, Haematocrit PCV/HCT Result (27.20% - Low), Haemoglobin Result (9.10 gm/dl - Low), Platelets Section, Total R.B.C. Count Result (3.19 mill/cmm - Low) (+2 more)

### Community 9 - "Patient Data Reports"
Cohesion: 0.31
Nodes (9): Patient John Smith (Age 45, Male), CBC Section - Photo Version, City General Hospital Lab Report - John Smith (Photo), Lipid Panel - Photo Version, Liver Function - Photo Version, Metabolic Panel - Photo Version, Notes/Impression - Photo Version, Patient John Smith (Age 45, Male) - Photo Version (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.47
Nodes (6): Diagnostic Tests Conducted Section, Dr. Alan Green MD - Cardiology, Patient Emily Johnson (DOB 01/15/1989), Medical Report - Emily Johnson Cardiology, Medical History Section - Hypertension, Family CAD History, Presenting Complaints - Chest Pain, Palpitations, Shortness of Breath

## Knowledge Gaps
- **74 isolated node(s):** `LLaMA 3.3-70B`, `Groq API`, `FAISS`, `Hugging Face embeddings`, `Patient Profile Extraction` (+69 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MedicalRAG` connect `RAG Engine` to `Clinical Trials API`, `Main Application`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `Medical Report Simplifier` connect `OCR & Packages` to `Clinical Trial Matching Plan`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `Medical Report Simplifier` (e.g. with `langchain` and `langgraph`) actually correct?**
  _`Medical Report Simplifier` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `build_agent()` (e.g. with `AgentState` and `call_model()`) actually correct?**
  _`build_agent()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `clinical_trial_matcher()` (e.g. with `build_agent()` and `_get_llm_with_tools()`) actually correct?**
  _`clinical_trial_matcher()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `LLaMA 3.3-70B`, `Groq API`, `FAISS` to the rest of the system?**
  _74 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Clinical Trials API` be split into smaller, more focused modules?**
  _Cohesion score 0.10887096774193548 - nodes in this community are weakly interconnected._