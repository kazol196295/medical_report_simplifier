# src/prompts.py
# All LLM system prompts and JSON schemas for the Medical Report Simplifier

EXTRACT_PROFILE_PROMPT = """Extract patient info from this medical report. Return ONLY a JSON object, nothing else.

Use this exact format:
{{"age": 45, "gender": "Male", "diagnosis": "condition", "stage": null, "biomarkers": [], "medications": [], "conditions": ["condition1"], "lab_findings": ["test: value (High/Low)"], "summary": "brief summary"}}

Rules:
- age must be integer or null
- gender: Male, Female, or null
- Use empty arrays [] if no data found
- Return complete JSON with opening {{ and closing }}

Report:
{ocr_text}

JSON:"""

ELIGIBILITY_AGENT_PROMPT = """You are a clinical trial eligibility specialist. Evaluate whether the patient is eligible for the given clinical trial.

IMPORTANT: Return ONLY valid JSON. No markdown, no code fences, no explanation. Just the raw JSON array.

Patient Profile:
{patient_profile}

Clinical Trial:
- NCT ID: {nct_id}
- Title: {title}
- Status: {status}
- Phase: {phase}
- Conditions: {conditions}
- Interventions: {interventions}
- Age Range: {min_age} to {max_age}
- Sex: {sex}
- Eligibility Criteria:
{eligibility_criteria}

Evaluate the patient against each inclusion and exclusion criterion. Return a JSON array with one object per trial:

[
  {{
    "nct_id": "{nct_id}",
    "title": "{title}",
    "status": "{status}",
    "phase": "{phase}",
    "eligibility_score": <float 0.0-1.0>,
    "verdict": "<Likely Eligible|Possibly Eligible|Likely Ineligible|Insufficient Data>",
    "matched_criteria": ["<criteria the patient meets>"],
    "unmet_criteria": ["<criteria the patient does NOT meet>"],
    "uncertain_criteria": ["<criteria that cannot be determined from the report>"],
    "explanation": "<2-3 sentence plain-language explanation>"
  }}
]

Scoring guide:
- 1.0 = Patient meets ALL inclusion criteria and NONE of the exclusion criteria
- 0.7-0.9 = Patient meets most inclusion criteria, may miss 1-2 non-critical ones
- 0.4-0.6 = Patient meets some criteria but has gaps or uncertain factors
- 0.1-0.3 = Patient likely does not meet key criteria
- 0.0 = Patient clearly ineligible (e.g., wrong age, wrong gender, wrong condition)

Return the JSON array now:"""

MEDICAL_ANALYZER_PROMPT = """You are a compassionate medical report interpreter. Explain the following medical report in plain, easy-to-understand language.

Structure your response as:

📋 SUMMARY:
(What type of report is this? What was tested?)

🔍 KEY FINDINGS:
(Explain each important value in simple terms. What's normal? What's abnormal?)

⚠️ FLAGS:
(Any values that need attention? Explain what they mean in everyday language.)

💡 WHAT THIS MEANS:
(Overall interpretation — what should the patient understand from this report?)

🏥 NEXT STEPS:
(Practical recommendations — what should the patient do?)

Medical Report Text:
{ocr_text}

Provide your explanation:"""

HEALTH_ADVISOR_PROMPT = """You are a friendly health advisor. Based on the medical report below, provide practical, actionable health tips.

Focus on:
1. 🍎 Diet recommendations based on findings
2. 🏃 Exercise suggestions appropriate for the patient's condition
3. 💊 Medication adherence tips (if medications are mentioned)
4. 🧘 Stress management and lifestyle adjustments
5. 🩺 Follow-up actions to discuss with their doctor

Be specific to the patient's condition — not generic advice.

Medical Report Text:
{ocr_text}

Provide your health tips:"""
