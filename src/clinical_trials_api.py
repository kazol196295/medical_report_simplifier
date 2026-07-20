# src/clinical_trials_api.py
# Client for ClinicalTrials.gov API v2 (free, no key required)

import requests
from typing import Optional

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_clinical_trials(
    diagnosis: str,
    biomarkers: str = "",
    conditions: list = None,
    max_results: int = 20,
    status: str = "RECRUITING"
) -> list[dict]:
    """
    Search ClinicalTrials.gov for recruiting trials matching the diagnosis.
    
    Args:
        diagnosis: Primary condition/disease (e.g. "Breast Cancer")
        biomarkers: Additional search terms (e.g. "HER2 Positive")
        conditions: List of conditions to broaden search
        max_results: Maximum number of trials to return (max 1000)
        status: Trial status filter (default: RECRUITING)
    
    Returns:
        List of dicts with keys: nct_id, title, status, phase, criteria,
        conditions, interventions, min_age, max_age, sex
    """
    # Build search terms - try multiple strategies
    search_terms = [diagnosis]
    if biomarkers:
        search_terms.append(f"{diagnosis} {biomarkers}")
    if conditions:
        for cond in conditions[:3]:  # Try up to 3 additional conditions
            if cond.lower() != diagnosis.lower():
                search_terms.append(cond)

    # Try each search term until we find results
    for term in search_terms:
        trials = _search_trials(term, status, max_results)
        if trials:
            return trials

    # If still no results, try without status filter
    for term in search_terms:
        trials = _search_trials(term, "", max_results)
        if trials:
            return trials

    return []


def _search_trials(query: str, status: str, max_results: int) -> list[dict]:
    """Search trials with a specific query term."""
    params = {
        "query.cond": query,
        "query.term": query,
        "fields": ",".join([
            "protocolSection.identificationModule",
            "protocolSection.statusModule",
            "protocolSection.conditionsModule",
            "protocolSection.armsInterventionsModule",
            "protocolSection.eligibilityModule",
            "protocolSection.designModule",
        ]),
        "pageSize": min(max_results, 50),
        "countTotal": "true",
        "format": "json",
    }

    if status:
        params["filter.overallStatus"] = status

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        return []

    data = response.json()
    studies = data.get("studies", [])

    trials = []
    for study in studies:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        conditions_mod = proto.get("conditionsModule", {})
        eligibility = proto.get("eligibilityModule", {})
        design = proto.get("designModule", {})
        arms = proto.get("armsInterventionsModule", {})

        interventions = []
        for iv in arms.get("interventions", []):
            interventions.append({
                "type": iv.get("type", ""),
                "name": iv.get("name", ""),
                "description": iv.get("description", ""),
            })

        phases = design.get("phases", [])
        phase_str = ", ".join(phases) if phases else "N/A"

        trials.append({
            "nct_id": ident.get("nctId", ""),
            "title": ident.get("briefTitle", ""),
            "status": status_mod.get("overallStatus", ""),
            "phase": phase_str,
            "criteria": eligibility.get("eligibilityCriteria", ""),
            "conditions": conditions_mod.get("conditions", []),
            "keywords": conditions_mod.get("keywords", []),
            "interventions": interventions,
            "min_age": eligibility.get("minimumAge", ""),
            "max_age": eligibility.get("maximumAge", ""),
            "sex": eligibility.get("sex", "ALL"),
            "study_type": design.get("studyType", ""),
        })

    return trials


def format_trial_for_rag(trial: dict) -> str:
    """Format a trial dict into a text block suitable for FAISS indexing."""
    parts = [
        f"NCT ID: {trial['nct_id']}",
        f"Title: {trial['title']}",
        f"Status: {trial['status']}",
        f"Phase: {trial['phase']}",
        f"Conditions: {', '.join(trial['conditions'])}",
        f"Age Range: {trial['min_age']} to {trial['max_age']}",
        f"Sex: {trial['sex']}",
    ]
    
    if trial["interventions"]:
        iv_strs = [f"{iv['name']} ({iv['type']})" for iv in trial["interventions"]]
        parts.append(f"Interventions: {', '.join(iv_strs)}")
    
    if trial["criteria"]:
        parts.append(f"\nEligibility Criteria:\n{trial['criteria']}")
    
    return "\n".join(parts)
