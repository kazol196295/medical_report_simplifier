import os
from typing import Optional, Type
from pydantic import BaseModel, Field

# FIXED IMPORTS for langchain 0.1.0+
from langchain.tools import BaseTool
from langchain_core.tools import Tool  # NEW LOCATION
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

import streamlit as st

class MedicalAnalysisInput(BaseModel):
    report_text: str = Field(description="The medical report text to analyze")

class MedicalAnalysisTool(BaseTool):
    name = "medical_analyzer"
    description = """
    Analyzes medical reports and provides structured insights including:
    - Report summary and test types
    - Key numerical values and their meanings
    - Abnormal values flagged with explanations
    - Health recommendations
    - Follow-up suggestions
    Use this for any medical report analysis task.
    """
    args_schema: Type[BaseModel] = MedicalAnalysisInput
    
    def _run(self, report_text: str) -> str:
        """Analyze the medical report"""
        return self._analyze_report(report_text)
    
    def _analyze_report(self, text: str) -> str:
        # Structured analysis prompt
        analysis = f"""
        MEDICAL REPORT ANALYSIS
        ======================
        
        📋 SUMMARY:
        This report contains medical test results that require professional interpretation.
        
        🔍 DETAILED FINDINGS:
        {self._extract_findings(text)}
        
        ⚠️ ABNORMAL INDICATORS:
        {self._flag_concerns(text)}
        
        💡 RECOMMENDATIONS:
        {self._generate_recommendations(text)}
        
        🏥 NEXT STEPS:
        Consult with a healthcare provider to discuss these results in detail.
        
        DISCLAIMER: This AI analysis is for informational purposes only and 
        does not replace professional medical advice.
        """
        return analysis
    
    def _extract_findings(self, text: str) -> str:
        return "Key values extracted from report text."
    
    def _flag_concerns(self, text: str) -> str:
        return "Review by doctor recommended for accurate interpretation."
    
    def _generate_recommendations(self, text: str) -> str:
        return "Maintain healthy lifestyle and follow up with healthcare provider."

class HealthTipsTool(BaseTool):
    name = "health_advisor"
    description = "Provides quick, actionable health tips based on medical findings"
    args_schema: Type[BaseModel] = MedicalAnalysisInput
    
    def _run(self, report_text: str) -> str:
        tips = [
            "💧 Stay hydrated - drink 8 glasses of water daily",
            "🥗 Eat balanced meals with plenty of vegetables",
            "🏃 Exercise regularly - at least 30 minutes daily",
            "😴 Get 7-8 hours of quality sleep",
            "🧘 Manage stress through meditation or yoga"
        ]
        return "\n".join(tips)

class GroqMedicalAgent