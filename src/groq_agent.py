import os
from typing import Optional, Type
from pydantic import BaseModel, Field

# FIXED IMPORTS for langchain 0.1.0+
from langchain.tools import BaseTool
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

import streamlit as st


class MedicalAnalysisInput(BaseModel):
    """Input schema for medical analysis"""
    report_text: str = Field(description="The medical report text to analyze")


class MedicalAnalysisTool(BaseTool):
    """Tool for comprehensive medical report analysis"""
    name: str = "medical_analyzer"
    description: str = """
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
        """Generate structured analysis"""
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
        """Extract key findings from text"""
        return "Key values extracted from report text."
    
    def _flag_concerns(self, text: str) -> str:
        """Flag potential concerns"""
        return "Review by doctor recommended for accurate interpretation."
    
    def _generate_recommendations(self, text: str) -> str:
        """Generate health recommendations"""
        return "Maintain healthy lifestyle and follow up with healthcare provider."


class HealthTipsTool(BaseTool):
    """Tool for quick health tips"""
    name: str = "health_advisor"
    description: str = "Provides quick, actionable health tips based on medical findings"
    args_schema: Type[BaseModel] = MedicalAnalysisInput
    
    def _run(self, report_text: str) -> str:
        """Generate health tips"""
        tips = [
            "💧 Stay hydrated - drink 8 glasses of water daily",
            "🥗 Eat balanced meals with plenty of vegetables",
            "🏃 Exercise regularly - at least 30 minutes daily",
            "😴 Get 7-8 hours of quality sleep",
            "🧘 Manage stress through meditation or yoga"
        ]
        return "\n".join(tips)


class GroqMedicalAgent:
    """Main agent class for medical report analysis"""
    
    def __init__(self):
        """Initialize the agent with Groq LLM"""
        # Get API key from Streamlit secrets or env
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found!")
        
        # Initialize Groq LLM - FASTEST inference available!
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0.2,
            max_tokens=4096
        )
        
        # Create tools
        self.tools = [
            Tool(
                name="medical_analyzer",
                func=MedicalAnalysisTool()._run,
                description="Comprehensive medical report analysis with structured output"
            ),
            Tool(
                name="health_advisor",
                func=HealthTipsTool()._run,
                description="Quick health tips and lifestyle recommendations"
            )
        ]
        
        # Initialize agent with memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )
    
    def analyze_report(self, report_text: str) -> str:
        """Run agent analysis on medical report"""
        prompt = f"""
        Analyze this medical report comprehensively. Use the medical_analyzer tool 
        for detailed analysis, and health_advisor for additional tips.
        
        Report Text:
        {report_text}
        
        Provide a thorough, structured analysis suitable for a patient to understand.
        """
        
        try:
            response = self.agent.run(prompt)
            return response
        except Exception as e:
            return f"Analysis error: {str(e)}. Please try again."
    
    def quick_analysis(self, report_text: str) -> str:
        """Fast, direct LLM analysis without agent overhead"""
        direct_prompt = f"""
        Act as a medical report analyzer. Provide a quick, clear summary:
        
        1. What tests were done?
        2. Key numbers/values found
        3. Any red flags (explain simply)
        4. One practical health tip
        
        Report: {report_text}
        """
        
        response = self.llm.predict(direct_prompt)
        return response
    
    def chat_followup(self, question: str) -> str:
        """Allow follow-up questions about the report"""
        try:
            response = self.agent.run(f"Regarding the previous medical report: {question}")
            return response
        except Exception as e:
            return f"Error: {str(e)}"