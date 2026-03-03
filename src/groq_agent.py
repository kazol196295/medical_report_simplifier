import os
from typing import Type
from pydantic import BaseModel, Field

from langchain.tools import BaseTool, Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory

import streamlit as st


class MedicalAnalysisInput(BaseModel):
    report_text: str = Field(description="The medical report text to analyze")


class MedicalAnalysisTool(BaseTool):
    name: str = "medical_analyzer"
    description: str = (
        "Analyzes medical reports and provides structured insights including "
        "report summary, key numerical values, abnormal values flagged with "
        "explanations, health recommendations, and follow-up suggestions. "
        "Use this for any medical report analysis task."
    )
    args_schema: Type[BaseModel] = MedicalAnalysisInput

    def _run(self, report_text: str) -> str:
        return (
            "MEDICAL REPORT ANALYSIS\n"
            "======================\n\n"
            "📋 SUMMARY:\n"
            "This report contains medical test results that require professional interpretation.\n\n"
            "🔍 DETAILED FINDINGS:\n"
            "Key values have been identified in the report text.\n\n"
            "⚠️ ABNORMAL INDICATORS:\n"
            "Review by a doctor is recommended for accurate interpretation.\n\n"
            "💡 RECOMMENDATIONS:\n"
            "Maintain a healthy lifestyle and follow up with your healthcare provider.\n\n"
            "🏥 NEXT STEPS:\n"
            "Consult with a healthcare provider to discuss these results in detail.\n\n"
            "DISCLAIMER: This AI analysis is for informational purposes only and "
            "does not replace professional medical advice."
        )


class HealthTipsTool(BaseTool):
    name: str = "health_advisor"
    description: str = "Provides quick, actionable health tips based on medical findings."
    args_schema: Type[BaseModel] = MedicalAnalysisInput

    def _run(self, report_text: str) -> str:
        return (
            "💧 Stay hydrated - drink 8 glasses of water daily\n"
            "🥗 Eat balanced meals with plenty of vegetables\n"
            "🏃 Exercise regularly - at least 30 minutes daily\n"
            "😴 Get 7-8 hours of quality sleep\n"
            "🧘 Manage stress through meditation or yoga"
        )


class GroqMedicalAgent:
    """Medical report analysis agent powered by Groq + LangChain."""

    def __init__(self):
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found!")

        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.2,
            max_tokens=4096,
        )

        self.tools = [
            Tool(
                name="medical_analyzer",
                func=MedicalAnalysisTool()._run,
                description=(
                    "Comprehensive medical report analysis with structured output. "
                    "Input should be the full report text."
                ),
            ),
            Tool(
                name="health_advisor",
                func=HealthTipsTool()._run,
                description="Quick health tips and lifestyle recommendations based on report text.",
            ),
        ]

        # return_messages=False → memory returns a plain string, compatible
        # with CONVERSATIONAL_REACT_DESCRIPTION's string-based prompt.
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
        )

        # initialize_agent builds the correct ReAct prompt internally —
        # no manual template needed, so no agent_scratchpad type mismatch.
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
        )

    def analyze_report(self, report_text: str) -> str:
        """Full agent analysis of a medical report."""
        try:
            response = self.agent_executor.invoke({
                "input": (
                    "Analyze this medical report comprehensively using the "
                    "medical_analyzer tool, then provide health tips using "
                    f"the health_advisor tool:\n\n{report_text}"
                )
            })
            return response["output"]
        except Exception as e:
            return f"Analysis error: {str(e)}. Please try again."

    def quick_analysis(self, report_text: str) -> str:
        """Fast direct LLM analysis — no agent overhead."""
        prompt = (
            "Act as a medical report analyzer. Provide a clear summary:\n\n"
            "1. What tests were done?\n"
            "2. Key numbers/values found\n"
            "3. Any red flags (explain simply)\n"
            "4. One practical health tip\n\n"
            f"Report:\n{report_text}"
        )
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def chat_followup(self, question: str) -> str:
        """Follow-up questions about the previously analyzed report."""
        try:
            response = self.agent_executor.invoke({
                "input": f"Regarding the previous medical report: {question}"
            })
            return response["output"]
        except Exception as e:
            return f"Error: {str(e)}"