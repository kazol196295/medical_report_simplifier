# src/langgraph_agent.py
# LangGraph state graph replacing the deprecated initialize_agent approach

import os
import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.tools import medical_analyzer, health_advisor, clinical_trial_matcher


class AgentState(TypedDict):
    """State for the medical report agent graph."""
    messages: Annotated[list, add_messages]
    ocr_text: str


SYSTEM_PROMPT = """You are MedReport AI, a helpful medical report assistant. You can:

1. **Analyze medical reports** — explain findings in plain language
2. **Give health tips** — provide actionable lifestyle recommendations  
3. **Find clinical trials** — match patients to recruiting clinical trials

When the user asks you to:
- "explain", "analyze", "interpret" the report → use the medical_analyzer tool
- "tips", "advice", "health tips" → use the health_advisor tool
- "find trials", "clinical trials", "match trials" → use the clinical_trial_matcher tool
- Anything else → respond conversationally

Always be helpful, accurate, and remind users that AI analysis is not a substitute for professional medical advice."""


def _get_llm_with_tools():
    """Get or create a ChatGroq instance with tools bound, cached in session state."""
    if "agent_llm" not in st.session_state:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found!")
        
        llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=4096,
        )
        tools = [medical_analyzer, health_advisor, clinical_trial_matcher]
        st.session_state.agent_llm = llm.bind_tools(tools)
        st.session_state.agent_tools = tools
    
    return st.session_state.agent_llm


def call_model(state: AgentState):
    """Call the LLM with the current message history."""
    llm = _get_llm_with_tools()
    
    messages = state["messages"]
    
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Check if the last message has tool calls."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


def build_agent():
    """Build and compile the LangGraph agent."""
    tools = [medical_analyzer, health_advisor, clinical_trial_matcher]
    tool_node = ToolNode(tools)
    
    builder = StateGraph(AgentState)
    
    builder.add_node("agent", call_model)
    builder.add_node("tools", tool_node)
    
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    
    return builder.compile()


def chat_with_agent(agent, ocr_text: str, user_message: str, thread_id: str = "default") -> str:
    """
    Send a message to the LangGraph agent and get a response.
    
    Args:
        agent: Compiled LangGraph app
        ocr_text: The extracted OCR text (stored in state)
        user_message: User's message
        thread_id: Conversation thread ID for memory
    
    Returns:
        Agent's response as a string
    """
    from langgraph.checkpoint.memory import InMemorySaver
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "ocr_text": ocr_text,
        },
        config=config,
    )
    
    last_message = result["messages"][-1]
    return last_message.content if hasattr(last_message, "content") else str(last_message)


def run_tool_directly(tool_name: str, input_text: str) -> str:
    """
    Run a specific tool directly without going through the agent.
    Used for the Clinical Trials tab where we call tools explicitly.
    """
    tools_map = {
        "medical_analyzer": medical_analyzer,
        "health_advisor": health_advisor,
        "clinical_trial_matcher": clinical_trial_matcher,
    }
    
    if tool_name not in tools_map:
        return f"Unknown tool: {tool_name}"
    
    try:
        result = tools_map[tool_name].invoke(input_text)
        return result
    except Exception as e:
        return f"Error running {tool_name}: {str(e)}"
