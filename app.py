import streamlit as st
from PIL import Image
import os
import io
import base64
import time

# Page config - MUST be first
st.set_page_config(
    page_title="⚡ Medical Agent AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules
from src.ocr_engine import OCREngine
from src.groq_agent import GroqMedicalAgent

# Custom CSS for modern UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .speed-badge {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 8px 20px;
        border-radius: 25px;
        font-weight: 600;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 30px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    .report-container {
        background: #f8f9fa;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e9ecef;
    }
    
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    .user-message {
        background: #667eea;
        color: white;
        margin-left: 20%;
    }
    
    .ai-message {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        margin-right: 20%;
    }
    </style>
""", unsafe_allow_html=True)

# Session state
def init_state():
    defaults = {
        'ocr': None,
        'agent': None,
        'extracted_text': None,
        'analysis': None,
        'chat_history': []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def main():
    init_state()
    
    # Header
    st.markdown('<h1 class="main-header">⚡ Medical Agent AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align: center;"><span class="speed-badge">🚀 Powered by Groq • 800+ tokens/sec • Agent AI</span></div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔐 API Configuration")
        
        # Groq API Key
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.secrets.get("GROQ_API_KEY", ""),
            help="Get free key at console.groq.com"
        )
        
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
            st.success("✅ Groq API connected!")
        
        st.markdown("---")
        st.header("📊 Stats")
        
        if st.session_state.extracted_text:
            st.metric("Characters Extracted", len(st.session_state.extracted_text))
            st.metric("Analysis Speed", "~2 seconds")
        
        st.markdown("---")
        st.info("""
        **🔒 Privacy First**
        - No data stored permanently
        - Processing happens in real-time
        - HIPAA-aware AI handling
        """)
        
        st.markdown("---")
        st.caption("Built with LangChain + Groq + Streamlit")
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ Please add your Groq API key in the sidebar")
        st.markdown("""
        ### How to get FREE Groq API Key:
        1. Visit [console.groq.com](https://console.groq.com)
        2. Sign up with email/GitHub
        3. Create API key (free tier includes generous credits)
        4. Paste it in the sidebar
        """)
        st.stop()
    
    # Initialize engines
    if not st.session_state.ocr:
        st.session_state.ocr = OCREngine()
    
    if not st.session_state.agent:
        try:
            st.session_state.agent = GroqMedicalAgent()
        except Exception as e:
            st.error(f"Failed to initialize AI Agent: {str(e)}")
            st.stop()
    
    # Main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Medical Report")
        
        uploaded = st.file_uploader(
            "Drop image here (JPG, PNG, JPEG)",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="For best results, use clear, well-lit images"
        )
        
        if uploaded:
            image = Image.open(uploaded)
            
            # Smart resize for display
            display_img = image.copy()
            if display_img.width > 600:
                ratio = 600 / display_img.width
                display_img = display_img.resize(
                    (600, int(display_img.height * ratio)), 
                    Image.Resampling.LANCZOS
                )
            
            st.image(display_img, caption="📸 Uploaded Report", use_column_width=True)
            
            # Action buttons
            c1, c2 = st.columns(2)
            
            with c1:
                if st.button("🔍 Full Agent Analysis", use_container_width=True):
                    process_image(image, mode="full")
            
            with c2:
                if st.button("⚡ Quick Analysis", use_container_width=True):
                    process_image(image, mode="quick")
    
    with col2:
        if st.session_state.analysis:
            display_results()
        else:
            display_welcome()

def process_image(image, mode="full"):
    """Process image with OCR and AI"""
    
    # Step 1: OCR
    start_time = time.time()
    
    with st.spinner("🔤 Extracting text with EasyOCR..."):
        st.session_state.extracted_text = st.session_state.ocr.extract(image)
    
    if not st.session_state.extracted_text:
        st.error("❌ Could not read text. Try a clearer image.")
        return
    
    ocr_time = time.time() - start_time
    
    # Step 2: AI Analysis
    with st.spinner(f"🤖 Running {'Agent' if mode == 'full' else 'Quick'} Analysis on Groq..."):
        if mode == "full":
            st.session_state.analysis = st.session_state.agent.analyze_report(
                st.session_state.extracted_text
            )
        else:
            st.session_state.analysis = st.session_state.agent.quick_analysis(
                st.session_state.extracted_text
            )
    
    total_time = time.time() - start_time
    
    st.success(f"✅ Done in {total_time:.1f}s (OCR: {ocr_time:.1f}s, AI: {total_time-ocr_time:.1f}s)")
    st.rerun()

def display_results():
    """Display analysis results"""
    st.subheader("📋 AI Analysis Results")
    
    # Tabs for different views
    tabs = st.tabs(["🧠 Agent Analysis", "💬 Chat", "📝 Raw Data"])
    
    with tabs[0]:
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.markdown(st.session_state.analysis)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action buttons
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("💾 Save Analysis"):
                st.download_button(
                    "Download Report",
                    st.session_state.analysis,
                    file_name="medical_analysis.txt"
                )
        with c2:
            if st.button("🔄 Re-analyze"):
                st.session_state.analysis = None
                st.rerun()
        with c3:
            if st.button("🗑️ Clear"):
                st.session_state.extracted_text = None
                st.session_state.analysis = None
                st.rerun()
    
    with tabs[1]:
        st.markdown("### 💬 Ask Follow-up Questions")
        st.caption("Chat with the AI agent about your report")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            role, content = msg
            css_class = "user-message" if role == "user" else "ai-message"
            st.markdown(f'<div class="chat-message {css_class}">{content}</div>', 
                       unsafe_allow_html=True)
        
        # Input for new question
        question = st.text_input("Your question:", placeholder="e.g., What does high hemoglobin mean?")
        
        if question and st.button("Send", key="chat_send"):
            st.session_state.chat_history.append(("user", question))
            
            with st.spinner("Agent thinking..."):
                response = st.session_state.agent.chat_followup(question)
            
            st.session_state.chat_history.append(("ai", response))
            st.rerun()
    
    with tabs[2]:
        st.text_area("Extracted OCR Text", st.session_state.extracted_text, height=300)
        
        # Show metrics
        st.markdown("### 📊 Processing Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Text Length", len(st.session_state.extracted_text))
        col2.metric("Words", len(st.session_state.extracted_text.split()))
        col3.metric("Model", "Mixtral-8x7B")

def display_welcome():
    """Show welcome screen"""
    st.info("👈 Upload a medical report to get started")
    
    with st.expander("🎯 What This Agent Can Do"):
        st.markdown("""
        ### 🧠 AI Agent Capabilities:
        
        1. **📋 Comprehensive Analysis**
           - Identifies test types (Blood, Urine, Imaging, etc.)
           - Extracts numerical values automatically
           - Flags abnormal results with explanations
        
        2. **💡 Smart Recommendations**
           - Diet suggestions based on results
           - Lifestyle modifications
           - Exercise recommendations
        
        3. **💬 Interactive Chat**
           - Ask follow-up questions
           - Clarify medical terms
           - Get second opinions on specific values
        
        4. **⚡ Ultra-Fast Processing**
           - Groq LPU (Language Processing Unit)
           - 800+ tokens/second inference
           - Sub-second responses
        """)
    
    with st.expander("🔒 Privacy & Safety"):
        st.markdown("""
        - ✅ No data stored on servers
        - ✅ Processing happens in real-time
        - ✅ No training on your data
        - ⚠️ Always consult doctors for medical decisions
        """)

if __name__ == "__main__":
    main()