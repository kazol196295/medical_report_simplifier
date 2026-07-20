# app.py
import os, warnings
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import streamlit as st
from PIL import Image
import time
import json

st.set_page_config(
    page_title="Medical Report Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.ocr_tesseract import TesseractOCR
from src.patient_profiler import extract_patient_profile
from src.clinical_trials_api import fetch_clinical_trials, format_trial_for_rag
from src.langgraph_agent import build_agent, chat_with_agent, run_tool_directly

# ── Design System CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
  --navy:        #0B1B2B;
  --navy-mid:    #132337;
  --navy-light:  #1C3352;
  --teal:        #00C9A7;
  --teal-dim:    #00A38A;
  --sky:         #38BDF8;
  --amber:       #F59E0B;
  --rose:        #F43F5E;
  --surface:     #162840;
  --surface2:    #1E3452;
  --border:      rgba(0,201,167,0.18);
  --text:        #E2EBF5;
  --text-muted:  #7A9BB5;
  --radius-sm:   8px;
  --radius-md:   14px;
  --radius-lg:   22px;
  --shadow:      0 8px 32px rgba(0,0,0,0.35);
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--text); }
.stMarkdown p, .stMarkdown li, .stMarkdown span,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 { color: var(--text) !important; }
.stMarkdown blockquote {
  border-left: none !important; padding-left: 0 !important;
  margin-left: 0 !important; background: transparent !important;
  color: var(--text) !important; font-style: normal !important;
}
.stMarkdown blockquote p { color: var(--text) !important; font-style: normal !important; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 1.5rem 3rem !important; max-width: 1400px !important; }

section[data-testid="stSidebar"] {
  background: var(--navy) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stTextInput input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius-sm) !important;
}

.stApp { background: var(--navy-mid) !important; }

.logo-block {
  display: flex; align-items: center; gap: 12px;
  padding: 20px 0 28px; border-bottom: 1px solid var(--border); margin-bottom: 24px;
}
.logo-icon {
  width: 48px; height: 48px; border-radius: 12px;
  background: linear-gradient(135deg, var(--teal) 0%, var(--sky) 100%);
  display: flex; align-items: center; justify-content: center;
  font-size: 24px; flex-shrink: 0; box-shadow: 0 4px 16px rgba(0,201,167,0.3);
}
.logo-text { line-height: 1.2; }
.logo-text span:first-child {
  display: block; font-family: 'DM Serif Display', serif;
  font-size: 1.15rem; color: var(--teal);
}
.logo-text span:last-child {
  display: block; font-size: 0.72rem;
  color: var(--text-muted); letter-spacing: 0.06em; text-transform: uppercase;
}

.page-title {
  font-family: 'DM Serif Display', serif;
  font-size: clamp(1.8rem, 4vw, 2.8rem); color: var(--text);
  line-height: 1.15; margin-bottom: 4px;
}
.page-title span { color: var(--teal); }
.page-subtitle {
  font-size: 0.9rem; color: var(--text-muted);
  margin-bottom: 28px; font-weight: 300;
}

.pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 14px; border-radius: 999px; font-size: 0.78rem;
  font-weight: 600; letter-spacing: 0.04em;
}
.pill-teal  { background: rgba(0,201,167,0.12); color: var(--teal); border: 1px solid rgba(0,201,167,0.3); }
.pill-amber { background: rgba(245,158,11,0.12); color: var(--amber); border: 1px solid rgba(245,158,11,0.3); }
.pill-rose  { background: rgba(244,63,94,0.12);  color: var(--rose);  border: 1px solid rgba(244,63,94,0.3);  }
.pill-sky   { background: rgba(56,189,248,0.12); color: var(--sky);   border: 1px solid rgba(56,189,248,0.3);  }

.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-lg); padding: 24px; box-shadow: var(--shadow);
}
.card-header {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--teal); margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}

.stButton > button {
  background: linear-gradient(135deg, var(--teal) 0%, var(--sky) 100%) !important;
  color: var(--navy) !important; border: none !important;
  border-radius: var(--radius-md) !important; font-weight: 700 !important;
  font-size: 0.88rem !important; padding: 12px 20px !important;
  transition: opacity 0.2s, transform 0.15s !important;
  letter-spacing: 0.03em !important; width: 100% !important;
}
.stButton > button:hover {
  opacity: 0.9 !important; transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px rgba(0,201,167,0.35) !important;
}

.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important; border-radius: var(--radius-md) !important;
  padding: 4px !important; gap: 4px !important; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: var(--text-muted) !important;
  border-radius: 10px !important; font-weight: 600 !important;
  font-size: 0.85rem !important; padding: 8px 18px !important; border: none !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--teal), var(--sky)) !important;
  color: var(--navy) !important;
}
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding-top: 20px !important; }

.stTextArea textarea, .stTextInput input {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important; color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 2px rgba(0,201,167,0.2) !important;
}

.chat-wrap { display: flex; flex-direction: column; gap: 12px; padding: 4px 0; }
.bubble {
  max-width: 82%; padding: 12px 16px;
  border-radius: 16px; font-size: 0.88rem; line-height: 1.55;
}
.bubble-user {
  align-self: flex-end;
  background: linear-gradient(135deg, var(--teal), var(--sky));
  color: var(--navy); border-bottom-right-radius: 4px; font-weight: 500;
}
.bubble-ai {
  align-self: flex-start; background: var(--surface2);
  border: 1px solid var(--border); color: var(--text); border-bottom-left-radius: 4px;
}
.bubble-label {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; margin-bottom: 4px; opacity: 0.6;
}

.analysis-box {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 22px 24px;
  font-size: 0.9rem; line-height: 1.7; color: var(--text); white-space: pre-wrap;
}
.ocr-box {
  background: var(--navy); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 18px 20px;
  font-family: 'Courier New', monospace; font-size: 0.82rem; line-height: 1.65;
  color: var(--teal); max-height: 380px; overflow-y: auto; white-space: pre-wrap;
}

.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0; }
.stat-chip {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: var(--radius-sm); padding: 8px 14px;
  font-size: 0.78rem; color: var(--text-muted);
}
.stat-chip strong { color: var(--teal); font-size: 1rem; display: block; }

.rag-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(0,201,167,0.08); border: 1px solid rgba(0,201,167,0.25);
  border-radius: 8px; padding: 6px 12px; font-size: 0.78rem; color: var(--teal);
  margin-bottom: 12px;
}

.trial-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 18px 20px; margin-bottom: 14px;
}
.trial-card-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 10px;
}
.trial-nct {
  font-family: 'Courier New', monospace; font-size: 0.78rem;
  color: var(--sky); font-weight: 600;
}
.trial-title {
  font-size: 0.95rem; font-weight: 600; color: var(--text);
  margin-bottom: 6px;
}
.trial-meta {
  display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;
}
.score-bar {
  height: 8px; border-radius: 4px; background: var(--navy);
  overflow: hidden; margin: 8px 0;
}
.score-fill {
  height: 100%; border-radius: 4px;
  background: linear-gradient(90deg, var(--teal), var(--sky));
}
.criteria-list {
  font-size: 0.82rem; color: var(--text-muted); margin: 6px 0;
  padding-left: 18px;
}
.criteria-list li { margin-bottom: 3px; }

.divider { height: 1px; background: var(--border); margin: 20px 0; }
.sb-label {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--text-muted) !important; margin: 20px 0 8px;
}

.stSuccess { background: rgba(0,201,167,0.1) !important; border: 1px solid rgba(0,201,167,0.3) !important; color: var(--teal) !important; }
.stError   { background: rgba(244,63,94,0.1)  !important; border: 1px solid rgba(244,63,94,0.3)  !important; color: var(--rose)  !important; }
.stWarning { background: rgba(245,158,11,0.1) !important; border: 1px solid rgba(245,158,11,0.3) !important; color: var(--amber) !important; }
.stInfo    { background: rgba(56,189,248,0.08)!important; border: 1px solid rgba(56,189,248,0.25)!important; color: var(--sky)   !important; }
.stSpinner > div { border-top-color: var(--teal) !important; }

[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 2px dashed rgba(0,201,167,0.35) !important;
  border-radius: var(--radius-lg) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--teal) !important; }
[data-testid="stFileUploader"] label { color: var(--text-muted) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--text-muted) !important; }

@media (max-width: 768px) {
  .block-container { padding: 1rem !important; }
  .card { padding: 16px; }
  .bubble { max-width: 95%; }
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        'ocr': None,
        'agent': None,
        'extracted_text': None,
        'analysis': None,
        'chat_history': [],
        'rag_chunks': 0,
        'patient_profile': None,
        'trial_results': None,
        'profile_error': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="logo-block">
          <div class="logo-icon">🩺</div>
          <div class="logo-text">
            <span>MedReport AI</span>
            <span>Clinical Trial Matcher</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="sb-label">API Configuration</p>', unsafe_allow_html=True)
        groq_key = st.text_input(
            "Groq API Key", type="password",
            value=st.secrets.get("GROQ_API_KEY", ""),
            placeholder="gsk_…",
            help="Free key at console.groq.com"
        )
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key
            st.markdown('<span class="pill pill-teal">Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill-amber">Key required</span>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="sb-label">Session Stats</p>', unsafe_allow_html=True)

        if st.session_state.extracted_text:
            chars = len(st.session_state.extracted_text)
            words = len(st.session_state.extracted_text.split())
            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-chip"><strong>{chars:,}</strong>Characters</div>
              <div class="stat-chip"><strong>{words:,}</strong>Words</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:var(--text-muted);font-size:0.82rem;">Upload a report to see stats</p>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="sb-label">RAG Index</p>', unsafe_allow_html=True)
        if st.session_state.rag_chunks > 0:
            st.markdown(f"""
            <div class="rag-badge">
              <strong>{st.session_state.rag_chunks}</strong>&nbsp;chunks indexed
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<span class="pill pill-teal">RAG Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill-amber">Not indexed yet</span>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.7;">
          No data stored permanently<br>
          LangGraph-powered agent<br>
          RAG-powered chat &amp; trials<br>
          Always consult a doctor
        </div>
        """, unsafe_allow_html=True)


# ── Image processing ───────────────────────────────────────────────────────────
def process_image(image, mode="full"):
    start = time.time()

    with st.spinner("Extracting text from image…"):
        st.session_state.extracted_text = st.session_state.ocr.extract_text(image)

    if not st.session_state.extracted_text:
        st.error("Could not read text — try a clearer, well-lit image.")
        return

    ocr_t = time.time() - start

    from src.rag_engine import MedicalRAG
    with st.spinner("Building RAG index…"):
        rag = MedicalRAG()
        n_chunks = rag.index_report(st.session_state.extracted_text)
        st.session_state.rag_chunks = n_chunks

    rag_t = time.time() - start - ocr_t

    label = "Full" if mode == "full" else "Quick"
    with st.spinner(f"Running {label} Analysis via Groq…"):
        if mode == "full":
            result = run_tool_directly("medical_analyzer", st.session_state.extracted_text)
        else:
            result = run_tool_directly("health_advisor", st.session_state.extracted_text)
        st.session_state.analysis = result

    total_t = time.time() - start
    ai_t = total_t - ocr_t - rag_t

    st.success(
        f"Done in {total_t:.1f}s  —  "
        f"OCR {ocr_t:.1f}s · RAG index {rag_t:.1f}s · AI {ai_t:.1f}s"
    )
    st.rerun()


# ── Welcome placeholder ────────────────────────────────────────────────────────
def render_welcome():
    st.markdown("""
    <div class="card" style="text-align:center; padding: 48px 32px;">
      <div style="font-size:3rem; margin-bottom:16px;">📋</div>
      <p style="font-family:'DM Serif Display',serif; font-size:1.25rem; color:var(--text); margin-bottom:8px;">
        No report analysed yet
      </p>
      <p style="color:var(--text-muted); font-size:0.88rem; max-width:320px; margin:0 auto 24px;">
        Upload a medical report image on the left, then tap
        <strong style="color:var(--teal);">Full Analysis</strong> or
        <strong style="color:var(--sky);">Quick Analysis</strong>.
      </p>
      <div style="display:flex; gap:10px; justify-content:center; flex-wrap:wrap;">
        <span class="pill pill-teal">🩸 Blood Tests</span>
        <span class="pill pill-teal">🧪 Lab Reports</span>
        <span class="pill pill-teal">🩻 Radiology</span>
        <span class="pill pill-amber">💊 Prescriptions</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Tab 1 & 2: Analysis & OCR ──────────────────────────────────────────────────
def render_analysis_tab():
    st.markdown('<div class="card-header">AI Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="analysis-box">{st.session_state.analysis}</div>',
        unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download", st.session_state.analysis,
            file_name="medical_analysis.txt", width='stretch'
        )
    with col2:
        if st.button("Re-analyse", width='stretch'):
            st.session_state.analysis = None
            st.rerun()
    with col3:
        if st.button("Clear All", width='stretch'):
            st.session_state.extracted_text = None
            st.session_state.analysis = None
            st.session_state.chat_history = []
            st.session_state.rag_chunks = 0
            st.session_state.patient_profile = None
            st.session_state.trial_results = None
            st.rerun()


def render_ocr_tab():
    st.markdown('<div class="card-header">Extracted OCR Text</div>', unsafe_allow_html=True)

    chars = len(st.session_state.extracted_text)
    words = len(st.session_state.extracted_text.split())
    lines = st.session_state.extracted_text.count('\n') + 1
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-chip"><strong>{chars:,}</strong>Characters</div>
      <div class="stat-chip"><strong>{words:,}</strong>Words</div>
      <div class="stat-chip"><strong>{lines:,}</strong>Lines</div>
      <div class="stat-chip"><strong>{st.session_state.rag_chunks}</strong>RAG Chunks</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="ocr-box">{st.session_state.extracted_text}</div>',
        unsafe_allow_html=True
    )
    st.download_button(
        "Download Raw Text", st.session_state.extracted_text,
        file_name="ocr_extracted.txt", width='stretch'
    )


# ── Tab 3: Chat (LangGraph) ────────────────────────────────────────────────────
def render_chat_tab():
    st.markdown('<div class="card-header">Ask Follow-up Questions</div>', unsafe_allow_html=True)

    if st.session_state.rag_chunks > 0:
        st.markdown(
            f'<div class="rag-badge">RAG active · '
            f'{st.session_state.rag_chunks} chunks · '
            f'answers grounded in your report</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="rag-badge" style="border-color:rgba(245,158,11,0.3);color:var(--amber);">'
            'RAG not active — upload and analyse a report first</div>',
            unsafe_allow_html=True
        )

    if st.session_state.chat_history:
        bubbles_html = '<div class="chat-wrap">'
        for role, content in st.session_state.chat_history:
            if role == "user":
                bubbles_html += (
                    f'<div class="bubble bubble-user">'
                    f'<div class="bubble-label">You</div>{content}</div>'
                )
            else:
                bubbles_html += (
                    f'<div class="bubble bubble-ai">'
                    f'<div class="bubble-label">MedReport AI</div>{content}</div>'
                )
        bubbles_html += '</div>'
        st.markdown(bubbles_html, unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <p style="color:var(--text-muted);font-size:0.85rem; margin-bottom:16px;">
          Try: <em>"What does a high WBC count mean?"</em> or
          <em>"Explain my cholesterol levels."</em> or
          <em>"Find clinical trials for my condition"</em>
        </p>
        """, unsafe_allow_html=True)

    question = st.text_input(
        "Your question",
        placeholder="Ask anything about your report…",
        label_visibility="collapsed"
    )
    if st.button("Send", width='stretch') and question.strip():
        st.session_state.chat_history.append(("user", question))
        with st.spinner("Thinking…"):
            agent = st.session_state.agent
            reply = chat_with_agent(
                agent,
                st.session_state.extracted_text or "",
                question,
                thread_id="main_chat"
            )
        st.session_state.chat_history.append(("ai", reply))
        st.rerun()


# ── Tab 4: Clinical Trials ─────────────────────────────────────────────────────
def render_trials_tab():
    st.markdown('<div class="card-header">Clinical Trial Matcher</div>', unsafe_allow_html=True)

    if not st.session_state.extracted_text:
        st.markdown("""
        <div style="text-align:center; padding:40px 20px; color:var(--text-muted);">
          <div style="font-size:2.5rem; margin-bottom:12px;">🔬</div>
          <p style="font-size:0.88rem;">Upload and analyse a medical report first to find matching clinical trials.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("""
    <p style="color:var(--text-muted);font-size:0.85rem; margin-bottom:16px;">
      Step 1: Extract a structured patient profile from your report, then Step 2: Find matching clinical trials.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Step 1: Extract Patient Profile", width='stretch'):
            with st.spinner("Extracting patient profile via LLM…"):
                try:
                    llm = _get_profile_llm()
                    profile = extract_patient_profile(st.session_state.extracted_text, llm)
                    st.session_state.patient_profile = profile
                    st.session_state.trial_results = None
                    st.session_state.profile_error = None
                except Exception as e:
                    st.session_state.profile_error = str(e)
            st.rerun()

    if st.session_state.profile_error:
        st.error(f"Profile extraction failed: {st.session_state.profile_error}")
        return

    if not st.session_state.patient_profile:
        st.markdown("""
        <div style="text-align:center; padding:30px; color:var(--text-muted);">
          Click "Extract Patient Profile" to begin.
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Patient Profile (Editable)</div>', unsafe_allow_html=True)

    profile_json = json.dumps(st.session_state.patient_profile, indent=2)
    edited_json = st.text_area(
        "Edit profile JSON",
        value=profile_json,
        height=220,
        label_visibility="collapsed"
    )

    try:
        edited_profile = json.loads(edited_json)
        st.session_state.patient_profile = edited_profile
    except json.JSONDecodeError:
        st.warning("Invalid JSON — please fix the syntax.")
        return

    with col2:
        if st.button("Step 2: Find Clinical Trials", width='stretch'):
            with st.spinner("Searching ClinicalTrials.gov + running eligibility analysis…"):
                try:
                    profile_str = json.dumps(st.session_state.patient_profile)
                    result = run_tool_directly("clinical_trial_matcher", profile_str)
                    parsed = json.loads(result)
                    if isinstance(parsed, str):
                        st.session_state.trial_results = {"error": parsed}
                    else:
                        st.session_state.trial_results = parsed
                except Exception as e:
                    st.error(f"Trial matching failed: {str(e)}")
                    return
            st.rerun()

    if not st.session_state.trial_results:
        return

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Ranked Clinical Trials</div>', unsafe_allow_html=True)

    results = st.session_state.trial_results
    if isinstance(results, str):
        st.error(results)
        return

    if isinstance(results, dict):
        if "error" in results:
            st.error(results["error"])
            return
        if "message" in results:
            st.info(results["message"])
            return
        st.error(f"Unexpected response: {results}")
        return

    if not isinstance(results, list):
        st.error(f"Unexpected result type: {type(results).__name__}")
        return

    results = [r for r in results if isinstance(r, dict)]

    if len(results) == 0:
        st.info("No matching trials found. Try editing the profile with a different diagnosis.")
        return

    for i, trial in enumerate(results):
        score = trial.get("eligibility_score", 0)
        score_pct = int(score * 100)

        if score >= 0.7:
            color = "var(--teal)"
            badge = "Likely Eligible"
        elif score >= 0.4:
            color = "var(--amber)"
            badge = "Possibly Eligible"
        else:
            color = "var(--rose)"
            badge = trial.get("verdict", "Likely Ineligible")

        st.markdown(f"""
        <div class="trial-card">
          <div class="trial-card-header">
            <div>
              <span class="trial-nct">{trial.get('nct_id', 'N/A')}</span>
              <span class="pill pill-{'teal' if score >= 0.7 else 'amber' if score >= 0.4 else 'rose'}" style="margin-left:8px;">{badge}</span>
            </div>
            <div style="font-size:0.82rem; color:var(--text-muted);">Score: {score_pct}%</div>
          </div>
          <div class="trial-title">{trial.get('title', 'No title')}</div>
          <div class="trial-meta">
            <span class="pill pill-sky">{trial.get('status', '')}</span>
            <span class="pill pill-sky">{trial.get('phase', '')}</span>
          </div>
          <div class="score-bar"><div class="score-fill" style="width:{score_pct}%"></div></div>
          <p style="font-size:0.85rem; color:var(--text); margin:10px 0;">{trial.get('explanation', '')}</p>
        """, unsafe_allow_html=True)

        matched = trial.get("matched_criteria", [])
        unmet = trial.get("unmet_criteria", [])
        uncertain = trial.get("uncertain_criteria", [])

        if matched:
            st.markdown('<p style="font-size:0.78rem; color:var(--teal); font-weight:600; margin:6px 0 2px;">Matched Criteria:</p>', unsafe_allow_html=True)
            items = "".join(f"<li>{c}</li>" for c in matched)
            st.markdown(f'<ul class="criteria-list">{items}</ul>', unsafe_allow_html=True)

        if unmet:
            st.markdown('<p style="font-size:0.78rem; color:var(--rose); font-weight:600; margin:6px 0 2px;">Unmet Criteria:</p>', unsafe_allow_html=True)
            items = "".join(f"<li>{c}</li>" for c in unmet)
            st.markdown(f'<ul class="criteria-list">{items}</ul>', unsafe_allow_html=True)

        if uncertain:
            st.markdown('<p style="font-size:0.78rem; color:var(--amber); font-weight:600; margin:6px 0 2px;">Uncertain:</p>', unsafe_allow_html=True)
            items = "".join(f"<li>{c}</li>" for c in uncertain)
            st.markdown(f'<ul class="criteria-list">{items}</ul>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:16px; padding:12px; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25); border-radius:8px; font-size:0.78rem; color:var(--amber);">
      This tool is for informational purposes only. Always consult with a healthcare provider and the trial investigators before making decisions about clinical trial participation.
    </div>
    """, unsafe_allow_html=True)


def _get_profile_llm():
    """Get ChatGroq for profile extraction."""
    from langchain_groq import ChatGroq
    if "profile_llm" not in st.session_state:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        st.session_state.profile_llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=4096,
        )
    return st.session_state.profile_llm


# ── Results panel ──────────────────────────────────────────────────────────────
def render_results():
    tabs = st.tabs(["Analysis", "Extracted Text", "Chat", "Clinical Trials"])

    with tabs[0]:
        render_analysis_tab()

    with tabs[1]:
        render_ocr_tab()

    with tabs[2]:
        render_chat_tab()

    with tabs[3]:
        render_trials_tab()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_state()
    render_sidebar()

    st.markdown("""
    <div style="margin-bottom: 6px;">
      <h1 class="page-title">Medical Report <span>Assistant</span></h1>
      <p class="page-subtitle">Upload · extract · analyse · match clinical trials</p>
    </div>
    """, unsafe_allow_html=True)

    if not os.getenv("GROQ_API_KEY"):
        st.warning("Add your Groq API key in the sidebar to continue.")
        st.markdown(
            "**Get a free key:** visit [console.groq.com](https://console.groq.com), "
            "sign up, create an API key, and paste it in the sidebar."
        )
        st.stop()

    if not st.session_state.ocr:
        st.session_state.ocr = TesseractOCR()
    if not st.session_state.agent:
        try:
            st.session_state.agent = build_agent()
        except Exception as e:
            st.error(f"Failed to initialise agent: {e}")
            st.stop()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card-header">Upload Report</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop image here", type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded)

            disp = image.copy()
            if disp.width > 580:
                r = 580 / disp.width
                disp = disp.resize((580, int(disp.height * r)), Image.Resampling.LANCZOS)
            st.image(disp, width='stretch')

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Full Analysis", width='stretch'):
                    process_image(image, mode="full")
            with c2:
                if st.button("Quick Analysis", width='stretch'):
                    process_image(image, mode="quick")

        else:
            st.markdown("""
            <div style="text-align:center; padding:40px 20px; color:var(--text-muted);">
              <div style="font-size:2.5rem; margin-bottom:12px;">🖼️</div>
              <p style="font-size:0.88rem;">JPG · PNG · JPEG · BMP</p>
              <p style="font-size:0.78rem; margin-top:4px;">Max file size 200 MB</p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        if st.session_state.analysis:
            render_results()
        else:
            render_welcome()


if __name__ == "__main__":
    main()
