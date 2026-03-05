# app.py
import streamlit as st
from PIL import Image
import os
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Report Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Imports ────────────────────────────────────────────────────────────────────
from src.ocr_tesseract import TesseractOCR
from src.groq_agent import GroqMedicalAgent

# ── Design System CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── CSS Variables ── */
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

/* RAG status badge */
.rag-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(0,201,167,0.08); border: 1px solid rgba(0,201,167,0.25);
  border-radius: 8px; padding: 6px 12px; font-size: 0.78rem; color: var(--teal);
  margin-bottom: 12px;
}

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
        'active_tab': 0,
        'rag_chunks': 0,          # NEW: track how many chunks are indexed
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
            <span>Intelligent Analysis</span>
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
            st.markdown('<span class="pill pill-teal">✓ &nbsp;Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill-amber">⚠ &nbsp;Key required</span>', unsafe_allow_html=True)

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

        # ── NEW: RAG index status ──────────────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="sb-label">RAG Index</p>', unsafe_allow_html=True)
        if st.session_state.rag_chunks > 0:
            st.markdown(f"""
            <div class="rag-badge">
              🗂 &nbsp;<strong>{st.session_state.rag_chunks}</strong>&nbsp;chunks indexed
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<span class="pill pill-teal">✓ &nbsp;RAG Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill-amber">⏳ &nbsp;Not indexed yet</span>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="sb-label">Model</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="stat-chip" style="margin-bottom:8px;">
          <strong>llama-3.3-70b</strong>via Groq LPU
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.78rem; color:var(--text-muted); line-height:1.7;">
          🔒 &nbsp;No data stored permanently<br>
          ⚡ &nbsp;800+ tokens / second<br>
          🗂 &nbsp;RAG-powered chat<br>
          🏥 &nbsp;Always consult a doctor
        </div>
        """, unsafe_allow_html=True)


# ── Image processing ───────────────────────────────────────────────────────────
def process_image(image, mode="full"):
    start = time.time()

    # ── Step 1: OCR ───────────────────────────────────────────────────────────
    with st.spinner("🔬 Extracting text from image…"):
        st.session_state.extracted_text = st.session_state.ocr.extract_text(image)

    if not st.session_state.extracted_text:
        st.error("Could not read text — try a clearer, well-lit image.")
        return

    ocr_t = time.time() - start

    # ── Step 2 (NEW): Build RAG index from extracted text ────────────────────
    with st.spinner("📚 Building RAG index…"):
        n_chunks = st.session_state.agent.build_rag_index(
            st.session_state.extracted_text
        )
        st.session_state.rag_chunks = n_chunks  # save for sidebar display

    rag_t = time.time() - start - ocr_t

    # ── Step 3: AI Analysis ───────────────────────────────────────────────────
    label = "Agent" if mode == "full" else "Quick"
    with st.spinner(f"🤖 Running {label} Analysis via Groq…"):
        if mode == "full":
            st.session_state.analysis = st.session_state.agent.analyze_report(
                st.session_state.extracted_text
            )
        else:
            st.session_state.analysis = st.session_state.agent.quick_analysis(
                st.session_state.extracted_text
            )

    total_t = time.time() - start
    ai_t = total_t - ocr_t - rag_t

    st.success(
        f"✓ Done in {total_t:.1f}s  —  "
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
        <strong style="color:var(--teal);">Full Agent Analysis</strong> or
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


# ── Results panel ──────────────────────────────────────────────────────────────
def render_results():
    tabs = st.tabs(["🧠 Agent Analysis", "📝 Extracted Text", "💬 Chat"])

    # ── Tab 1: Analysis ────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="card-header">🧠 AI Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="analysis-box">{st.session_state.analysis}</div>',
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "⬇ Download", st.session_state.analysis,
                file_name="medical_analysis.txt", use_container_width=True
            )
        with col2:
            if st.button("🔄 Re-analyse", use_container_width=True):
                st.session_state.analysis = None
                st.rerun()
        with col3:
            if st.button("🗑 Clear All", use_container_width=True):
                st.session_state.extracted_text = None
                st.session_state.analysis = None
                st.session_state.chat_history = []
                st.session_state.rag_chunks = 0   # NEW: reset chunk counter
                st.rerun()

    # ── Tab 2: OCR text ────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="card-header">📝 Extracted OCR Text</div>', unsafe_allow_html=True)

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
            "⬇ Download Raw Text", st.session_state.extracted_text,
            file_name="ocr_extracted.txt", use_container_width=True
        )

    # ── Tab 3: Chat (RAG-powered) ──────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="card-header">💬 Ask Follow-up Questions</div>', unsafe_allow_html=True)

        # Show RAG status in chat tab
        if st.session_state.rag_chunks > 0:
            st.markdown(
                f'<div class="rag-badge">🗂 RAG active &nbsp;·&nbsp; '
                f'{st.session_state.rag_chunks} chunks &nbsp;·&nbsp; '
                f'answers grounded in your report</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="rag-badge" style="border-color:rgba(245,158,11,0.3);color:var(--amber);">'
                '⚠ RAG not active — upload and analyse a report first</div>',
                unsafe_allow_html=True
            )

        # Render chat history
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
              💡 Try: <em>"What does a high WBC count mean?"</em> or
              <em>"Explain my cholesterol levels."</em>
            </p>
            """, unsafe_allow_html=True)

        question = st.text_input(
            "Your question",
            placeholder="Ask anything about your report…",
            label_visibility="collapsed"
        )
        if st.button("Send ➤", use_container_width=True) and question.strip():
            st.session_state.chat_history.append(("user", question))
            with st.spinner("🔍 Retrieving relevant context…"):
                reply = st.session_state.agent.chat_followup(question)
            st.session_state.chat_history.append(("ai", reply))
            st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_state()
    render_sidebar()

    st.markdown("""
    <div style="margin-bottom: 6px;">
      <h1 class="page-title">Medical Report <span>Assistant</span></h1>
      <p class="page-subtitle">Upload · extract · index · get AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Guard: API key
    if not os.getenv("GROQ_API_KEY"):
        st.warning("Add your Groq API key in the sidebar to continue.")
        st.markdown(
            "**Get a free key:** visit [console.groq.com](https://console.groq.com), "
            "sign up, create an API key, and paste it in the sidebar."
        )
        st.stop()

    # Init engines once per session
    if not st.session_state.ocr:
        st.session_state.ocr = TesseractOCR()
    if not st.session_state.agent:
        try:
            st.session_state.agent = GroqMedicalAgent()
        except Exception as e:
            st.error(f"Failed to initialise agent: {e}")
            st.stop()

    # ── Two-column layout ──────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card-header">📤 Upload Report</div>', unsafe_allow_html=True)

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
            st.image(disp, use_column_width=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔍 Full Agent Analysis", use_container_width=True):
                    process_image(image, mode="full")
            with c2:
                if st.button("⚡ Quick Analysis", use_container_width=True):
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