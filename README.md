# Medical Report Simplifier

An AI-powered web application that extracts text from medical report images using OCR, provides clear explanations, and matches patients to clinical trials. Built with LangGraph, RAG, and Groq's LPU for fast, accurate analysis.

**[Live Demo](https://medicalreportsimplifier-kazol.streamlit.app/#medical-report-assistant)** | **[GitHub](https://github.com/kazol196295/medical_report_simplifier)**

---

## Features

### Document Processing
- **OCR Extraction**: Extract text from medical images (JPG, PNG, JPEG, BMP) using Tesseract OCR
- **Image Support**: Blood tests, lab reports, radiology images, prescriptions
- **High Accuracy**: Medical-grade text recognition with English language optimization

### AI-Powered Analysis
- **Full Analysis**: Comprehensive breakdown with detailed insights
- **Quick Analysis**: Fast summary of key findings
- **LLaMA 3.3-70B**: State-of-the-art language model via Groq API

### RAG (Retrieval-Augmented Generation)
- **Intelligent Indexing**: Automatic RAG index from extracted text
- **Contextual Retrieval**: Answers grounded in your specific report
- **Semantic Search**: FAISS + Hugging Face embeddings

### Clinical Trial Matching
- **Patient Profile Extraction**: Auto-extract age, gender, diagnosis, biomarkers, conditions from reports
- **ClinicalTrials.gov Integration**: Search real recruiting trials
- **Eligibility Scoring**: AI-powered matching with detailed criteria analysis
- **Editable Profiles**: Modify extracted profiles before searching

### Interactive Chat
- **Follow-up Questions**: Ask clarifying questions about your report
- **RAG-Powered**: Get answers specific to your medical data
- **Context Awareness**: Maintains conversation history

### Modern UI
- **Dark Theme**: Navy, teal, and sky blue color scheme
- **Responsive Design**: Works on desktop and mobile
- **Real-time Stats**: Character, word, line counts and RAG status

---

## Quick Start

### Prerequisites
- Python 3.9+
- Tesseract OCR installed
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone repository
git clone https://github.com/kazol196295/medical_report_simplifier.git
cd medical_report_simplifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng libgl1 libglx0 libsm6 libxext6 libxrender-dev libgomp1 libtesseract-dev
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Configuration

#### Option 1: Secrets File (Recommended)
Copy the example config and add your API key:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and replace with your actual API key
```

#### Option 2: Environment Variable
```bash
export GROQ_API_KEY="gsk_your_api_key_here"
```

#### Option 3: Manual Entry
Run the app and paste your API key in the sidebar when prompted.

### Run the App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## How to Use

### Step 1: Upload a Medical Report
- Click upload area or drag & drop a medical report image
- Supported: JPG, PNG, JPEG, BMP (clear, well-lit images work best)

### Step 2: Choose Analysis Mode
- **Full Analysis**: Comprehensive breakdown with detailed insights
- **Quick Analysis**: Fast summary of key findings

### Step 3: Explore Results
- **Analysis Tab**: AI-generated insights and explanations
- **Extracted Text Tab**: Raw OCR output with statistics
- **Chat Tab**: Ask follow-up questions about your report
- **Clinical Trials Tab**: Find matching clinical trials

### Step 4: Clinical Trials (Optional)
1. Click "Extract Patient Profile" to auto-extract patient data
2. Review/edit the extracted profile JSON
3. Click "Find Clinical Trials" to search ClinicalTrials.gov
4. View eligibility scores and matched/unmet criteria

### Example Questions
```
"What does a high WBC count mean?"
"Explain my cholesterol levels"
"Give me some health suggestions"
"Find clinical trials for my condition"
```

---

## Project Structure

```
medical_report_simplifier/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (Streamlit Cloud)
├── runtime.txt                     # Python runtime version
├── README.md                       # This file
├── .streamlit/
│   ├── config.toml                 # Streamlit theme configuration
│   ├── secrets.toml                # API keys (not committed)
│   └── secrets.toml.example        # Example secrets template
├── src/
│   ├── __init__.py
│   ├── ocr_tesseract.py            # OCR text extraction
│   ├── langgraph_agent.py          # LangGraph agent with tools
│   ├── tools.py                    # LangChain tool definitions
│   ├── prompts.py                  # LLM prompts and schemas
│   ├── patient_profiler.py         # Patient profile extraction
│   ├── clinical_trials_api.py      # ClinicalTrials.gov API client
│   └── rag_engine.py               # FAISS RAG engine
└── sample image/                   # Example medical reports
```

### Core Components

| File | Purpose |
|------|---------|
| `ocr_tesseract.py` | Extracts text from images using Tesseract |
| `langgraph_agent.py` | LangGraph state graph for conversational AI |
| `tools.py` | Defines tools: medical_analyzer, health_advisor, clinical_trial_matcher |
| `prompts.py` | LLM prompt templates for all tasks |
| `patient_profiler.py` | Extracts structured patient profiles from OCR text |
| `clinical_trials_api.py` | Searches ClinicalTrials.gov API v2 |
| `rag_engine.py` | FAISS-based RAG for semantic retrieval |

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Groq (LLaMA 3.3-70B) | Medical analysis & NLU |
| **OCR** | Tesseract OCR | Text extraction from images |
| **Agent** | LangGraph | State-based conversational agent |
| **RAG** | LangChain + FAISS | Retrieval-augmented generation |
| **Embeddings** | Sentence Transformers | Semantic text embeddings |
| **Frontend** | Streamlit | Web interface |
| **Trials API** | ClinicalTrials.gov API v2 | Real clinical trial data |

---

## Configuration

### Secrets File Format

Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_api_key_here"
```

See `.streamlit/secrets.toml.example` for the template.

### Streamlit Theme

`.streamlit/config.toml` is pre-configured with the dark theme. No changes needed.

---

## Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add secret: `GROQ_API_KEY = "your_key"`
5. Deploy

### Docker
```dockerfile
FROM python:3.9-slim
RUN apt-get update && apt-get install -y tesseract-ocr tesseract-ocr-eng libgl1 libglx0 libsm6 libxext6 libxrender-dev libgomp1 libtesseract-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

---

## Privacy & Security

- **No permanent storage**: Session-based data only
- **Secure API**: Key stored locally, never logged
- **Direct API calls**: No intermediary servers
- **GDPR-friendly**: No tracking or analytics on report content

### Disclaimer
> **This tool is for EDUCATIONAL and INFORMATIONAL purposes ONLY.**
> - NOT a medical diagnosis tool
> - NOT a substitute for professional medical advice
> - ALWAYS consult qualified healthcare professionals

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pytesseract` | `pip install pytesseract` + install Tesseract OCR |
| `Tesseract not in PATH` | Add Tesseract to system PATH or reinstall |
| `GROQ_API_KEY not found` | Add to sidebar, env var, or secrets.toml |
| Poor OCR accuracy | Use higher resolution, well-lit images |
| Profile extraction fails | App auto-retries; check if OCR text is readable |
| No clinical trials found | Try editing profile with different diagnosis |

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Acknowledgments

- **[Groq](https://groq.com/)** - LPU-powered inference
- **[LangChain](https://langchain.com/)** - LLM orchestration
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Agent framework
- **[Streamlit](https://streamlit.io/)** - Web framework
- **[ClinicalTrials.gov](https://clinicaltrials.gov/)** - Trial data API
- **[Hugging Face](https://huggingface.co/)** - Embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** - Vector search
- **[Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)** - OCR engine

---

**Made with ❤️ by [kazol196295](https://github.com/kazol196295)**
