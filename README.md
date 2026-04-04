# 🏥 Medical Report Simplifier

An intelligent, AI-powered web application that extracts text from medical report images using OCR and provides clear, easy-to-understand explanations using advanced language models. Transform complex medical jargon into actionable insights.

**[🌐 Live Demo](https://medicalreportsimplifier-kazol.streamlit.app/#medical-report-assistant)** | **[GitHub](https://github.com/kazol196295/medical_report_simplifier)**

---
## Project Screenshot
<table>
  <tr>
    <td><img src="[https://github.com](https://github.com/kazol196295/medical_report_simplifier/blob/main/sample%201.jpeg)" width="400" alt="Sample 1"></td>
    <td><img src="[https://github.com](https://github.com/kazol196295/medical_report_simplifier/blob/main/sample%202.png)" width="400" alt="Sample 2"></td>
  </tr>
</table>


## ✨ Features

### 📋 Smart Document Processing
- **OCR Extraction**: Extract text from medical images (JPG, PNG, JPEG, BMP) using Tesseract OCR
- **Image Support**: Handles blood tests, lab reports, radiology images, prescriptions, and more
- **High Accuracy**: Supports medical-grade text recognition with English language optimization

### 🤖 AI-Powered Analysis
- **Full Agent Analysis**: Comprehensive AI-driven insights powered by Groq's LPU
- **Quick Analysis**: Fast processing for rapid results
- **LLaMA 3.3-70B Model**: State-of-the-art language model via Groq API
- **800+ Tokens/Second**: Lightning-fast inference for real-time responses

### 📚 RAG (Retrieval-Augmented Generation)
- **Intelligent Indexing**: Automatic RAG index creation from extracted text
- **Contextual Retrieval**: Answers grounded in your specific medical report
- **Semantic Search**: Use FAISS + Hugging Face embeddings for accurate retrieval

### 💬 Interactive Q&A
- **Follow-up Questions**: Ask clarifying questions about your report
- **RAG-Powered Chat**: Get answers specific to your medical data
- **Context Awareness**: Maintains conversation history for better understanding

### 📊 Detailed Statistics
- **Real-time Metrics**: Character count, word count, line count tracking
- **RAG Status**: Monitor indexed chunks and retrieval readiness
- **Performance Timing**: OCR, RAG indexing, and AI analysis duration tracking

### 🎨 Modern UI/UX
- **Beautiful Dark Theme**: Navy, teal, and sky blue color scheme
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Smooth Interactions**: Polished animations and transitions
- **Accessibility**: Clean typography and readable contrast ratios

### 🔐 Privacy-First Design
- **No Permanent Storage**: Session-based data only
- **Secure API Integration**: Your Groq API key never leaves your session
- **Medical Confidentiality**: No data sent to external servers

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Tesseract OCR installed on your system
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/kazol196295/medical_report_simplifier.git
cd medical_report_simplifier
```

#### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng libgl1 libglx0 libsm6 libxext6 libxrender-dev libgomp1 libtesseract-dev
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download and install from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

#### 4. Get Your Groq API Key
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Create a new API key
4. Copy the key for use in the app

### Running Locally

#### Option A: With Streamlit Secrets (Recommended)
```bash
mkdir -p ~/.streamlit
echo 'GROQ_API_KEY = "your_api_key_here"' >> ~/.streamlit/secrets.toml
streamlit run app.py
```

#### Option B: Manual API Key Entry
```bash
streamlit run app.py
```
Then paste your API key in the sidebar when prompted.

The app will open at `http://localhost:8501`

---

## 📖 How to Use

### Step 1: Upload a Medical Report
1. Click the upload area or drag & drop a medical report image
2. Supported formats: JPG, PNG, JPEG, BMP
3. Image quality affects OCR accuracy — well-lit, clear images work best

### Step 2: Choose Analysis Mode
- **🔍 Full Agent Analysis**: Comprehensive breakdown with detailed insights and medical interpretation
- **⚡ Quick Analysis**: Fast summary for immediate understanding of key findings

### Step 3: Review Results
- **🧠 Agent Analysis Tab**: AI-generated insights and explanations
- **📝 Extracted Text Tab**: Raw OCR output with character/word/line statistics
- **💬 Chat Tab**: Ask follow-up questions about your report

### Step 4: Download or Ask Questions
- Download analysis as `.txt` file
- Download raw OCR text
- Ask follow-up questions using the RAG-powered chat
- Export results for sharing with healthcare providers

### Example Questions You Can Ask
```
"What does a high WBC count mean?"
"Explain my cholesterol levels"
"Are these results normal for my age?"
"What do these abnormal values indicate?"
"Should I be concerned about this result?"
```

---

## 🏗️ Project Architecture

### Directory Structure
```
medical_report_simplifier/
├── app.py                      # Main Streamlit application (592 lines)
├── requirements.txt            # Python package dependencies
├── packages.txt               # System dependencies for Streamlit deployment
├── runtime.txt                # Python runtime version specification
├── README.md                  # This file
├── src/
│   ├── __init__.py
│   ├── ocr_tesseract.py      # OCR text extraction module
│   └── groq_agent.py         # AI analysis, RAG indexing, and chat agent
└── sample image/              # Example medical report images for testing
```

### Core Components

#### 1. **OCR Module** (`src/ocr_tesseract.py`)
Responsible for:
- Extracting text from medical images
- Image preprocessing and optimization
- Handling various image formats
- Returns structured, clean text for analysis

#### 2. **Groq Medical Agent** (`src/groq_agent.py`)
Handles:
- Medical report analysis using LLaMA 3.3-70B
- Building and managing RAG (Retrieval-Augmented Generation) indices
- Semantic search using FAISS vector database
- Generating contextual follow-up responses
- Grounding answers in extracted report text

#### 3. **Streamlit Frontend** (`app.py`)
Features:
- Modern, responsive web interface
- Real-time image processing
- Interactive chat with RAG support
- Performance monitoring and statistics
- Beautiful dark theme with custom CSS

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Groq (LLaMA 3.3-70B) | Medical analysis & natural language understanding |
| **OCR** | Tesseract OCR | Text extraction from images |
| **RAG Framework** | LangChain | LLM orchestration and memory management |
| **Vector DB** | FAISS | Semantic similarity search |
| **Embeddings** | Sentence Transformers | Converting text to embeddings |
| **Frontend** | Streamlit 1.32.2 | Web interface and user interactions |
| **Image Processing** | Pillow + NumPy | Image handling and manipulation |
| **Data Validation** | Pydantic 2.7.0 | Schema validation and type checking |

### Key Dependencies

```
groq==latest                    # Groq API client
pytesseract                     # Python wrapper for Tesseract
Pillow                          # Image processing
numpy                           # Numerical computing
langchain==0.2.0                # LLM framework
langchain-core==0.2.2           # Core LangChain utilities
langchain-groq==0.1.6           # Groq integration for LangChain
langchain-community             # Community integrations
langchain-huggingface           # Hugging Face integration
sentence-transformers           # Embedding models
faiss-cpu                       # Vector similarity search
streamlit==1.32.2               # Web framework
pydantic==2.7.0                 # Data validation
```

---

## ⚙️ System Requirements

### Python Environment
- **Python Version**: 3.9 or higher
- **Virtual Environment**: Recommended

### System Packages (for OCR)
```bash
# Ubuntu/Debian
libgl1 libglx0 libsm6 libxext6 libxrender-dev libgomp1 tesseract-ocr tesseract-ocr-eng libtesseract-dev

# CentOS/RHEL
tesseract-devel tesseract-langpack-eng

# macOS
brew install tesseract

# Windows
Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

---

## 📊 Performance Characteristics

| Operation | Typical Duration | Factors |
|-----------|-----------------|---------|
| OCR Processing | 1-2 seconds | Image resolution, clarity |
| RAG Indexing | 2-5 seconds | Report length, complexity |
| AI Analysis | 3-8 seconds | Model processing, token count |
| Follow-up Response | 1-3 seconds | Query complexity, context size |
| **Total Workflow** | **7-18 seconds** | Depends on all above factors |

### Performance Tips
- Use clear, well-lit medical report images
- Ensure tesseract-ocr is properly installed
- Run on a machine with stable internet for Groq API calls
- Use Streamlit Cloud for consistent performance

---

## 🔐 Privacy & Security

### Data Handling
✅ **No permanent data storage**
- Session-based processing only
- Data cleared when browser closes
- No logs containing medical content
- No data retention between sessions

### API Security
✅ **Secure API Integration**
- Your Groq API key stored only in your browser/local machine
- No API keys logged or stored server-side
- Direct API calls from your environment
- Support for `.streamlit/secrets.toml` for safe key storage

### Medical Data Privacy
✅ **Compliant Handling**
- No external data uploads to third parties
- All processing through Groq API only
- GDPR-friendly design
- No tracking or analytics on report content

### ⚠️ Important Disclaimer
> **This tool is for EDUCATIONAL and INFORMATIONAL purposes ONLY.**
> 
> - **NOT** a medical diagnosis tool
> - **NOT** a substitute for professional medical advice
> - **ALWAYS** consult qualified healthcare professionals
> - Results should **NEVER** be used for treatment decisions
> - User assumes all responsibility for any use of this tool

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Easiest)

**Step-by-step:**
1. Push your code to a public GitHub repository
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and `app.py` file
5. In **Advanced settings**, add your secret:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
6. Deploy!

**Live Demo Already Running:**
🌐 **[https://medicalreportsimplifier-kazol.streamlit.app/](https://medicalreportsimplifier-kazol.streamlit.app/#medical-report-assistant)**

### Option 2: Docker Deployment

```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 libglx0 libsm6 libxext6 libxrender-dev \
    libgomp1 libtesseract-dev

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

### Option 3: Heroku Deployment

```yaml
# Procfile
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 📋 Configuration

### Environment Variables

```bash
# Required
GROQ_API_KEY=gsk_your_api_key_here

# Optional
STREAMLIT_SERVER_FILE_WATCHER_TYPE=none  # Use on Linux systems
```

### Streamlit Secrets File
Create `~/.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_api_key_here"
```

---

## 🤝 Contributing

We welcome contributions! Here's how to help:

### Development Setup
```bash
git clone https://github.com/kazol196295/medical_report_simplifier.git
cd medical_report_simplifier
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Contribution Steps
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes
4. **Commit**: `git commit -m 'Add amazing feature'`
5. **Push**: `git push origin feature/amazing-feature`
6. **Create** a Pull Request

### Areas for Contribution
- [ ] Multi-language support (Spanish, German, French, etc.)
- [ ] Additional OCR engines (EasyOCR, PaddleOCR)
- [ ] Export formats (PDF, DOCX, JSON)
- [ ] Medical history tracking
- [ ] Mobile app version (React Native)
- [ ] EHR system integration (HL7, FHIR)
- [ ] Specialized analysis modes (Pediatric, Geriatric, Cardiology)
- [ ] Bulk processing for multiple reports
- [ ] Advanced filtering and search
- [ ] User authentication system

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'pytesseract'"
**Solution:**
```bash
pip install pytesseract
# Also ensure Tesseract OCR is installed on your system
```

#### 2. "Tesseract is not installed or it's not in your PATH"
**Solution:**
- **Ubuntu**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

#### 3. "GROQ_API_KEY not found"
**Solution:**
- Add key to Streamlit sidebar before running
- Or set environment variable: `export GROQ_API_KEY="your_key"`
- Or add to `~/.streamlit/secrets.toml`

#### 4. Poor OCR Accuracy
**Solutions:**
- Use higher resolution images (300+ DPI)
- Ensure good lighting and contrast
- Avoid shadows and blur
- Use JPG or PNG format
- Try rotating the image if text is skewed

#### 5. "Permission denied: 'tesseract'"
**Solution:**
```bash
# Ubuntu/Debian
sudo chmod +x /usr/bin/tesseract
```

### Getting Help

- 🐛 **Report bugs**: [GitHub Issues](https://github.com/kazol196295/medical_report_simplifier/issues)
- 💡 **Request features**: [GitHub Discussions](https://github.com/kazol196295/medical_report_simplifier/discussions)
- 📧 **Email support**: Open an issue with [support] tag

---

## 📚 API Reference

### Groq API Setup

```python
from groq import Groq

client = Groq(api_key="gsk_your_key")
message = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Your prompt"}]
)
```

### LangChain Integration

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key="gsk_your_key"
)
```

### FAISS Vector Store

```python
from faiss import IndexFlatL2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)
index = IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
```

---

## 📈 Roadmap

### v1.0 ✅ (Current Release)
- ✅ OCR text extraction
- ✅ AI-powered analysis
- ✅ RAG-powered Q&A
- ✅ Beautiful UI/UX
- ✅ Performance monitoring

### v1.1 (Q2 2026)
- [ ] Multi-language support (6+ languages)
- [ ] Batch processing (multiple reports)
- [ ] Advanced filtering and search
- [ ] Export to multiple formats (PDF, DOCX)
- [ ] Custom analysis templates

### v2.0 (Q4 2026)
- [ ] Mobile app (iOS/Android)
- [ ] EHR system integration
- [ ] Doctor verification system
- [ ] Health trends analysis
- [ ] Telemedicine integration
- [ ] Medical history tracking
- [ ] Prescription management

---

## 📞 Support & Community

### Where to Get Help
| Channel | Best For | Link |
|---------|----------|------|
| **GitHub Issues** | Bug reports | [Issues](https://github.com/kazol196295/medical_report_simplifier/issues) |
| **Discussions** | Questions & ideas | [Discussions](https://github.com/kazol196295/medical_report_simplifier/discussions) |
| **Email** | Business inquiries | Open an issue with [inquiry] tag |

### FAQ

**Q: Is this app accurate?**
A: The accuracy depends on image quality and report clarity. OCR accuracy is typically 85-95% for clear medical documents. Always verify AI analysis with a healthcare professional.

**Q: Can I use this commercially?**
A: The MIT license allows commercial use. However, ensure you comply with medical data regulations (HIPAA, GDPR) if handling real patient data.

**Q: Does it work offline?**
A: No. The app requires internet for Groq API and Hugging Face embeddings. Tesseract OCR is the only offline component.

**Q: Is my data safe?**
A: Yes! All data is session-based and never permanently stored. No medical content is logged or retained.

**Q: Can I integrate this into my health platform?**
A: Yes! The modular design allows integration. Check the `src/` modules for API interfaces.

**Q: How do I report a security vulnerability?**
A: Email directly or open a private security issue. Do not publicly disclose vulnerabilities.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
```

---

## 🙏 Acknowledgments

### Technology Partners
- **[Groq](https://groq.com/)** - LPU-powered inference, 800+ tok/sec
- **[LangChain](https://langchain.com/)** - LLM orchestration framework
- **[Streamlit](https://streamlit.io/)** - Rapid web development
- **[Hugging Face](https://huggingface.co/)** - Pre-trained embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** - Vector similarity search
- **[Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)** - OCR engine

### Open Source Community
- Contributors and issue reporters
- Stack Overflow community
- GitHub community

---

## ⭐ Show Your Support

If this project has been helpful to you:

- ⭐ **Star** the repository
- 🔗 **Share** with others who might benefit
- 🤝 **Contribute** improvements and features
- 💬 **Give feedback** and suggestions
- 📢 **Spread the word** on social media

---

## 📞 Connect With Us

- **GitHub**: [@kazol196295](https://github.com/kazol196295)
- **Project Demo**: [Live App](https://medicalreportsimplifier-kazol.streamlit.app/#medical-report-assistant)
- **Repository**: [medical_report_simplifier](https://github.com/kazol196295/medical_report_simplifier)

---

## 🎯 Key Takeaways

| Feature | Benefit |
|---------|---------|
| **OCR + AI** | Accurate text extraction + intelligent analysis |
| **RAG-Powered** | Answers grounded in your specific report |
| **Fast** | 800+ tokens/sec with Groq LPU |
| **Private** | Session-based, no permanent storage |
| **Beautiful** | Modern dark theme with smooth interactions |
| **Easy** | One-click deployment to Streamlit Cloud |

---

**Made with ❤️ by [kazol196295](https://github.com/kazol196295)**

**Last Updated**: April 4, 2026
**Version**: 1.0.0
**Status**: ✅ Active & Maintained

---

*Disclaimer: This tool is for educational purposes. Always consult healthcare professionals for medical decisions.*
