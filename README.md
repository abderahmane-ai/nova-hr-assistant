# 🚀 Nova HR Assistant

> **Next-generation AI-powered CV analysis and candidate evaluation platform**

Transform your hiring process with intelligent CV analysis powered by cutting-edge AI. Nova HR Assistant provides comprehensive candidate evaluation, smart scoring, and actionable insights to help you make better hiring decisions faster.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-Powered-green.svg)](https://langchain.com/)

## ✨ Key Features

### 🎯 **Intelligent Analysis**
- **Multi-LLM Support**: Choose from Groq, Google Gemini, OpenAI, Anthropic, or local models
- **Advanced Scoring**: Comprehensive evaluation across experience, skills, education, and certifications
- **Bias Detection**: Built-in bias mitigation and fairness scoring
- **Confidence Metrics**: AI confidence levels for each evaluation component

### 📄 **Universal File Support**
- **PDF Processing**: Extract text from professional PDF resumes
- **Word Documents**: Full support for .docx files
- **Plain Text**: Process .txt format CVs
- **Batch Processing**: Analyze entire directories of CVs simultaneously

### 🎨 **Rich Interfaces**
- **Command Line**: Powerful CLI for batch processing and automation
- **Web Interface**: Beautiful Streamlit app with interactive visualizations
- **Comparison Tools**: Side-by-side candidate analysis and ranking

### 📊 **Advanced Analytics**
- **Interactive Charts**: Radar plots, score gauges, and skill distributions
- **Candidate Ranking**: Sortable tables with detailed metrics
- **Export Options**: JSON reports and HTML visualizations
- **Progress Tracking**: Real-time processing updates

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/abderahmane-ai/nova-hr-assistant.git
cd nova-hr-assistant

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys and preferences
```

### 2. Configuration

Edit `.env` file with your preferred settings:

```bash
# Choose your AI provider (groq recommended for speed)
NOVA_LLM_PROVIDER=groq
NOVA_LLM_MODEL=openai/gpt-oss-120b
NOVA_LLM_API_KEY=your_api_key_here

# Customize scoring weights
NOVA_WEIGHT_EXPERIENCE=0.3
NOVA_WEIGHT_SKILLS=0.25
NOVA_WEIGHT_EDUCATION=0.2
NOVA_WEIGHT_CERTIFICATIONS=0.15
NOVA_WEIGHT_OVERALL_FIT=0.1
```

### 3. Start Analyzing

#### 🖥️ **Web Interface** (Recommended)
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` for the full interactive experience.

#### ⚡ **Command Line** (Power Users)
```bash
# Single CV analysis
python nova.py resume.pdf "Senior Python Developer with 5+ years experience"

# Batch processing
python nova.py --directory ./cvs "Data Scientist role requiring ML expertise" --delay 2

# Generate comparison reports
python visualize_candidates.py --output ./reports/comparison.html
```

## 🎯 Usage Examples

### Single File Analysis
```bash
# Analyze a PDF resume
python nova.py john_doe_resume.pdf "Senior Software Engineer with React and Node.js experience"

# Analyze with debug output
python nova.py jane_smith_cv.docx "Product Manager with 3+ years in SaaS" --debug
```

### Batch Processing
```bash
# Process entire directory
python nova.py --directory ./candidate_cvs "Full Stack Developer position"

# Custom processing with 5-second delays
python nova.py --directory ./cvs "DevOps Engineer with Kubernetes" --delay 5 --debug
```

### Visualization & Comparison
```bash
# Generate comparison from results folder
python visualize_candidates.py

# Custom output location
python visualize_candidates.py --output ./hiring_reports/q4_candidates.html

# Process specific results folder
python visualize_candidates.py --results-folder ./archived_results
```

## 🏗️ Architecture

Nova HR Assistant uses a sophisticated workflow architecture:

```
CV Input → Parser → Experience → Skills → Education → Certifications → Scorer → Report
    ↓         ↓         ↓         ↓         ↓             ↓          ↓        ↓
  Text     Candidate  Analysis  Analysis  Analysis    Analysis   Score   Final
Extract    Info      Results   Results   Results     Results   (0-100)  Report
```

### Core Components

- **🧠 LangGraph Workflow**: Orchestrates the analysis pipeline
- **🔄 State Management**: Tracks progress and handles errors gracefully  
- **🎯 Multi-Node Processing**: Specialized analysis for each CV component
- **📊 Advanced Scoring**: Weighted evaluation with confidence metrics
- **🛡️ Error Recovery**: Robust handling of failures and partial results

## 🎨 Web Interface Features

### Dashboard Overview
- **📈 Real-time Progress**: Watch analysis unfold step-by-step
- **🎯 Score Visualization**: Interactive gauges and radar charts
- **📋 Candidate Profiles**: Comprehensive candidate information
- **⚡ Quick Actions**: Upload, analyze, and export in one flow

### Advanced Analytics
- **🔍 Component Breakdown**: Detailed scoring for each evaluation area
- **💪 Strengths & Gaps**: AI-identified candidate highlights and improvement areas
- **🎨 Customizable Themes**: Multiple color schemes for presentations
- **📱 Responsive Design**: Works perfectly on desktop and mobile

## ⚙️ Configuration Options

### LLM Providers

| Provider | Speed | Quality | Cost | Best For |
|----------|-------|---------|------|----------|
| **Groq** | ⚡⚡⚡ | ⭐⭐⭐ | 💰 | High-volume processing |
| **Google Gemini** | ⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | Advanced reasoning |
| **OpenAI GPT-4** | ⚡ | ⭐⭐⭐⭐⭐ | 💰💰💰 | Highest quality |
| **Anthropic Claude** | ⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | Balanced performance |
| **OpenRouter** | ⚡⚡ | ⭐⭐⭐⭐ | 💰💰 | Model variety |

### Scoring Weights

Customize evaluation criteria to match your hiring priorities:

```bash
# Technical roles
NOVA_WEIGHT_SKILLS=0.4
NOVA_WEIGHT_EXPERIENCE=0.3
NOVA_WEIGHT_EDUCATION=0.2
NOVA_WEIGHT_CERTIFICATIONS=0.1

# Leadership roles  
NOVA_WEIGHT_EXPERIENCE=0.4
NOVA_WEIGHT_OVERALL_FIT=0.3
NOVA_WEIGHT_SKILLS=0.2
NOVA_WEIGHT_EDUCATION=0.1
```

## 📊 Output Formats

### JSON Reports
Comprehensive machine-readable analysis with:
- Candidate information and contact details
- Detailed component scores and reasoning
- Strengths, gaps, and recommendations
- Confidence metrics and bias detection
- Processing metadata and timestamps

### Interactive HTML
Beautiful web reports featuring:
- Executive summary dashboards
- Interactive score visualizations
- Candidate comparison tables
- Exportable charts and graphs
- Mobile-responsive design

### Console Output
Real-time processing updates with:
- Progress indicators and timelines
- Score breakdowns and highlights
- Error handling and recovery status
- Batch processing summaries
- Performance metrics

## 🛠️ Advanced Features

### Bias Detection & Mitigation
- **Automatic Detection**: Identifies potential biases in evaluation
- **Score Adjustment**: Applies fairness corrections when needed
- **Transparency**: Reports detected biases and mitigation actions
- **Compliance**: Helps maintain fair hiring practices

### Confidence Scoring
- **Component Confidence**: Individual confidence for each analysis area
- **Overall Confidence**: Aggregate confidence in the final score
- **Risk Assessment**: Identifies low-confidence evaluations
- **Quality Indicators**: Helps prioritize manual review

### Error Recovery
- **Graceful Degradation**: Continues processing despite individual failures
- **Partial Results**: Generates reports even with incomplete analysis
- **Retry Logic**: Automatically retries failed operations
- **Detailed Logging**: Comprehensive error tracking and debugging

## 🔧 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **API Key Error** | Verify your API key in `.env` file |
| **File Not Found** | Check file path and permissions |
| **Unsupported Format** | Use PDF, DOCX, or TXT files only |
| **Rate Limiting** | Increase `--delay` parameter |
| **Memory Issues** | Process smaller batches or upgrade system |

### Debug Mode
Enable detailed logging for troubleshooting:
```bash
# CLI debug mode
python nova.py resume.pdf "Job description" --debug

# Environment debug mode
NOVA_DEBUG_MODE=true streamlit run app.py
```

### Log Files
- `nova.log` - Single file processing logs
- `nova_batch.log` - Batch processing logs  
- `nova_visualization.log` - Visualization logs

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black .

# Type checking
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for the powerful LLM orchestration framework
- **Streamlit** for the beautiful web interface capabilities
- **Plotly** for interactive visualizations
- **The AI Community** for continuous innovation in language models

---

<div align="center">

**Made with ❤️ by the Nova Team**

[🌟 Star us on GitHub](https://github.com/your-username/nova-hr-assistant) • [📖 Documentation](https://docs.nova-hr.com) • [💬 Discord Community](https://discord.gg/nova-hr)

</div>