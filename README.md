# NEMESIS-NEXUS v1.0

🚀 **Advanced Multi-Agent AI Cybersecurity Framework**

An next-generation cybersecurity platform featuring AI agent orchestration, uncensored LLM integration, and advanced red team automation capabilities.

## ⚠️ DISCLAIMER

**FOR AUTHORIZED PENETRATION TESTING ONLY**

This framework is designed exclusively for professional penetration testing and red team operations with proper authorization. Users are responsible for ensuring compliance with all applicable laws and regulations.

## 🛡️ Features

### AI-Powered Security Testing
- **Uncensored LLM Integration**: Leverages abliterated models via Ollama for unrestricted security research
- **Multi-Agent Orchestration**: LangGraph + CrewAI for coordinated security operations
- **Advanced Reasoning**: 27B parameter models for complex security analysis

### Comprehensive Tool Integration
- **Network Scanning**: Nmap integration with AI-guided reconnaissance
- **Web Security**: Selenium-based testing with intelligent crawling
- **OSINT Capabilities**: Automated intelligence gathering and analysis
- **Report Generation**: Professional PDF reports with executive summaries

### Hardware Optimization
- **High-Performance Computing**: Optimized for Intel i7-7800X (6 cores, 12 threads)
- **Memory Management**: Intelligent allocation for 80GB+ systems
- **Concurrent Processing**: Up to 16 parallel security scans

## 🚀 Installation

### Prerequisites
- Python 3.10+
- 16GB+ RAM recommended (80GB+ optimal)
- Ollama installed and running
- Linux environment (tested on Tsurugi Linux)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Xiah-666/Nexus-Framework.git
cd Nexus-Framework
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install additional packages:**
```bash
pip install langchain-community python-nmap
```

### Ollama Setup

1. **Install Ollama:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. **Download recommended models:**
```bash
ollama pull huihui_ai/gemma3-abliterated:27b
ollama pull dolphin-mixtral:8x7b
ollama pull nomic-embed-text
```

## 🎯 Usage

### Launch the Framework
```bash
source venv/bin/activate
streamlit run nemesis_web_ui.py
```

**Alternative launch methods:**
```bash
# Using the launcher script (auto-detects environment)
./launch_nemesis.sh

# Direct Python execution (GUI mode)
python3 pray.py
```

### Key Components

#### AI Model Management
- Automatic model discovery and optimization
- Dynamic memory allocation based on hardware
- Fallback model support for reliability

#### Multi-Agent System
- Specialized agents for different security domains
- Coordinated task execution with LangGraph
- Real-time collaboration and data sharing

#### Security Testing Modules
- Network reconnaissance and vulnerability scanning
- Web application security testing
- Social engineering and OSINT operations
- Post-exploitation and lateral movement simulation

## 🔧 Configuration

### Hardware Optimization
The framework automatically detects and optimizes for your hardware:
- CPU cores and thread allocation
- Memory distribution for AI models
- Concurrent operation limits

### Model Configuration
Customize AI models in `NemesisConfig`:
```python
default_model: str = "huihui_ai/gemma3-abliterated:27b"
fallback_model: str = "dolphin-mixtral:8x7b"
embedding_model: str = "nomic-embed-text"
```

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 4 cores, 8 threads
- **RAM**: 16GB
- **Storage**: 50GB free space
- **OS**: Linux (Ubuntu 20.04+, Debian 11+)

### Recommended Configuration
- **CPU**: Intel i7-7800X or equivalent (6+ cores)
- **RAM**: 80GB+
- **GPU**: 6GB+ VRAM (optional, for acceleration)
- **Storage**: 100GB+ SSD

## 🛠️ Dependencies

### Core AI Framework
- `ollama` - Local LLM inference
- `langchain` - AI application framework
- `langchain-community` - Extended integrations
- `langgraph` - Multi-agent orchestration
- `crewai` - Collaborative AI agents

### Security Tools
- `python-nmap` - Network scanning
- `scapy` - Packet manipulation
- `dnspython` - DNS operations
- `python-whois` - Domain intelligence
- `paramiko` - SSH operations
- `cryptography` - Encryption utilities

### Web & OSINT
- `requests` - HTTP operations
- `beautifulsoup4` - Web scraping
- `selenium` - Browser automation
- `requests-cache` - Response caching

### Visualization & Reporting
- `matplotlib` - Data visualization
- `seaborn` - Statistical plots
- `reportlab` - PDF generation
- `weasyprint` - HTML to PDF
- `pillow` - Image processing

### GUI Framework
- `tkinter` - Native GUI framework
- `pyfiglet` - ASCII art generation
- `colorama` - Terminal colors

## 🔒 Security Considerations

### Responsible Use
- Only use against systems you own or have explicit permission to test
- Follow responsible disclosure practices
- Comply with local laws and regulations
- Document all testing activities

### Data Protection
- Automatic encryption of sensitive data
- Secure credential storage
- Session timeout mechanisms
- Automatic cleanup of temporary files

## 📈 Performance Optimization

### Memory Management
- Dynamic model loading based on available resources
- Intelligent caching of frequently used data
- Automatic garbage collection

### Parallel Processing
- Multi-threaded scanning operations
- Asynchronous AI model inference
- Concurrent agent execution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚡ Hardware Specifications

**Optimized for:**
- **CPU**: Intel i7-7800X (6 cores, 12 threads)
- **RAM**: 80GB DDR4
- **Chipset**: X299
- **GPU**: GTX 980 Ti (6GB VRAM)
- **OS**: Tsurugi Linux (Security-focused distribution)

## 🔗 Related Projects

- [Ollama](https://ollama.ai/) - Local LLM inference
- [LangChain](https://python.langchain.com/) - AI application framework
- [CrewAI](https://crewai.com/) - Multi-agent AI systems

## 📞 Support

For support and questions:
- Open an issue on GitHub
- Check the documentation
- Review existing discussions

---

**Remember: With great power comes great responsibility. Use this framework ethically and legally.**
