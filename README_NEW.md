# NEMESIS-NEXUS v2.0 - Consolidated Architecture

🚀 **Advanced Multi-Agent AI Cybersecurity Framework** - Now with clean, consolidated architecture!

## ⚠️ DISCLAIMER

**FOR AUTHORIZED PENETRATION TESTING ONLY**

This framework is designed exclusively for professional penetration testing and red team operations with proper authorization.

## 🔄 What's New in v2.0

### Consolidated Architecture
- **Single Entry Point**: `nemesis_nexus.py` - one script to rule them all
- **Clean Module Structure**: Organized into logical `core/`, `interfaces/`, `agents/`, and `utils/` directories
- **Auto-Detection**: Intelligent interface selection based on environment
- **Backwards Compatibility**: Legacy scripts redirect to new architecture

### Simplified Usage
```bash
# Auto-detect best interface
python nemesis_nexus.py

# Specific interface selection
python nemesis_nexus.py --cli              # Command-line interface
python nemesis_nexus.py --web              # Web interface (Streamlit)
python nemesis_nexus.py --api              # API interface (FastAPI)

# Legacy compatibility
python main.py                             # Redirects to nemesis_nexus.py
python pray.py                             # Still works (deprecated)
```

## 🏗️ New Architecture

```
nemesis-nexus/
├── nemesis_nexus.py           # Main entry point
├── launch.py                  # Legacy compatibility launcher
├── main.py                    # Legacy redirect
├── core/                      # Core framework components
│   ├── config.py              # Unified configuration
│   ├── logging_config.py      # Centralized logging
│   └── banner.py              # System info & dependencies
├── interfaces/                # User interfaces
│   ├── base_interface.py      # Common interface functionality
│   ├── cli_interface.py       # Interactive command-line
│   ├── web_interface.py       # Modern web UI (Streamlit)
│   └── api_interface.py       # REST API (FastAPI)
├── agents/                    # AI agent orchestration
│   └── orchestrator.py       # Multi-agent coordinator
├── utils/                     # System utilities
│   └── system_detection.py   # Environment detection
└── legacy/                    # Legacy files (preserved)
    ├── pray.py                # Original monolithic script
    ├── cli_interface.py       # Old CLI
    ├── nemesis_web_ui.py      # Streamlit UI
    └── nemesis_web_api.py     # FastAPI backend
```

## 🚀 Quick Start

### 1. Auto-Launch (Recommended)
```bash
python nemesis_nexus.py
```
The system will automatically detect your environment and start the best interface:
- **Desktop with display**: Web interface on port 8501
- **Terminal/SSH**: Interactive CLI
- **Headless/Server**: API interface on port 8000

### 2. Manual Interface Selection
```bash
# Command-line interface
python nemesis_nexus.py --cli

# Web interface (custom port)
python nemesis_nexus.py --web --port 8080

# API interface (custom host/port)
python nemesis_nexus.py --api --host 127.0.0.1 --port 9000
```

### 3. Check Dependencies
```bash
python nemesis_nexus.py --check-deps
```

## 🛠️ Features

### Unified Configuration
- **Auto-Hardware Detection**: Optimizes based on CPU/memory
- **Smart Model Selection**: Chooses best AI model for your system
- **Secure Storage**: Encrypted configuration and session data
- **Centralized Logging**: All components use unified logging

### Intelligent Interface Selection
- **Environment Detection**: X11, Wayland, SSH, headless support
- **Fallback Logic**: Gracefully handles missing dependencies
- **Cross-Platform**: Works on Linux, Windows, macOS

### Multi-Agent Orchestration
- **Simplified Agents**: Clean, modular agent architecture
- **Mission Coordination**: Automated security testing workflows
- **Real-time Status**: Monitor agent progress and results

## 📋 CLI Commands

The CLI interface provides interactive commands:

```
nemesis> help                           # Show all commands
nemesis> status                         # System and agent status
nemesis> mission example.com            # Run security assessment
nemesis> mission example.com --type recon  # Reconnaissance only
nemesis> agents list                    # List available agents
nemesis> config show                    # Display configuration
nemesis> exit                           # Exit application
```

## 🔧 Configuration

### Automatic Configuration
The system automatically detects and optimizes for your hardware:
- CPU cores and threading
- Available memory
- AI model selection
- Concurrent operation limits

### Manual Configuration
Configuration is stored in `~/.nemesis/config.yaml`:

```yaml
# Hardware limits
max_concurrent_agents: 8
max_concurrent_scans: 16

# AI model preferences  
default_model: "huihui_ai/gemma3-abliterated:27b"
llm_memory_percentage: 0.5

# Security settings
stealth_mode: true
session_timeout: 7200
```

## 🔒 Security Features

- **Encryption**: All sensitive data encrypted at rest
- **Session Management**: Automatic session timeouts
- **Access Control**: Interface-level security controls
- **Audit Logging**: Comprehensive operation logging

## 📊 Monitoring

### System Status
- Real-time hardware monitoring
- Agent status and health
- Mission progress tracking
- Resource utilization

### Logging
- Centralized log management
- Rotating log files
- Configurable log levels
- Structured logging format

## 🔄 Migration from v1.0

### Existing Scripts
All existing scripts continue to work:
- `python pray.py` → Continues to work (deprecated)
- `python main.py` → Redirects to new system
- Legacy arguments automatically mapped

### Data Migration
- Configuration automatically migrated
- Session data preserved
- Plugin compatibility maintained

## 🛡️ Best Practices

### Development
1. Use `nemesis_nexus.py` for new deployments
2. Test interface auto-detection with `--check-deps`
3. Use CLI for scripting and automation
4. Use Web UI for interactive operations
5. Use API for integration with other tools

### Production
1. Always verify authorization before testing
2. Use headless mode for server deployments
3. Monitor resource usage during operations
4. Regular backup of configuration and data
5. Keep dependencies updated

## 🤝 Contributing

### Code Organization
- Core functionality in `core/` directory
- Interface implementations in `interfaces/`
- Agent logic in `agents/`
- Utilities in `utils/`

### Adding Features
1. Follow existing module structure
2. Use centralized logging
3. Add appropriate error handling
4. Update documentation
5. Test across all interfaces

## 📞 Support

- **Issues**: Open GitHub issues for bugs
- **Documentation**: Check existing docs first
- **Security**: Report vulnerabilities privately

---

**Remember: Use this framework responsibly and legally. Always obtain proper authorization before testing.**
