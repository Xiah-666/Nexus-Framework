# Cross-Platform Setup and Configuration Instructions for Nemesis-Nexus

## 1. Installation Scripts

### 1.1. Linux/Mac Installation
Run the provided `/home/xiah/tools/ai-ml/pray/install.sh` script:

```sh
bash ./install.sh
```
This will:
- Create and activate a Python virtual environment (`venv`)
- Install dependencies from `requirements.txt`
- Install/enable Ollama, check service
- Create a default `config.json` from template

### 1.3. Quick Start (Streamlit Web UI)
For the fastest setup, use the quick-start script:
```bash
./quickstart.sh
```
Or launch Streamlit directly:
```bash
streamlit run nemesis_web_ui.py
```

**Note:** Streamlit is now the default launch method for NEMESIS-NEXUS, providing a web-based interface accessible from any browser.

### 1.4. Windows Installation
A Powershell script is recommended (see `install_windows.ps1`):
```powershell
# Example install_windows.ps1
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
# TODO: Windows Ollama install
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Output "Install instructions for Ollama here."
}
# Copy config template if missing
if (!(Test-Path "config.json")) { Copy-Item "config.json.sample" "config.json" }
```

## 2. Docker/Compose Support
Add `Dockerfile` and (if required) `docker-compose.yml` for containerized operation. Example snippets:

**Dockerfile:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt
# Default to Streamlit web interface
EXPOSE 8501
CMD ["streamlit", "run", "nemesis_web_ui.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

**docker-compose.yml:**
```yaml
version: '3.9'
services:
  nexus:
    build: .
    volumes:
      - ./config.json:/app/config.json
    ports:
      - "8501:8501" # Streamlit Web UI
      - "11434:11434" # Ollama
    environment:
      - OLLAMA_HOST=http://localhost:11434
      - NEMESIS_MODE=docker
```

## 3. Configuration Management
- Primary config in `config.json`. For YAML: create `config.yaml` (identical structure, can convert using PyYAML).
- Validation: use `pykwalify` or `pySchema` for schema checks in CLI/GUI startup.
- **Sensitive data:** Never store cleartext API keys/passwords in config files! Use secure secrets backends.

Example `config.json`/`config.yaml` sections:
```json
{
  "version": "1.0",
  "ollama": {"host": "http://localhost:11434"},
  "tools": {"max_concurrent_scans": 8}
}
```

## 4. Interactive Setup Wizard
Recommend an interactive Python CLI wizard (`setup_wizard.py`) to:
- Prompt user for API endpoints, concurrency, options for agent models
- Write config.json/config.yaml
- Integrate with OS keyring or Vault for secret input
- Validate configuration with pykwalify

## 5. Secure Credential Storage
- Use the `keyring` Python library for storing/retrieving credentials from OS keychains (Windows, Mac, Linux)
- Example usage in config loader/wizard:
```python
import keyring
# To save:
keyring.set_password('nemesis-nexus', 'OLLAMA_API_KEY', 'the-key-value')
# To retrieve:
api_key = keyring.get_password('nemesis-nexus', 'OLLAMA_API_KEY')
```
- For advanced security, support HashiCorp Vault etc. as a backend via plug-in.

---

### Summary of What To Add:
- [ ] Add `install_windows.ps1` (see above for starter)
- [ ] Add `Dockerfile` and optionally `docker-compose.yml`
- [ ] Add `setup_wizard.py` (Python, interactive, stores secrets securely)
- [ ] Integrate pykwalify validation in config load path
- [ ] Store ALL sensitive values using OS keyring/Vault
- [ ] Mention config YAML as option for advanced users.

# End of Checklist

