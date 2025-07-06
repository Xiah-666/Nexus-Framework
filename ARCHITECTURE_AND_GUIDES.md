# Nemesis-Nexus v1.0 Documentation

## Architecture Overview

### Diagram
```
+-------------------------+
| Nemesis GUI/Web/CLI     |
+-----------+-------------+
            |
            v
+-------------------------+
|   Nemesis Orchestrator  |
|  (Multi-Agent Engine)   |
+---+-----+-----+-----+---+
    |     |     |     |
    v     v     v     v
+------+ +-----+ +---+ +----+
|Recon | | Expl| |Net| |Soc |
|Agent | |Agent| |Agent|Agent|
+------+ +-----+ +---+ +----+
    |       |     |      |
    +-------+-----+------+
               |
         +-----v-----+
         | Plugins   |
         +-----------+
               |
        +------v-------+
        | Tool APIs    |
        +-------------+
```

**Description:**
- **GUI/Web/CLI:** User interfaces (Tkinter UI, FastAPI, CLI Shell)
- **Orchestrator:** Controls agent spawning, workflow (LangGraph/CrewAI)
- **Agents:** ReconAgent, ExploitAgent, NetworkAgent, SocialAgent, IntelAgent
- **Plugins:** Drop-in tools (red-team, OSINT, C2) via `PluginInterface`
- **Tool APIs:** Nmap, Selenium, Scapy, LLMs, etc.

---
## Plugin Development & API Reference

### Plugin Interface
See `/home/xiah/tools/ai-ml/pray/plugin_interface.py`:

```python
class PluginBase(ABC):
    name: str
    category: str # 'red_team', 'osint', 'c2', etc.
    config: dict = {}

    def load(self): ...
    def unload(self): ...
    def configure(self, config: dict): ...
    def execute(self, **kwargs): ...
```

- Register custom plugins with `PluginRegistry.register_plugin()`.
- Plugins can be discovered automatically via setuptools entrypoints or explicit import.

**Example Plugin:**
```python
class ExampleOSINTPlugin(PluginBase):
    name = "whois_lookup_plugin"
    category = "osint"
    def load(self): ...
    def execute(self, **kwargs):
        # Return WHOIS data or other OSINT info
        pass
```

---
### API Reference
**Core API Endpoints** (`/home/xiah/tools/ai-ml/pray/nemesis_web_api.py`):

- `/plugin/exec` — Run any registered plugin by name
- `/agent/exec` — Instruct an agent to perform a task
- `/plugins`, `/agents` — List available plugins/agents

Interaction sample:
```json
POST /plugin/exec {"name": "whois_lookup_plugin", "options": {"domain": "example.com"}}
```

---
## Security Practices & Hardening Guide

- **Authorization:**
  - _Never_ test unauthorized systems
  - Use only with explicit written permission
- **Data Handling:**
  - Sensitive outputs encrypted (see `NemesisConfig.encryption_key`)
  - Use secure credential storage methods; avoid plaintext secrets
  - Maintain strict session timeouts and logs
- **System Hardening:**
  - Run on security-hardened Linux (recommended: Tsurugi Linux)
  - Use system firewall and network segmentation
  - Regularly apply system and LLM model updates
- **OPSEC Guidance:**
  - Clean up artifacts after every engagement
  - Use anonymizing utilities or airgapped VMs for higher-risk OPSEC
  - Log all actions for audit and reporting

---
## Red-Team/OSINT User & Workflow Guides

### Red-Team Assessment Workflow
1. **Mission Launch:** Launch via GUI/CLI/API. Specify target and mission type (e.g., Full Red Team, Phishing, Web Test).
2. **Reconnaissance:** ReconAgent automates domain, social, and OSINT.
3. **Vulnerability Analysis:** NetworkAgent/ExploitAgent perform scans and exploitation (Nmap, custom exploits, etc).
4. **Social Engineering:** SocialAgent runs spear-phishing, vishing, etc.
5. **Post-Exploitation & Reporting:** Results collected and PDF reports generated (with summary and details).
6. **Cleanup:** Ensures all traces and temporary files wiped.

### OSINT Operations
- Use ReconAgent and plugins (e.g., whois, DNS, certificate, social recon)
- Results are automatically structured in the system and accessibly via reports/UI/API
- All OSINT data is kept local; no 3rd-party storage

---
## Contribution & Issue-Reporting Guidelines

**Contributions:**
- Fork and branch the repo
- Follow core coding style & `PluginBase` practices
- Document new plugins clearly in `/docs/plugins/` and add comments
- Add and/or update tests under `/tests/`
- Submit PR with detailed info (context/usage)

**Reporting Issues:**
- Check open GitHub issues before submitting
- Clearly describe the bug or enhancement
- Include environment details and error traces if possible

**Security Bugs:**
- DO NOT disclose in public issues — contact project maintainer privately first
- Provide PoC and context _discreetly_

---
For more, see the main `README.md` and source code for up-to-date integration specifics.
