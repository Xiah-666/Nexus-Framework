# Model Selection Interface for Nemesis-Nexus

## Overview

This implementation adds a comprehensive model selection interface to the Nemesis-Nexus Streamlit Web UI, allowing users to:

- Select active LLM models from a user-friendly dropdown
- View real-time model status and availability
- Switch models per task and maintain session-wide model selection
- Download new models from the Ollama registry
- Monitor model usage and system information

## Features Implemented

### 1. Chat Copilot Model Selection

**Location:** Chat Copilot tab
**Features:**
- Real-time dropdown showing available models from Ollama
- Current model status indicator (🟢 Active / 🔴 Unavailable)
- One-click model switching with instant feedback
- Session persistence for selected model
- Model name displayed in chat responses

### 2. Dedicated Model Management Tab

**Location:** Model Management tab
**Features:**
- **Model Status Dashboard:** Shows currently active model
- **Available Models Table:** Displays all models with size, status, and modification date
- **Model Switching:** Dropdown selection with one-click switching
- **Model Download:** Pre-configured buttons for popular cybersecurity models
- **Custom Model Download:** Text input for any Ollama registry model
- **System Information:** Backend status and configuration details

### 3. Sidebar Model Indicator

**Location:** Left sidebar (all tabs)
**Features:**
- Always-visible current model indicator
- Real-time status updates
- Direct link to Model Management tab

### 4. API Endpoints

**New Endpoints Added:**
- `GET /models` - List available models from Ollama
- `GET /models/current` - Get currently selected model
- `POST /models/select` - Switch active model
- `POST /models/download` - Download model from registry

## Technical Implementation

### Backend (FastAPI)

```python
# Model management endpoints in nemesis_web_api.py
@app.get("/models")  # Get available models
@app.get("/models/current")  # Get current model
@app.post("/models/select")  # Switch model
@app.post("/models/download")  # Download model
```

### Frontend (Streamlit)

```python
# Model selection dropdown in Chat Copilot
selected_model = st.selectbox("🧠 Active LLM Model", options=available_models)

# Model management interface
with main_tabs[3]:  # Model Management tab
    st.header("🧠 Model Management")
    # Full model management interface
```

### Session State Management

```python
# Persistent model selection across page interactions
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
```

## Usage Instructions

### For End Users

1. **Quick Model Switch (Chat Copilot):**
   - Select model from dropdown in Chat Copilot tab
   - Model switches immediately for current session
   - All chat responses will use selected model

2. **Advanced Model Management:**
   - Go to "Model Management" tab
   - View all available models in detailed table
   - Switch models using dropdown + button
   - Download new models using quick buttons or custom input

3. **Model Status Monitoring:**
   - Check sidebar for current active model
   - Green indicator = model active and ready
   - Red indicator = model unavailable or error

### For Developers

1. **Adding New Models:**
   ```bash
   # Download via UI or command line
   ollama pull model-name:tag
   ```

2. **API Integration:**
   ```python
   # Get models programmatically
   models = requests.get("http://localhost:8000/models").json()
   
   # Switch model programmatically  
   requests.post("http://localhost:8000/models/select", 
                json={"model_name": "new-model"})
   ```

## Supported Models

### Pre-configured Models (Quick Download)
- `dolphin-mixtral:8x7b` - Uncensored Mixtral for cybersecurity
- `nous-hermes-2-mixtral-8x7b-dpo` - Red team operations
- `codellama:34b-instruct` - Code generation for security tools
- `wizard-vicuna-30b-uncensored` - Penetration testing specialist
- `blacksheep:latest` - Lightweight uncensored model

### Custom Models
- Any model available in Ollama registry
- Enter exact model name in custom download field
- Models download in background (check periodically)

## Configuration

### Requirements
```txt
streamlit
pandas
fastapi
uvicorn
requests
ollama
```

### Environment Setup
1. Ensure Ollama is running: `ollama serve`
2. Start API backend: `python nemesis_web_api.py`
3. Start Streamlit UI: `streamlit run nemesis_web_ui.py`

## Testing

Run the test suite to verify functionality:
```bash
python test_model_selection.py
```

The test suite verifies:
- Ollama connection
- Model listing endpoints
- Model selection functionality
- LLM chat with selected models

## Security Considerations

- **Authorization:** Model management requires operator/admin roles
- **Model Downloads:** Only admins can download new models
- **Session Isolation:** Model selection is per-session
- **Audit Logging:** All model operations are logged

## Troubleshooting

### Common Issues

1. **"No models available"**
   - Check Ollama is running: `ollama serve`
   - Verify models are installed: `ollama list`
   - Download a model: `ollama pull llama2`

2. **"API Unavailable"**
   - Ensure FastAPI backend is running
   - Check API_URL in nemesis_web_ui.py
   - Verify no firewall blocking localhost:8000

3. **Model switch not working**
   - Refresh the page
   - Check model exists in Ollama
   - Verify user has operator/admin role

### Debug Commands

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Check API status  
curl http://localhost:8000/models

# List installed models
ollama list
```

## Future Enhancements

Potential improvements for future versions:
- Model performance metrics
- Auto-refresh model list
- Model usage statistics
- Bulk model operations
- Model presets for different tasks
- Resource usage monitoring
- Model comparison interface

## API Reference

### GET /models
Returns list of available models from Ollama.

**Response:**
```json
{
  "models": [
    {
      "name": "llama2:latest",
      "size": 3825819519,
      "status": "available",
      "modified_at": "2024-01-15T10:30:00Z",
      "digest": "sha256:..."
    }
  ]
}
```

### POST /models/select
Select active model for session.

**Request:**
```json
{
  "model_name": "llama2:latest"
}
```

**Response:**
```json
{
  "status": "success",
  "selected_model": "llama2:latest"
}
```

## License

This model selection interface is part of the Nemesis-Nexus project and follows the same licensing terms. For authorized cybersecurity research and penetration testing only.
