# nemesis_web_ui.py - Streamlit Web UI consuming FastAPI backend
import streamlit as st
import requests

API_URL = "http://localhost:8000"
st.set_page_config(page_title="Nemesis-Nexus Web UI", layout="wide")

st.title("Nemesis-Nexus 🌐 - Modern Multi-Agent Cybersecurity Platform")
st.markdown(">**For authorized penetration testing only!**")

# Initialize session state for model management
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "model_switch_success" not in st.session_state:
    st.session_state.model_switch_success = False

# Add current model indicator in sidebar
with st.sidebar:
    st.header("🧠 Current Model")
    try:
        current_resp = requests.get(f"{API_URL}/models/current")
        if current_resp.status_code == 200:
            current_model_name = current_resp.json()["current_model"]
            st.success(f"🟢 **{current_model_name}**")
        else:
            st.error("🔴 Model Unknown")
    except:
        st.warning("⚠️ API Unavailable")
    
    st.divider()
    st.info("📊 Switch models in the Model Management tab for session-wide changes.")

main_tabs = st.tabs([
    "Chat Copilot",
    "Agent Orchestration",
    "Plugins",
    "Model Management",
    "Alias Manager",
    "Credential Vault",
    "Visualization"
])

# ---- CHAT COPILOT PANEL ----
with main_tabs[0]:
    st.header("🤖 Chat Copilot (LLM Assistance)")
    
    # Model Selection Section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Get available models
        try:
            models_resp = requests.get(f"{API_URL}/models")
            if models_resp.status_code == 200:
                available_models = [model["name"] for model in models_resp.json()["models"]]
            else:
                available_models = ["No models available"]
        except:
            available_models = ["Error loading models"]
        
        # Get current model
        try:
            current_resp = requests.get(f"{API_URL}/models/current")
            if current_resp.status_code == 200:
                current_model = current_resp.json()["current_model"]
            else:
                current_model = available_models[0] if available_models else "None"
        except:
            current_model = available_models[0] if available_models else "None"
        
        # Model selection dropdown
        selected_model = st.selectbox(
            "🧠 Active LLM Model",
            options=available_models,
            index=available_models.index(current_model) if current_model in available_models else 0,
            help="Select the LLM model for chat responses"
        )
        
        # Update model if changed
        if selected_model != current_model and selected_model not in ["No models available", "Error loading models"]:
            try:
                update_resp = requests.post(f"{API_URL}/models/select", json={"model_name": selected_model})
                if update_resp.status_code == 200:
                    st.success(f"✅ Switched to {selected_model}")
                    st.session_state.selected_model = selected_model
                    st.session_state.model_switch_success = True
                    st.rerun()
                else:
                    st.error("Failed to switch model")
            except Exception as e:
                st.error(f"Error switching model: {e}")
        
        # Use session state model if available
        active_model = st.session_state.selected_model or selected_model
    
    with col2:
        # Model status indicator
        if selected_model not in ["No models available", "Error loading models"]:
            st.metric("Model Status", "🟢 Active", help="Current model is ready for inference")
        else:
            st.metric("Model Status", "🔴 Unavailable", help="No models available")
    
    with col3:
        # Refresh models button
        if st.button("🔄 Refresh Models", help="Refresh the list of available models"):
            st.rerun()
    
    st.divider()
    
    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for h in st.session_state.chat_history:
        with st.chat_message(h["role"]):
            st.write(h["content"])
    user_prompt = st.chat_input("Ask anything, or describe a workflow...")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.spinner(f"LLM generating with {active_model}..."):
            try:
                # Include selected model in request
                chat_payload = {
                    "prompt": user_prompt,
                    "model": active_model if active_model not in ["No models available", "Error loading models"] else None
                }
                resp = requests.post(f"{API_URL}/llm/chat", json=chat_payload)
                result_json = resp.json()
                ai_response = result_json.get("response", "[no answer]")
            except Exception as e:
                ai_response = f"[Error: {e}]"
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant"):
            st.write(ai_response)
            # Show which model was used
            st.caption(f"Response generated by: {active_model}")
    st.divider()
    st.subheader("Copilot Suggestions")
    suggestions = [
        "Scan target network for live hosts",
        "List all available agents",
        "Find and exploit web vulnerabilities",
        "Generate OSINT intelligence report",
        "Show recent agent status",
        "Export results as markdown"
    ]
    for ss in suggestions:
        st.button(f"💡 {ss}")

# ---- AGENT ORCHESTRATION PANEL ----
with main_tabs[1]:
    st.header("Live Agent Orchestration")
    st.write("Control and monitor agents, issue tasks, see real-time status.")
    try:
        ag_resp = requests.get(f"{API_URL}/agents")
        agents = ag_resp.json()["agents"]
    except Exception as e:
        agents = []
        st.error(f"Could not fetch agent list: {e}")
    agent_sel = st.selectbox("Select agent", agents)
    task = st.text_area("Enter command or task for agent")
    run_agent = st.button("Run Agent Task")
    agent_status_placeholder = st.empty()
    st.subheader("Queued/Running Operations")
    ops = []
    try:
        resp = requests.get(f"{API_URL}/status/live")
        ops = resp.json().get("active_ops", [])
    except Exception as e:
        st.warning(f"Could not fetch live ops: {e}")
    for op in ops:
        with st.expander(f"Op {op.get('id', 'unknown')} [{op.get('status')}] :: {op.get('agent', op.get('plugin', ''))}"):
            st.json(op)
            if op.get('status') in ("queued", "running"):
                if st.button("Kill", key=f"kill_{op['id']}"):
                    requests.post(f"{API_URL}/ops/{op['id']}/kill")
                    st.info('Operation kill requested.')
            if op.get('status') in ("completed", "error", "killed"):
                if st.button("Retask", key=f"retask_{op['id']}"):
                    # Only a demo. Should pop up modal for input, use last task/context
                    requests.post(f"{API_URL}/ops/{op['id']}/retask", json={})
                    st.success('Retask requested.')
            if st.button("Show Logs", key=f"logs_{op['id']}"):
                logs_resp = requests.get(f"{API_URL}/ops/{op['id']}/logs")
                st.json(logs_resp.json())

    if run_agent and agent_sel and task:
        with st.status("Submitting task to agent...") as stat:
            try:
                # Submit to queue, not direct exec for async
                resp = requests.post(f"{API_URL}/ops/queue/agent", json={"agent": agent_sel, "task": task})
                output = resp.json()
                stat.update(label="Agent task queued.", state="complete")
                agent_status_placeholder.json(output)
            except Exception as e:
                stat.update(label=f"Error: {e}", state="error")
    st.info("Operations will show above. Refresh to update status/logs.")

# ---- PLUGIN EXECUTION PANEL ----
with main_tabs[2]:
    st.header("Plugin Execution")
    try:
        pl_resp = requests.get(f"{API_URL}/plugins")
        plugins = pl_resp.json()["plugins"]
    except Exception as e:
        plugins = []
        st.error(f"Could not fetch plugin list: {e}")
    plugin_sel = st.selectbox("Select a plugin", plugins)
    options = st.text_input("Plugin options (key=value, space separated)")
    if st.button("Run Plugin"):
        opts_dict = dict(opt.split('=') for opt in options.split() if '=' in opt)
        # Submit to async op queue
        resp = requests.post(f"{API_URL}/ops/queue/plugin", json={"name": plugin_sel, "options": opts_dict})
        st.write(resp.json())

# ---- MODEL MANAGEMENT PANEL ----
with main_tabs[3]:
    st.header("🧠 Model Management")
    st.info("Manage LLM models, view status, and download new models for the Nemesis-Nexus platform.")
    
    # Current model status
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Model Status")
        try:
            # Get current model
            current_resp = requests.get(f"{API_URL}/models/current")
            if current_resp.status_code == 200:
                current_model = current_resp.json()["current_model"]
                st.success(f"✅ **Active Model:** {current_model}")
            else:
                st.error("❌ Could not retrieve current model")
        except Exception as e:
            st.error(f"❌ Error getting current model: {e}")
    
    with col2:
        if st.button("🔄 Refresh Status", help="Refresh model status and list"):
            st.rerun()
    
    st.divider()
    
    # Available models section
    st.subheader("📋 Available Models")
    
    try:
        models_resp = requests.get(f"{API_URL}/models")
        if models_resp.status_code == 200:
            models_data = models_resp.json()["models"]
            
            if models_data:
                # Create model selection interface
                model_names = [model["name"] for model in models_data]
                
                # Model selection dropdown
                selected_model_for_switch = st.selectbox(
                    "🔄 Switch Active Model",
                    options=model_names,
                    help="Select a model to make it active for all operations"
                )
                
                # Switch model button
                if st.button("🔄 Switch Model", type="primary"):
                    try:
                        switch_resp = requests.post(f"{API_URL}/models/select", json={"model_name": selected_model_for_switch})
                        if switch_resp.status_code == 200:
                            st.success(f"✅ Successfully switched to {selected_model_for_switch}")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Failed to switch model")
                    except Exception as e:
                        st.error(f"❌ Error switching model: {e}")
                
                st.divider()
                
                # Display models in a table format
                st.subheader("📊 Model Details")
                
                # Create dataframe for model display
                import pandas as pd
                
                model_table_data = []
                for model in models_data:
                    size_mb = model.get('size', 0)
                    size_gb = f"{size_mb / (1024**3):.1f} GB" if size_mb > 0 else "Unknown"
                    
                    # Determine if this is the current model
                    is_current = "🟢 Active" if model["name"] == current_model else "⚪ Available"
                    
                    model_table_data.append({
                        "Model Name": model["name"],
                        "Status": is_current,
                        "Size": size_gb,
                        "Last Modified": model.get("modified_at", "Unknown")[:19] if model.get("modified_at") else "Unknown"
                    })
                
                # Display as dataframe
                df = pd.DataFrame(model_table_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
            else:
                st.warning("⚠️ No models available. Please download models using the section below.")
        else:
            st.error("❌ Could not retrieve model list from Ollama backend")
    except Exception as e:
        st.error(f"❌ Error retrieving models: {e}")
    
    st.divider()
    
    # Model download section
    st.subheader("⬇️ Download New Models")
    st.info("💡 Popular uncensored models for cybersecurity research:")
    
    # Predefined model suggestions
    suggested_models = [
        {"name": "dolphin-mixtral:8x7b", "description": "Uncensored Mixtral model for cybersecurity"},
        {"name": "nous-hermes-2-mixtral-8x7b-dpo", "description": "Excellent for red team operations"},
        {"name": "codellama:34b-instruct", "description": "Code generation for security tools"},
        {"name": "wizard-vicuna-30b-uncensored", "description": "Specialized for penetration testing"},
        {"name": "blacksheep:latest", "description": "Lightweight uncensored model"}
    ]
    
    # Quick download buttons for suggested models
    st.write("**Quick Download (Suggested Models):**")
    cols = st.columns(3)
    for i, model in enumerate(suggested_models[:3]):
        with cols[i]:
            if st.button(f"⬇️ {model['name']}", help=model['description'], key=f"quick_{i}"):
                try:
                    download_resp = requests.post(f"{API_URL}/models/download", json={"model_name": model['name']})
                    if download_resp.status_code == 200:
                        st.success(f"✅ Download started for {model['name']}")
                        st.info("🔄 Download is running in background. Check model list for updates.")
                    else:
                        st.error(f"❌ Failed to start download: {download_resp.text}")
                except Exception as e:
                    st.error(f"❌ Error starting download: {e}")
    
    # Custom model download
    st.write("**Custom Model Download:**")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        custom_model = st.text_input(
            "Model Name",
            placeholder="e.g., dolphin-mixtral:8x7b",
            help="Enter the exact model name from Ollama registry"
        )
    
    with col2:
        if st.button("⬇️ Download", type="primary", disabled=not custom_model.strip()):
            try:
                download_resp = requests.post(f"{API_URL}/models/download", json={"model_name": custom_model.strip()})
                if download_resp.status_code == 200:
                    st.success(f"✅ Download started for {custom_model.strip()}")
                    st.info("🔄 Download is running in background. This may take several minutes depending on model size.")
                else:
                    st.error(f"❌ Failed to start download: {download_resp.text}")
            except Exception as e:
                st.error(f"❌ Error starting download: {e}")
    
    st.divider()
    
    # System information
    st.subheader("🖥️ System Information")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.metric("Ollama Status", "🟢 Running", help="Ollama backend service status")
        st.metric("API Endpoint", API_URL, help="Backend API endpoint")
    
    with info_col2:
        st.metric("Model Backend", "Ollama", help="LLM backend service")
        st.metric("Session Model", current_model if 'current_model' in locals() else "Unknown", help="Currently active model for this session")
    
    # Warning disclaimer
    st.warning("⚠️ **Security Notice:** These models are designed for authorized security research and penetration testing only. Use responsibly and in compliance with applicable laws and regulations.")

# ---- ALIAS MANAGER PANEL ----
with main_tabs[4]:
    st.header("Alias Manager")
    st.info("Create, view, and delete aliases for payloads, environments, servers, and targets.")
    try:
        alias_resp = requests.get(f"{API_URL}/aliases")
        aliases = alias_resp.json()["aliases"]
    except Exception as e:
        aliases = {}
        st.error(f"Could not fetch aliases: {e}")
    st.subheader("Current Aliases")
    st.json(aliases)
    alias_types = ["payloads", "environments", "servers", "targets"]
    st.subheader("Create Alias")
    alias_type = st.selectbox("Type", alias_types)
    alias_name = st.text_input("Alias Name")
    alias_value = st.text_input("Alias Value (string or simple)")
    if st.button("Create Alias"):
        payload = {"kind": alias_type, "name": alias_name, "value": alias_value}
        resp = requests.post(f"{API_URL}/alias/create", params=payload)
        st.success(str(resp.json()))
    st.subheader("Delete Alias")
    del_type = st.selectbox("Type (delete)", alias_types, key="deltype")
    del_name = st.text_input("Alias Name (delete)", key="delname")
    if st.button("Delete Alias"):
        resp = requests.post(f"{API_URL}/alias/delete", params={"kind": del_type, "name": del_name})
        st.warning(str(resp.json()))

# ---- CREDENTIAL VAULT PANEL ----
from ui_panels import credential_vault_panel, realtime_visualization_panel  # new modular panel imports
with main_tabs[5]:
    credential_vault_panel.render()

# ---- VISUALIZATION/RESULTS PANEL ----
with main_tabs[6]:
    realtime_visualization_panel.render()
