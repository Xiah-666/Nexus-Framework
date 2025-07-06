import streamlit as st
import requests
import json
from cryptography.fernet import Fernet

API_URL = "http://localhost:8000"

# Panel - Credential Vault
class CredentialVaultUI:
    def __init__(self, api_url):
        self.api_url = api_url

    def render(self):
        st.header("🔒 Credential Vault")
        st.markdown(
            "Manage credentials securely with encryption and role-based access. Create, view, edit, and delete secrets."
        )

        action = st.radio("Action", ["View", "Create", "Edit", "Delete"])
        token = st.text_input("API Access Token", type="password")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        if action == "View":
            if st.button("Load Credentials"):
                resp = requests.get(f"{self.api_url}/vault/credentials", headers=headers)
                if resp.ok:
                    creds = resp.json().get("credentials", [])
                    for cred in creds:
                        st.json(cred)
                else:
                    st.error(resp.text)
        elif action == "Create":
            new_domain = st.text_input("Domain/App Name")
            new_user = st.text_input("Username")
            new_secret = st.text_input("Password/Secret", type="password")
            if st.button("Store Credential"):
                payload = {"domain": new_domain, "username": new_user, "secret": new_secret}
                resp = requests.post(f"{self.api_url}/vault/credentials", json=payload, headers=headers)
                st.success(str(resp.json()) if resp.ok else resp.text)
        elif action == "Edit":
            cred_id = st.text_input("Credential ID")
            edit_field = st.selectbox("Field to edit", ["domain", "username", "secret"])
            new_val = st.text_input(f"New value for {edit_field}")
            if st.button("Edit Credential"):
                payload = {"field": edit_field, "value": new_val}
                resp = requests.put(f"{self.api_url}/vault/credentials/{cred_id}", json=payload, headers=headers)
                st.success(str(resp.json()) if resp.ok else resp.text)
        elif action == "Delete":
            cred_id = st.text_input("Credential ID (delete)")
            if st.button("Delete Credential"):
                resp = requests.delete(f"{self.api_url}/vault/credentials/{cred_id}", headers=headers)
                st.warning(str(resp.json()) if resp.ok else resp.text)

# Panel - Real-Time Visualization
class RealTimeVisualizationUI:
    def __init__(self, api_url):
        self.api_url = api_url

    def render(self):
        st.header("📈 Real-Time Visualization (Live Ops Status)")
        st.markdown(
            "Live monitoring dashboard for agents, plugins, task activity, and health via streaming updates."
        )
        # Fetch live agent/plugin status
        if st.button("Refresh Status"):
            resp = requests.get(f"{self.api_url}/status/live")
            if resp.ok:
                data = resp.json()
                st.subheader("Agents")
                st.json(data.get("agents", {}))
                st.subheader("Plugins")
                st.json(data.get("plugins", {}))
                st.subheader("Active Ops/Tasks (Queued/Running)")
                for op in data.get("active_ops", []):
                    with st.expander(f"Op {op.get('id', 'unknown')} [{op.get('status')}] {op.get('agent', op.get('plugin',''))}"):
                        st.json(op)
                        if st.button("Show Logs", key=f"panel_logs_{op['id']}"):
                            logs_resp = requests.get(f"{self.api_url}/ops/{op['id']}/logs")
                            st.json(logs_resp.json())
            else:
                st.error(resp.text)
        # For more advanced streaming, use websocket in future.
        st.info("Streaming real-time updates requires websocket/async support.")

# Exportable classes for usage in Streamlit main file
credential_vault_panel = CredentialVaultUI(API_URL)
realtime_visualization_panel = RealTimeVisualizationUI(API_URL)

