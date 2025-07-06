"""
setup_wizard.py - Interactive Setup and Secrets Wizard for Nemesis-Nexus
- Guides user through config settings (JSON/YAML)
- Stores secrets securely using OS keyring
- Validates config
"""
import json
import os
import getpass
from pathlib import Path
import keyring
import yaml

try:
    import pykwalify.core
    has_pykwalify = True
except ImportError:
    has_pykwalify = False

CONFIG_FIELDS = [
    ("ollama.host", "Ollama API host", "http://localhost:11434"),
    ("ollama.default_model", "Default LLM model", "codellama:13b-instruct"),
    ("ollama.temperature", "Model temperature", 0.7),
    ("ollama.max_tokens", "Max tokens per request", 2048),
    ("tools.max_concurrent_scans", "Max concurrent scans", 8)
]

SECRET_FIELDS = [
    ("OLLAMA_API_KEY", "API key for Ollama", "ollama"),
]

def prompt_with_default(prompt, default):
    suffix = f" [{default}]" if default else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val or default

def set_value_by_path(dct, path, value):
    keys = path.split('.')
    for k in keys[:-1]:
        dct = dct.setdefault(k, {})
    dct[keys[-1]] = value

def generate_config():
    print("--- Configuration ---")
    config = {}
    for key, prompt_txt, default in CONFIG_FIELDS:
        value = prompt_with_default(prompt_txt, default)
        set_value_by_path(config, key, value)
    return config

def save_config(config, path):
    p = Path(path)
    if path.lower().endswith('.json'):
        with open(p, 'w') as f:
            json.dump(config, f, indent=2)
    elif path.lower().endswith(('.yaml', '.yml')):
        with open(p, 'w') as f:
            yaml.safe_dump(config, f)
    print(f"Saved config to {path}")

def store_secrets():
    print("--- Credentials/Secrets ---")
    for name, prompt_txt, service in SECRET_FIELDS:
        val = getpass.getpass(f"{prompt_txt}: ")
        keyring.set_password("nemesis-nexus", name, val)
    print("All secrets stored securely in OS keyring.")

def validate_config_with_schema(config, schema_path):
    if not has_pykwalify:
        print("pykwalify not installed, skipping schema validation.")
        return True
    import pykwalify.core
    core = pykwalify.core.Core(source_data=config, schema_files=[schema_path])
    core.validate()
    print("Config validation succeeded.")
    return True

def main():
    print("Welcome to Nemesis-Nexus Setup Wizard!")
    out_format = prompt_with_default("Config format (json/yaml)", "json")
    out_file = f"config.{out_format}"
    config = generate_config()
    save_config(config, out_file)
    store_secrets()
    schema_path = prompt_with_default("Path to schema YAML for validation (optional)", "")
    if schema_path:
        try:
            validate_config_with_schema(config, schema_path)
        except Exception as e:
            print(f"Validation error: {e}")

if __name__ == "__main__":
    main()

