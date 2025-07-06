import pytest
import yaml
import os

CONFIG_SCHEMA = 'config_schema.yml'
CONFIG_FILE = 'config.yaml'

REQUIRED_OPTIONS = [
    'llm',
    'plugins',
]

def load_schema():
    with open(CONFIG_SCHEMA) as f:
        return yaml.safe_load(f)

def load_config():
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)

def test_core_config_hardened():
    schema = load_schema()
    config = load_config()
    for field in REQUIRED_OPTIONS:
        assert field in config
    # Assert secrets are not hardcoded
    secrets = ['token', 'password', 'secret']
    flat_config = str(config).lower()
    assert not any(v in flat_config for v in ['changeme', '1234', 'password', 'secret'])
    for k in config:
        for s in secrets:
            assert s not in str(config[k]).lower() or '***' in str(config[k])
