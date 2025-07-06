import json
import yaml
from pathlib import Path
from typing import Dict, Any

class AliasManager:
    SUPPORTED_TYPES = {"payloads", "environments", "servers", "targets"}
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._aliases = {k: {} for k in self.SUPPORTED_TYPES}
        self.load()

    def load(self):
        if self.config_path.exists():
            if str(self.config_path).endswith('.json'):
                with open(self.config_path, 'r') as f:
                    raw = json.load(f)
            else:
                with open(self.config_path, 'r') as f:
                    raw = yaml.safe_load(f)
            self._aliases = raw.get('aliases', self._aliases)
        else:
            self._aliases = {k: {} for k in self.SUPPORTED_TYPES}

    def save(self):
        out = {"aliases": self._aliases}
        if str(self.config_path).endswith('.json'):
            with open(self.config_path, 'w') as f:
                json.dump(out, f, indent=2)
        else:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(out, f)

    def create_alias(self, kind: str, alias: str, value: Any):
        assert kind in self.SUPPORTED_TYPES, f"Alias type {kind} not supported."
        self._aliases[kind][alias] = value
        self.save()
    
    def view_aliases(self, kind: str = None):
        if kind:
            assert kind in self.SUPPORTED_TYPES, f"Alias type {kind} not supported."
            return self._aliases.get(kind, {})
        return self._aliases
    
    def delete_alias(self, kind: str, alias: str):
        assert kind in self.SUPPORTED_TYPES, f"Alias type {kind} not supported."
        if alias in self._aliases[kind]:
            del self._aliases[kind][alias]
            self.save()
            return True
        return False

    def import_aliases(self, import_path: str):
        """Import aliases from a JSON or YAML file, overwrite current ones."""
        path = Path(import_path)
        if path.suffix == '.json':
            with open(path, 'r') as f:
                imported = json.load(f)
        else:
            with open(path, 'r') as f:
                imported = yaml.safe_load(f)
        if 'aliases' in imported:
            self._aliases = imported['aliases']
            self.save()
            return True
        return False
    
    def export_aliases(self, export_path: str):
        """Export aliases to a JSON or YAML file."""
        path = Path(export_path)
        out = {"aliases": self._aliases}
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(out, f, indent=2)
        else:
            with open(path, 'w') as f:
                yaml.safe_dump(out, f)
        return True

    @staticmethod
    def validate_schema(aliases, schema={'payloads':dict, 'environments':dict, 'servers':dict, 'targets':dict}):
        for kind, typename in schema.items():
            if kind in aliases and not isinstance(aliases[kind], typename):
                return False
        return True

