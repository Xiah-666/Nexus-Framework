# Plugin Extension Guide

## Overview
This framework supports rapid red-team and C2 plugin integration. Each plugin derives from `PluginBase` and must implement the following lifecycle methods: `load`, `unload`, `configure`, `execute`. The execute interface must accept normalized operations:

- `scan(target)`
- `launch_payload(exploit, target, payload, ...)`
- `report(scan_id|operation_id)`

## Adding a New Plugin
1. **Create a Class** inheriting from `PluginBase`.
2. **Implement** required lifecycle and normalization method(s).
3. **Register** your plugin using `PluginRegistry.register_plugin()`.
4. **Support** secure API/CLI interaction and stateful session management as needed.

### Example Skeleton
```python
from plugin_interface import PluginBase

class MyToolPlugin(PluginBase):
    name = "my_tool"
    category = "red_team"

    def load(self):
        # Initialize API/CLI session here
        self.loaded = True

    def unload(self):
        # Clean up session/resources here
        self.loaded = False

    def configure(self, config: dict):
        self.config = config

    def execute(self, operation: str, **kwargs):
        if operation == 'scan':
            return self.scan(kwargs.get('target'))
        elif operation == 'launch_payload':
            return self.launch_payload(...)
        elif operation == 'report':
            return self.report(...)
        else:
            return {"error": f"Unknown operation: {operation}"}
```

## Normalized Operations
- `scan(target: str)` — Perform a scan.
- `launch_payload(...)` — Launch payload/exploit. Args vary based on integration.
- `report(id)` — Generate or fetch a report.

## Best Practices
- **Handle credentials via env variables/config, never hard code.**
- **Fail gracefully and return normalized error dicts.**
- **Apply thread-safety for async or multiagent use.**

