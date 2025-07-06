from plugin_interface import PluginBase
import requests
import subprocess
import logging

class MetasploitProPlugin(PluginBase):
    name = "metasploit_pro"
    category = "red_team"

    def load(self):
        self.session = requests.Session()
        self.api_url = self.config.get('api_url', 'https://localhost:3790/api/v1/')
        self.api_token = self.config.get('api_token')
        self.session.headers.update({'Authorization': f'Bearer {self.api_token}'})
        self.loaded = True
        logging.info("Metasploit Pro plugin loaded")

    def unload(self):
        self.loaded = False
        self.session.close()

    def configure(self, config: dict):
        self.config = config
        self.api_url = config.get('api_url', self.api_url)
        self.api_token = config.get('api_token', self.api_token)
        self.session.headers.update({'Authorization': f'Bearer {self.api_token}'})

    def execute(self, operation: str, **kwargs):
        if operation == 'scan':
            return self.scan(kwargs.get('target'))
        elif operation == 'launch_payload':
            return self.launch_payload(kwargs.get('exploit'), kwargs.get('target'), kwargs.get('payload'), kwargs)
        elif operation == 'report':
            return self.generate_report(kwargs.get('report_id'))
        else:
            return {"error": f"Unknown operation: {operation}"}

    def scan(self, target):
        resp = self.session.post(f"{self.api_url}/scans", json={"target": target})
        return resp.json()

    def launch_payload(self, exploit, target, payload, extra_params=None):
        params = {
            "exploit": exploit,
            "target": target,
            "payload": payload,
            **(extra_params or {})
        }
        resp = self.session.post(f"{self.api_url}/payloads/launch", json=params)
        return resp.json()

    def generate_report(self, report_id):
        resp = self.session.get(f"{self.api_url}/reports/{report_id}")
        return resp.json()


class NessusProPlugin(PluginBase):
    name = "nessus_pro"
    category = "red_team"

    def load(self):
        self.session = requests.Session()
        self.api_url = self.config.get('api_url', 'https://localhost:8834')
        self.api_token = self.config.get('api_token')
        self.session.headers.update({'X-ApiKeys': f'accessKey={self.api_token}; secretKey='})
        self.loaded = True
        logging.info("Nessus Pro plugin loaded")

    def unload(self):
        self.loaded = False
        self.session.close()

    def configure(self, config: dict):
        self.config = config
        self.api_url = config.get('api_url', self.api_url)
        self.api_token = config.get('api_token', self.api_token)
        self.session.headers.update({'X-ApiKeys': f'accessKey={self.api_token}; secretKey='})

    def execute(self, operation: str, **kwargs):
        if operation == 'scan':
            return self.scan(kwargs.get('target'))
        elif operation == 'report':
            return self.report(kwargs.get('scan_id'))
        else:
            return {"error": f"Unknown operation: {operation}"}

    def scan(self, target):
        resp = self.session.post(f"{self.api_url}/scans", json={"targets": target})
        return resp.json()

    def report(self, scan_id):
        resp = self.session.get(f"{self.api_url}/scans/{scan_id}/export")
        return resp.json()


class BurpSuiteProPlugin(PluginBase):
    name = "burp_suite_pro"
    category = "red_team"

    def load(self):
        self.api_url = self.config.get('api_url', 'http://localhost:1337/v0.1')
        self.loaded = True
        logging.info("Burp Suite Pro plugin loaded")

    def unload(self):
        self.loaded = False

    def configure(self, config: dict):
        self.config = config
        self.api_url = config.get('api_url', self.api_url)

    def execute(self, operation: str, **kwargs):
        if operation == 'scan':
            return self.scan(kwargs.get('target'))
        elif operation == 'report':
            return self.report(kwargs.get('scan_id'))
        else:
            return {"error": f"Unknown operation: {operation}"}

    def scan(self, target):
        resp = requests.post(f"{self.api_url}/scan", json={"urls": [target]})
        return resp.json()

    def report(self, scan_id):
        resp = requests.get(f"{self.api_url}/scan/{scan_id}/report")
        return resp.json()


class CobaltStrikePlugin(PluginBase):
    name = "cobalt_strike"
    category = "c2"

    def load(self):
        self.server = self.config.get('server', '127.0.0.1')
        self.port = self.config.get('port', 50050)
        self.password = self.config.get('password', '')
        self.loaded = True
        logging.info("Cobalt Strike plugin loaded")

    def unload(self):
        self.loaded = False

    def configure(self, config: dict):
        self.config = config
        self.server = config.get('server', self.server)
        self.port = config.get('port', self.port)
        self.password = config.get('password', self.password)

    def execute(self, operation: str, **kwargs):
        if operation == 'launch_payload':
            return self.launch_payload(kwargs.get('listener'), kwargs.get('payload'))
        elif operation == 'report':
            return self.report()
        else:
            return {"error": f"Unknown operation: {operation}"}

    def launch_payload(self, listener, payload):
        # Assume use of cscli or similar for automation
        cscli_cmd = [
            'cscli', '--host', self.server, '--port', str(self.port), '--pass', self.password,
            'deploy', '--listener', listener, '--payload', payload
        ]
        result = subprocess.run(cscli_cmd, capture_output=True, text=True)
        return {'status': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}

    def report(self):
        # Retrieve C2 state – mock
        return {'status': 'running', 'beacons': 3, 'active_sessions': 2}


class MythicPlugin(PluginBase):
    name = "mythic"
    category = "c2"

    def load(self):
        self.api_url = self.config.get('api_url', 'http://localhost:7443/api/v1.4')
        self.api_token = self.config.get('api_token')
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {self.api_token}'})
        self.loaded = True
        logging.info("Mythic plugin loaded")

    def unload(self):
        self.session.close()
        self.loaded = False

    def configure(self, config: dict):
        self.config = config
        self.api_url = config.get('api_url', self.api_url)
        self.api_token = config.get('api_token', self.api_token)
        self.session.headers.update({'Authorization': f'Bearer {self.api_token}'})

    def execute(self, operation: str, **kwargs):
        if operation == 'launch_payload':
            return self.launch_payload(kwargs.get('profile'), kwargs.get('payload'))
        elif operation == 'report':
            return self.report(kwargs.get('operation_id'))
        else:
            return {"error": f"Unknown operation: {operation}"}

    def launch_payload(self, profile, payload):
        resp = self.session.post(f"{self.api_url}/payloads/launch", json={"profile": profile, "payload": payload})
        return resp.json()

    def report(self, operation_id):
        resp = self.session.get(f"{self.api_url}/operations/{operation_id}")
        return resp.json()


def register_all_plugins():
    from plugin_interface import PluginRegistry
    PluginRegistry.register_plugin(MetasploitProPlugin())
    PluginRegistry.register_plugin(NessusProPlugin())
    PluginRegistry.register_plugin(BurpSuiteProPlugin())
    PluginRegistry.register_plugin(CobaltStrikePlugin())
    PluginRegistry.register_plugin(MythicPlugin())

