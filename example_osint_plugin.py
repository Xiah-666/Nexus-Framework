from plugin_interface import PluginBase

class ExampleOSINTPlugin(PluginBase):
    name = "whois_lookup_plugin"
    category = "osint"

    def load(self):
        self.loaded = True
        print(f"Loaded {self.name}")

    def unload(self):
        self.loaded = False
        print(f"Unloaded {self.name}")

    def configure(self, config: dict):
        self.config = config
        print(f"Configured {self.name} with {config}")

    def execute(self, **kwargs):
        domain = kwargs.get('domain', "example.com")
        # This would return WHOIS data, but just mock for now
        return {"domain": domain, "whois": "FAKE DATA"}

