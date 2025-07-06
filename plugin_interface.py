"""
plugin_interface.py - Standardized Plugin Architecture for Nexus-Framework

Defines the base interface and registry for dynamically loaded plugins,
supporting plug-and-play for red-team, OSINT, and C2 tool categories.
"""
import importlib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable
import pkg_resources

class PluginBase(ABC):
    """
    Abstract base class for all Nexus plugins.
    Plugin categories: red_team, osint, c2, etc.
    """
    name: str = None
    category: str = None  # red_team, osint, c2, etc.
    config: dict = {}

    def __init__(self, config: Optional[dict] = None):
        if config:
            self.config = config
        self.loaded = False

    @abstractmethod
    def load(self):
        """Initialize or allocate resources for plugin."""
        self.loaded = True

    @abstractmethod
    def unload(self):
        """Clean up or deallocate resources."""
        self.loaded = False

    @abstractmethod
    def configure(self, config: dict):
        """Update configuration at runtime."""
        self.config = config

    @abstractmethod
    def execute(self, **kwargs):
        """Run plugin's main functionality."""
        pass


class PluginRegistry:
    """Singleton for plugin discovery, registry, and management."""
    _plugins: Dict[str, PluginBase] = {}
    _categories: Dict[str, List[str]] = {"red_team": [], "osint": [], "c2": []}

    @classmethod
    def discover_plugins(cls, entrypoint_group: str = 'nexus.plugins'):
        """Find plugins registered via setuptools entry points or code."""
        for entry_point in pkg_resources.iter_entry_points(entrypoint_group):
            plugin_cls = entry_point.load()
            plugin: PluginBase = plugin_cls()
            cls.register_plugin(plugin)

    @classmethod
    def register_plugin(cls, plugin: PluginBase):
        if plugin.name and plugin.category:
            cls._plugins[plugin.name] = plugin
            if plugin.category not in cls._categories:
                cls._categories[plugin.category] = []
            cls._categories[plugin.category].append(plugin.name)

    @classmethod
    def get_plugins_by_category(cls, category: str) -> List[PluginBase]:
        return [cls._plugins[name] for name in cls._categories.get(category, [])]

    @classmethod
    def get_plugin(cls, name: str) -> Optional[PluginBase]:
        return cls._plugins.get(name)

    @classmethod
    def load_all(cls):
        for plugin in cls._plugins.values():
            plugin.load()

    @classmethod
    def unload_all(cls):
        for plugin in cls._plugins.values():
            plugin.unload()

