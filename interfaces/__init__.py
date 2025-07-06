"""
Interface modules for NEMESIS-NEXUS framework.
Contains CLI, Web, and API interface implementations.
"""

from .base_interface import BaseInterface
from .cli_interface import CLIInterface
from .web_interface import WebInterface
from .api_interface import APIInterface

__all__ = [
    'BaseInterface',
    'CLIInterface',
    'WebInterface',
    'APIInterface'
]
