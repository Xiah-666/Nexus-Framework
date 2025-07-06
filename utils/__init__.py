"""
Utility modules for NEMESIS-NEXUS framework.
Contains system detection and other utility functions.
"""

from .system_detection import detect_environment, EnvironmentInfo

__all__ = [
    'detect_environment',
    'EnvironmentInfo'
]
