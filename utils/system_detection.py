"""
System environment detection utilities for NEMESIS-NEXUS framework.
Detects display availability, terminal capabilities, and optimal interface selection.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvironmentInfo:
    """Information about the current system environment."""
    has_display: bool = False
    is_interactive: bool = False
    is_ssh: bool = False
    display_name: Optional[str] = None
    terminal_type: Optional[str] = None
    shell_name: Optional[str] = None


def detect_environment() -> EnvironmentInfo:
    """Detect the current system environment capabilities."""
    env = EnvironmentInfo()
    
    # Check for interactive terminal
    env.is_interactive = sys.stdin.isatty() and sys.stdout.isatty()
    
    # Check for SSH connection
    env.is_ssh = bool(os.environ.get('SSH_CLIENT') or os.environ.get('SSH_TTY'))
    
    # Get terminal and shell information
    env.terminal_type = os.environ.get('TERM')
    env.shell_name = os.path.basename(os.environ.get('SHELL', ''))
    
    # Check for X11/display availability
    env.display_name = os.environ.get('DISPLAY')
    if env.display_name:
        env.has_display = _test_x11_connection()
    
    # Alternative display checks for other environments
    if not env.has_display:
        env.has_display = _check_wayland() or _check_other_displays()
    
    return env


def _test_x11_connection() -> bool:
    """Test if X11 display is actually available."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the window
        root.destroy()   # Clean up
        return True
    except Exception:
        return False


def _check_wayland() -> bool:
    """Check for Wayland display server."""
    return bool(os.environ.get('WAYLAND_DISPLAY'))


def _check_other_displays() -> bool:
    """Check for other display servers (future compatibility)."""
    # Could add checks for other display servers here
    return False


def get_optimal_interface(env: EnvironmentInfo) -> str:
    """Determine the optimal interface based on environment."""
    if env.has_display and env.is_interactive and not env.is_ssh:
        return "web"  # Local with display - use web interface
    elif env.is_interactive:
        return "cli"  # Interactive terminal - use CLI
    else:
        return "api"  # Headless/non-interactive - use API


def format_environment_info(env: EnvironmentInfo) -> str:
    """Format environment information for display."""
    lines = []
    lines.append("🖥️ Environment Detection:")
    lines.append(f"  Interactive: {'✅' if env.is_interactive else '❌'}")
    lines.append(f"  Display: {'✅' if env.has_display else '❌'} ({env.display_name or 'none'})")
    lines.append(f"  SSH: {'✅' if env.is_ssh else '❌'}")
    lines.append(f"  Terminal: {env.terminal_type or 'unknown'}")
    lines.append(f"  Shell: {env.shell_name or 'unknown'}")
    
    optimal = get_optimal_interface(env)
    lines.append(f"  Recommended Interface: {optimal}")
    
    return "\n".join(lines)
