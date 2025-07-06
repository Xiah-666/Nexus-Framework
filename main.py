#!/usr/bin/env python3
"""
Legacy main.py - redirects to new consolidated entry point.
Maintained for backwards compatibility.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Legacy main function - redirects to nemesis_nexus.py"""
    print("🔄 Redirecting to new NEMESIS-NEXUS entry point...")
    
    # Launch the new consolidated system
    script_path = Path(__file__).parent / "nemesis_nexus.py"
    cmd = [sys.executable, str(script_path)] + sys.argv[1:]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 NEMESIS-NEXUS shutdown requested")
    except Exception as e:
        print(f"❌ Failed to launch NEMESIS-NEXUS: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
