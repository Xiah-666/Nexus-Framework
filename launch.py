#!/usr/bin/env python3
"""
Simple launcher script for NEMESIS-NEXUS.
Provides backwards compatibility with existing usage patterns.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch NEMESIS-NEXUS using the new consolidated entry point."""
    
    # Add current directory to path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Map old arguments to new ones
    args = sys.argv[1:]
    new_args = []
    
    # Handle legacy argument mapping
    for i, arg in enumerate(args):
        if arg == '--streamlit':
            new_args.append('--web')
        elif arg == '--web-api':
            new_args.append('--api')
        elif arg in ['--gui', '--tkinter']:
            # Legacy GUI options - default to web
            new_args.append('--web')
        else:
            new_args.append(arg)
    
    # Launch using the new entry point
    try:
        from nemesis_nexus import main as nemesis_main
        import asyncio
        
        # Temporarily modify sys.argv for the new main function
        old_argv = sys.argv
        sys.argv = ['nemesis_nexus.py'] + new_args
        
        try:
            asyncio.run(nemesis_main())
        finally:
            sys.argv = old_argv
            
    except ImportError:
        # Fallback: launch as subprocess
        python_cmd = sys.executable
        script_path = current_dir / "nemesis_nexus.py"
        cmd = [python_cmd, str(script_path)] + new_args
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n👋 NEMESIS-NEXUS shutdown requested")
        except Exception as e:
            print(f"❌ Failed to launch NEMESIS-NEXUS: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
