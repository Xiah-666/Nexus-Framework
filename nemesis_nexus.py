#!/usr/bin/env python3
"""
NEMESIS-NEXUS - Advanced Multi-Agent AI Cybersecurity Framework
Main entry point with consolidated architecture and clean interfaces.

For authorized penetration testing and security research only.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional

# Add the current directory to Python path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import NemesisConfig
from core.banner import display_banner, check_dependencies
from interfaces.cli_interface import CLIInterface
from interfaces.web_interface import WebInterface
from interfaces.api_interface import APIInterface
from utils.system_detection import detect_environment


class NemesisNexus:
    """Main application controller for NEMESIS-NEXUS framework."""
    
    def __init__(self):
        self.config = NemesisConfig()
        self.interface = None
        
    async def initialize(self) -> bool:
        """Initialize the framework asynchronously."""
        try:
            await self.config.initialize()
            return True
        except Exception as e:
            print(f"❌ Framework initialization failed: {e}")
            return False
    
    async def run_cli(self, args=None):
        """Run the CLI interface."""
        self.interface = CLIInterface(self.config)
        await self.interface.run(args)
    
    async def run_web(self, host="0.0.0.0", port=8501):
        """Run the web interface (Streamlit)."""
        self.interface = WebInterface(self.config)
        await self.interface.run(host, port)
    
    async def run_api(self, host="0.0.0.0", port=8000):
        """Run the API interface (FastAPI)."""
        self.interface = APIInterface(self.config)
        await self.interface.run(host, port)
    
    def cleanup(self):
        """Clean up resources."""
        if self.interface:
            self.interface.cleanup()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NEMESIS-NEXUS - Advanced Multi-Agent AI Cybersecurity Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Auto-detect best interface
  %(prog)s --cli              # Command-line interface
  %(prog)s --web              # Web interface (Streamlit)
  %(prog)s --api              # API interface (FastAPI)
  %(prog)s --web --port 8080  # Web interface on custom port
        """
    )
    
    # Interface selection
    interface_group = parser.add_mutually_exclusive_group()
    interface_group.add_argument("--cli", action="store_true", 
                               help="Launch CLI interface")
    interface_group.add_argument("--web", action="store_true", 
                               help="Launch web interface (Streamlit)")
    interface_group.add_argument("--api", action="store_true", 
                               help="Launch API interface (FastAPI)")
    
    # Connection options
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host address for web/API interfaces")
    parser.add_argument("--port", type=int, 
                       help="Port for web/API interfaces (default: 8501 for web, 8000 for API)")
    
    # System options
    parser.add_argument("--check-deps", action="store_true", 
                       help="Check system dependencies and exit")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    
    return parser.parse_args()


async def main():
    """Main application entry point."""
    args = parse_arguments()
    
    # Display banner
    display_banner()
    
    # Check dependencies if requested
    if args.check_deps:
        success = check_dependencies()
        sys.exit(0 if success else 1)
    
    # Check system dependencies
    if not check_dependencies():
        print("❌ Missing required dependencies. Use --check-deps for details.")
        sys.exit(1)
    
    # Initialize framework
    nexus = NemesisNexus()
    
    if not await nexus.initialize():
        print("❌ Framework initialization failed")
        sys.exit(1)
    
    try:
        # Determine interface to use
        if args.cli:
            await nexus.run_cli()
        elif args.web:
            port = args.port or 8501
            await nexus.run_web(args.host, port)
        elif args.api:
            port = args.port or 8000
            await nexus.run_api(args.host, port)
        else:
            # Auto-detect best interface
            env = detect_environment()
            
            if env.has_display and env.is_interactive:
                print("🖥️ Display detected - starting web interface")
                await nexus.run_web(args.host, 8501)
            elif env.is_interactive:
                print("💻 Terminal detected - starting CLI interface")
                await nexus.run_cli()
            else:
                print("🌐 Headless environment - starting API interface")
                await nexus.run_api(args.host, 8000)
                
    except KeyboardInterrupt:
        print("\n👋 NEMESIS-NEXUS shutdown requested")
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)
    finally:
        nexus.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
