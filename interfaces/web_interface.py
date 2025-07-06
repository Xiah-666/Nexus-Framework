"""
Web interface for NEMESIS-NEXUS framework using Streamlit.
Provides modern web UI for security testing operations.
"""

import asyncio
import subprocess
import sys
from typing import Optional

from .base_interface import BaseInterface


class WebInterface(BaseInterface):
    """Web interface using Streamlit for NEMESIS-NEXUS."""
    
    def __init__(self, config):
        super().__init__(config)
        self.streamlit_process = None
    
    async def run(self, host: str = "0.0.0.0", port: int = 8501):
        """Run the web interface using Streamlit."""
        self.running = True
        
        # Initialize the interface
        if not await self.initialize():
            print("❌ Failed to initialize web interface")
            return
        
        print(f"🌐 Starting NEMESIS-NEXUS Web Interface")
        print(f"📱 Access the interface at: http://{host}:{port}")
        print("⚠️ AUTHORIZED SECURITY TESTING ONLY")
        print("Press Ctrl+C to stop the service\n")
        
        try:
            # Start Streamlit with the existing web UI
            cmd = [
                sys.executable, "-m", "streamlit", "run", "nemesis_web_ui.py",
                "--server.port", str(port),
                "--server.address", host,
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
            
            self.streamlit_process = subprocess.Popen(cmd)
            
            # Wait for the process to complete
            while self.running:
                if self.streamlit_process.poll() is not None:
                    # Process has ended
                    break
                await asyncio.sleep(1)
                
        except FileNotFoundError:
            print("❌ Streamlit not found. Please install streamlit:")
            print("   pip install streamlit")
        except KeyboardInterrupt:
            print("\n👋 Web interface stopped by user")
        except Exception as e:
            print(f"❌ Failed to start web interface: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up the web interface."""
        try:
            if self.streamlit_process and self.streamlit_process.poll() is None:
                self.streamlit_process.terminate()
                self.streamlit_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if self.streamlit_process:
                self.streamlit_process.kill()
        except Exception as e:
            self.logger.error(f"Error cleaning up web interface: {e}")
        
        super().cleanup()
