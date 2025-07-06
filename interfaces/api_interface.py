"""
API interface for NEMESIS-NEXUS framework using FastAPI.
Provides REST API for automated security testing operations.
"""

import asyncio
import subprocess
import sys
from typing import Optional

from .base_interface import BaseInterface


class APIInterface(BaseInterface):
    """API interface using FastAPI for NEMESIS-NEXUS."""
    
    def __init__(self, config):
        super().__init__(config)
        self.fastapi_process = None
    
    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API interface using FastAPI."""
        self.running = True
        
        # Initialize the interface
        if not await self.initialize():
            print("❌ Failed to initialize API interface")
            return
        
        print(f"🚀 Starting NEMESIS-NEXUS API Interface")
        print(f"📡 API endpoint: http://{host}:{port}")
        print(f"📖 Documentation: http://{host}:{port}/docs")
        print("⚠️ AUTHORIZED SECURITY TESTING ONLY")
        print("Press Ctrl+C to stop the service\n")
        
        try:
            # Start FastAPI with uvicorn using the existing API
            cmd = [
                sys.executable, "-m", "uvicorn", "nemesis_web_api:app",
                "--host", host,
                "--port", str(port),
                "--reload", "false"
            ]
            
            self.fastapi_process = subprocess.Popen(cmd)
            
            # Wait for the process to complete
            while self.running:
                if self.fastapi_process.poll() is not None:
                    # Process has ended
                    break
                await asyncio.sleep(1)
                
        except FileNotFoundError:
            print("❌ uvicorn not found. Please install uvicorn:")
            print("   pip install uvicorn")
        except KeyboardInterrupt:
            print("\n👋 API interface stopped by user")
        except Exception as e:
            print(f"❌ Failed to start API interface: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up the API interface."""
        try:
            if self.fastapi_process and self.fastapi_process.poll() is None:
                self.fastapi_process.terminate()
                self.fastapi_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if self.fastapi_process:
                self.fastapi_process.kill()
        except Exception as e:
            self.logger.error(f"Error cleaning up API interface: {e}")
        
        super().cleanup()
