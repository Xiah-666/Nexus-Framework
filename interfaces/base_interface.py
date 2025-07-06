"""
Base interface class for NEMESIS-NEXUS framework.
Provides common functionality for all interface implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from core.config import NemesisConfig
from core.logging_config import get_logger


class BaseInterface(ABC):
    """Base class for all NEMESIS-NEXUS interfaces."""
    
    def __init__(self, config: NemesisConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.running = False
        self.orchestrator = None
    
    async def initialize(self) -> bool:
        """Initialize the interface and its components."""
        try:
            # Initialize orchestrator if not already done
            if not self.orchestrator:
                from agents.orchestrator import NemesisOrchestrator
                self.orchestrator = NemesisOrchestrator(self.config)
                await self.orchestrator.initialize()
            
            self.logger.info(f"{self.__class__.__name__} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            return False
    
    @abstractmethod
    async def run(self, *args, **kwargs):
        """Run the interface. Must be implemented by subclasses."""
        pass
    
    def cleanup(self):
        """Clean up resources when shutting down."""
        try:
            self.running = False
            if self.orchestrator:
                self.orchestrator.cleanup()
            self.logger.info(f"{self.__class__.__name__} cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def execute_mission(self, mission_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a security mission using the orchestrator."""
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized")
        
        try:
            result = await self.orchestrator.execute_mission(mission_config)
            return result
        except Exception as e:
            self.logger.error(f"Mission execution failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        if not self.orchestrator:
            return {"error": "Orchestrator not initialized"}
        
        return self.orchestrator.get_agent_status()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "interface": self.__class__.__name__,
            "running": self.running,
            "config": self.config.get_system_info(),
            "agents": self.get_agent_status()
        }
