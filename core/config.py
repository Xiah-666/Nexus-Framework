"""
Core configuration management for NEMESIS-NEXUS framework.
Consolidates all configuration logic into a single, clean module.
"""

import os
import yaml
import json
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from cryptography.fernet import Fernet

from .logging_config import get_logger


@dataclass
class NemesisConfig:
    """Unified configuration for NEMESIS-NEXUS framework."""
    
    # System Configuration
    debug_mode: bool = False
    log_level: str = "INFO"
    data_dir: Path = field(default_factory=lambda: Path.home() / ".nemesis")
    
    # Hardware Configuration (auto-detected and optimized)
    cpu_cores: int = field(default_factory=lambda: psutil.cpu_count(logical=False))
    cpu_threads: int = field(default_factory=lambda: psutil.cpu_count(logical=True))
    memory_gb: int = field(default_factory=lambda: psutil.virtual_memory().total // (1024**3))
    max_concurrent_agents: int = 8
    max_concurrent_scans: int = 16
    
    # AI Model Configuration
    default_model: str = "huihui_ai/gemma3-abliterated:27b"
    fallback_model: str = "huihui_ai/jan-nano-abliterated:latest"
    embedding_model: str = "nomic-embed-text:latest"
    code_model: str = "codellama:34b-instruct"
    current_model: Optional[str] = None
    
    # Model allocation (percentage of available memory for LLM)
    llm_memory_percentage: float = 0.5
    
    # Security Configuration
    encryption_key: Optional[str] = None
    api_timeout: int = 300
    max_retries: int = 3
    rate_limit_delay: float = 1.0
    session_timeout: int = 7200  # 2 hours
    
    # Tool Integration
    tool_paths: Dict[str, str] = field(default_factory=dict)
    
    # Operational Configuration
    stealth_mode: bool = True
    auto_cleanup: bool = True
    
    # Network Configuration
    default_api_host: str = "0.0.0.0"
    default_api_port: int = 8000
    default_web_port: int = 8501
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        self.logger = get_logger('NemesisConfig')
        self._detect_hardware()
        self._setup_directories()
        self._setup_encryption()
        self._load_config_file()
    
    def _detect_hardware(self):
        """Auto-detect and optimize hardware configuration."""
        # Optimize concurrent operations based on CPU
        if self.cpu_threads >= 16:
            self.max_concurrent_agents = min(16, self.cpu_threads)
            self.max_concurrent_scans = min(32, self.cpu_threads * 2)
        elif self.cpu_threads >= 8:
            self.max_concurrent_agents = min(12, self.cpu_threads)
            self.max_concurrent_scans = min(24, self.cpu_threads * 2)
        else:
            self.max_concurrent_agents = min(8, self.cpu_threads)
            self.max_concurrent_scans = min(16, self.cpu_threads * 2)
        
        # Adjust model selection based on available memory
        if self.memory_gb >= 64:
            self.default_model = "huihui_ai/gemma3-abliterated:27b"
        elif self.memory_gb >= 32:
            self.default_model = "huihui_ai/gemma3-abliterated:12b"
        else:
            self.default_model = "huihui_ai/jan-nano-abliterated:latest"
        
        self.logger.info(f"Hardware detected: {self.cpu_cores}C/{self.cpu_threads}T, {self.memory_gb}GB RAM")
        self.logger.info(f"Optimized for {self.max_concurrent_agents} agents, {self.max_concurrent_scans} scans")
    
    def _setup_directories(self):
        """Create necessary directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
        (self.data_dir / "sessions").mkdir(exist_ok=True)
        (self.data_dir / "reports").mkdir(exist_ok=True)
        (self.data_dir / "plugins").mkdir(exist_ok=True)
    
    def _setup_encryption(self):
        """Setup encryption for sensitive data."""
        key_file = self.data_dir / "encryption.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            self.encryption_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self.encryption_key)
            # Secure the key file
            os.chmod(key_file, 0o600)
    
    def _load_config_file(self):
        """Load configuration from file if it exists."""
        config_file = self.data_dir / "config.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update configuration with file values
                for key, value in config_data.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                
                self.logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load config file: {e}")
    
    def save_config(self):
        """Save current configuration to file."""
        config_file = self.data_dir / "config.yaml"
        
        # Convert to dict, excluding non-serializable fields
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key not in ['logger', 'encryption_key']:
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            self.logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    async def initialize(self):
        """Initialize async components."""
        try:
            # Initialize model management if available
            if hasattr(self, 'model_switcher'):
                await self.model_switcher.initialize()
            
            self.logger.info("Configuration initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Configuration initialization failed: {e}")
            return False
    
    def get_llm_memory_allocation(self) -> int:
        """Get memory allocation for LLM in GB."""
        return int(self.memory_gb * self.llm_memory_percentage)
    
    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data."""
        fernet = Fernet(self.encryption_key)
        return fernet.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data."""
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(encrypted_data).decode()
    
    def update_tool_path(self, tool_name: str, path: str):
        """Update tool path configuration."""
        self.tool_paths[tool_name] = path
        self.save_config()
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get AI model configuration."""
        return {
            "default_model": self.default_model,
            "fallback_model": self.fallback_model,
            "embedding_model": self.embedding_model,
            "code_model": self.code_model,
            "current_model": self.current_model,
            "memory_allocation": self.get_llm_memory_allocation()
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information summary."""
        return {
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "memory_gb": self.memory_gb,
            "max_concurrent_agents": self.max_concurrent_agents,
            "max_concurrent_scans": self.max_concurrent_scans,
            "llm_memory_allocation": self.get_llm_memory_allocation(),
            "data_dir": str(self.data_dir)
        }
