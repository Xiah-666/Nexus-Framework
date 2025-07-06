#!/usr/bin/env python3
"""
model_switcher.py - Advanced Model Switcher for NEMESIS-NEXUS
Provides dynamic model switching, configuration management, and comprehensive model information
"""

import json
import yaml
import asyncio
import subprocess
import requests
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import ollama
from logging_config import get_logger

@dataclass
class ModelInfo:
    """Comprehensive model information"""
    name: str
    type: str
    size_gb: float
    context_length: int
    capabilities: List[str]
    specialization: str
    description: str
    status: str = "unknown"
    local: bool = False
    downloaded_at: Optional[str] = None
    last_used: Optional[str] = None
    usage_count: int = 0
    performance_rating: str = "unknown"
    intended_use: List[str] = None
    
    def __post_init__(self):
        if self.intended_use is None:
            self.intended_use = []

class ModelSwitcher:
    """Advanced Model Switcher with comprehensive configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.logger = get_logger('ModelSwitcher')
        self.ollama_client = None
        self.current_model = None
        self.available_models = {}
        self.model_registry = self._load_model_registry()
        self.config_data = self._load_config()
        self.model_usage_stats = {}
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            else:
                return self._create_default_config()
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict:
        """Create default configuration"""
        return {
            "version": "1.0",
            "ollama": {
                "host": "http://localhost:11434",
                "default_model": "huihui_ai/qwen3-abliterated:14b",
                "fallback_model": "huihui_ai/jan-nano-abliterated:latest",
                "embedding_model": "nomic-embed-text:latest",
                "temperature": 0.7,
                "max_tokens": 4096,
                "current_model": None,
                "model_preferences": {}
            },
            "tools": {
                "max_concurrent_scans": 8,
                "nmap_enabled": True,
                "scapy_enabled": True
            },
            "security": {
                "authorized_testing_only": True,
                "log_all_activities": True
            },
            "model_switcher": {
                "auto_download_essential": True,
                "usage_tracking": True,
                "performance_monitoring": True,
                "model_recommendations": True
            }
        }
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            # Backup existing config
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix('.bak')
                self.config_path.rename(backup_path)
            
            # Save new config
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            raise
    
    def _load_model_registry(self) -> Dict[str, ModelInfo]:
        """Load comprehensive model registry with detailed information"""
        models = {
            "huihui_ai/qwen3-abliterated:14b": ModelInfo(
                name="huihui_ai/qwen3-abliterated:14b",
                type="uncensored",
                size_gb=8.5,
                context_length=32768,
                capabilities=["reasoning", "coding", "security", "unrestricted", "multilingual"],
                specialization="general_security",
                description="14B parameter abliterated Qwen3 model optimized for cybersecurity tasks",
                performance_rating="high",
                intended_use=["penetration_testing", "security_analysis", "code_review", "threat_modeling"]
            ),
            "huihui_ai/jan-nano-abliterated:latest": ModelInfo(
                name="huihui_ai/jan-nano-abliterated:latest",
                type="uncensored",
                size_gb=3.8,
                context_length=16384,
                capabilities=["reasoning", "security", "unrestricted", "fast_inference"],
                specialization="rapid_response",
                description="Lightweight abliterated model for quick security assessments",
                performance_rating="medium",
                intended_use=["quick_scans", "rapid_analysis", "lightweight_tasks"]
            ),
            "huihui_ai/gemma3-abliterated:27b": ModelInfo(
                name="huihui_ai/gemma3-abliterated:27b",
                type="uncensored",
                size_gb=16.0,
                context_length=128000,
                capabilities=["advanced_reasoning", "coding", "security", "unrestricted", "long_context"],
                specialization="advanced_reasoning",
                description="27B parameter abliterated Gemma3 model for complex security research",
                performance_rating="very_high",
                intended_use=["complex_analysis", "research", "deep_investigation", "report_generation"]
            ),
            "dolphin-mixtral:8x7b": ModelInfo(
                name="dolphin-mixtral:8x7b",
                type="uncensored",
                size_gb=26.0,
                context_length=32768,
                capabilities=["reasoning", "coding", "security", "creative", "multi_expert"],
                specialization="cybersecurity",
                description="Highly capable uncensored Mixtral model for cybersecurity operations",
                performance_rating="very_high",
                intended_use=["red_team_ops", "exploit_development", "social_engineering", "osint"]
            ),
            "wizard-vicuna-30b-uncensored": ModelInfo(
                name="wizard-vicuna-30b-uncensored",
                type="uncensored",
                size_gb=20.0,
                context_length=16384,
                capabilities=["security", "penetration_testing", "social_engineering", "unrestricted"],
                specialization="red_team",
                description="Specialized model for red team operations and penetration testing",
                performance_rating="high",
                intended_use=["penetration_testing", "vulnerability_assessment", "attack_simulation"]
            ),
            "codellama:34b-instruct": ModelInfo(
                name="codellama:34b-instruct",
                type="code_focused",
                size_gb=19.0,
                context_length=16384,
                capabilities=["exploit_development", "tool_creation", "automation", "code_analysis"],
                specialization="exploit_development",
                description="Code generation model specialized for security tools and exploits",
                performance_rating="high",
                intended_use=["exploit_development", "tool_creation", "code_analysis", "automation"]
            ),
            "nous-hermes-2-mixtral-8x7b-dpo": ModelInfo(
                name="nous-hermes-2-mixtral-8x7b-dpo",
                type="uncensored",
                size_gb=26.0,
                context_length=32768,
                capabilities=["red_team", "social_engineering", "osint", "reporting", "intelligence"],
                specialization="intelligence_gathering",
                description="Excellent model for red team operations and OSINT gathering",
                performance_rating="very_high",
                intended_use=["osint", "intelligence_gathering", "social_engineering", "reporting"]
            ),
            "blacksheep:latest": ModelInfo(
                name="blacksheep:latest",
                type="uncensored",
                size_gb=8.0,
                context_length=8192,
                capabilities=["unrestricted", "security", "research", "fast_inference"],
                specialization="rapid_response",
                description="Lightweight uncensored model for quick security tasks",
                performance_rating="medium",
                intended_use=["quick_tasks", "testing", "development", "lightweight_analysis"]
            ),
            "nomic-embed-text:latest": ModelInfo(
                name="nomic-embed-text:latest",
                type="embedding",
                size_gb=0.5,
                context_length=2048,
                capabilities=["text_embedding", "similarity", "clustering", "search"],
                specialization="embeddings",
                description="Text embedding model for semantic search and analysis",
                performance_rating="high",
                intended_use=["text_analysis", "similarity_search", "clustering", "semantic_analysis"]
            )
        }
        
        return models
    
    async def initialize(self) -> bool:
        """Initialize the model switcher"""
        try:
            # Initialize Ollama client
            self.ollama_client = ollama.Client(host=self.config_data["ollama"]["host"])
            
            # Discover available models
            await self._discover_models()
            
            # Set current model from config or default
            current_from_config = self.config_data["ollama"].get("current_model")
            if current_from_config and current_from_config in self.available_models:
                self.current_model = current_from_config
            else:
                self.current_model = self.config_data["ollama"]["default_model"]
                self.config_data["ollama"]["current_model"] = self.current_model
                self._save_config()
            
            self.logger.info(f"Model switcher initialized with current model: {self.current_model}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model switcher: {e}")
            return False
    
    async def _discover_models(self):
        """Discover available models from Ollama"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.config_data['ollama']['host']}/api/tags", timeout=5)
            if response.status_code != 200:
                self.logger.warning("Ollama service not available")
                return
            
            models_data = response.json().get("models", [])
            
            for model_data in models_data:
                model_name = model_data.get("name", "")
                if model_name in self.model_registry:
                    model_info = self.model_registry[model_name]
                    model_info.status = "available"
                    model_info.local = True
                    model_info.size_gb = model_data.get("size", 0) / (1024**3)
                    self.available_models[model_name] = model_info
                else:
                    # Create basic info for unknown models
                    self.available_models[model_name] = ModelInfo(
                        name=model_name,
                        type="unknown",
                        size_gb=model_data.get("size", 0) / (1024**3),
                        context_length=4096,  # Default
                        capabilities=["general"],
                        specialization="unknown",
                        description=f"Unknown model: {model_name}",
                        status="available",
                        local=True
                    )
            
            self.logger.info(f"Discovered {len(self.available_models)} available models")
            
        except Exception as e:
            self.logger.error(f"Failed to discover models: {e}")
    
    def get_current_model(self) -> Optional[str]:
        """Get the currently selected model"""
        return self.current_model
    
    def get_current_model_info(self) -> Optional[ModelInfo]:
        """Get detailed information about the current model"""
        if self.current_model:
            return self.available_models.get(self.current_model) or self.model_registry.get(self.current_model)
        return None
    
    def list_available_models(self) -> Dict[str, ModelInfo]:
        """List all available models with detailed information"""
        return self.available_models.copy()
    
    def list_all_models(self) -> Dict[str, ModelInfo]:
        """List all models in registry (available and not available)"""
        all_models = self.model_registry.copy()
        # Update with current availability status
        for name, model in self.available_models.items():
            all_models[name] = model
        return all_models
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            # Check if model is available
            if model_name not in self.available_models:
                # Try to refresh model list
                await self._discover_models()
                if model_name not in self.available_models:
                    self.logger.error(f"Model {model_name} not available")
                    return False
            
            # Update current model
            old_model = self.current_model
            self.current_model = model_name
            
            # Update configuration
            self.config_data["ollama"]["current_model"] = model_name
            self._save_config()
            
            # Update usage statistics
            self._update_usage_stats(model_name)
            
            self.logger.info(f"Switched from {old_model} to {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch to model {model_name}: {e}")
            return False
    
    def _update_usage_stats(self, model_name: str):
        """Update model usage statistics"""
        if model_name not in self.model_usage_stats:
            self.model_usage_stats[model_name] = {
                "usage_count": 0,
                "first_used": datetime.now().isoformat(),
                "last_used": None
            }
        
        self.model_usage_stats[model_name]["usage_count"] += 1
        self.model_usage_stats[model_name]["last_used"] = datetime.now().isoformat()
        
        # Update model info if available
        if model_name in self.available_models:
            self.available_models[model_name].usage_count = self.model_usage_stats[model_name]["usage_count"]
            self.available_models[model_name].last_used = self.model_usage_stats[model_name]["last_used"]
    
    async def download_model(self, model_name: str) -> bool:
        """Download a model from Ollama registry"""
        try:
            self.logger.info(f"Starting download of {model_name}")
            
            # Start download process
            process = await asyncio.create_subprocess_exec(
                'ollama', 'pull', model_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Successfully downloaded {model_name}")
                
                # Add to available models
                if model_name in self.model_registry:
                    model_info = self.model_registry[model_name]
                    model_info.status = "available"
                    model_info.local = True
                    model_info.downloaded_at = datetime.now().isoformat()
                    self.available_models[model_name] = model_info
                
                # Refresh model list
                await self._discover_models()
                return True
            else:
                self.logger.error(f"Failed to download {model_name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Exception downloading {model_name}: {e}")
            return False
    
    def get_model_recommendations(self, task_type: str = None) -> List[Tuple[str, ModelInfo, str]]:
        """Get model recommendations for specific task types"""
        recommendations = []
        
        # Task-specific model mapping
        task_recommendations = {
            "exploit_development": ["codellama:34b-instruct", "dolphin-mixtral:8x7b"],
            "red_team": ["wizard-vicuna-30b-uncensored", "nous-hermes-2-mixtral-8x7b-dpo"],
            "osint": ["dolphin-mixtral:8x7b", "nous-hermes-2-mixtral-8x7b-dpo"],
            "social_engineering": ["wizard-vicuna-30b-uncensored", "dolphin-mixtral:8x7b"],
            "reasoning": ["huihui_ai/gemma3-abliterated:27b", "dolphin-mixtral:8x7b"],
            "rapid_response": ["blacksheep:latest", "huihui_ai/jan-nano-abliterated:latest"],
            "general_security": ["huihui_ai/qwen3-abliterated:14b", "dolphin-mixtral:8x7b"],
            "intelligence_gathering": ["nous-hermes-2-mixtral-8x7b-dpo", "dolphin-mixtral:8x7b"],
            "code_analysis": ["codellama:34b-instruct", "huihui_ai/gemma3-abliterated:27b"]
        }
        
        if task_type and task_type in task_recommendations:
            # Get specific recommendations for task
            for model_name in task_recommendations[task_type]:
                if model_name in self.model_registry:
                    model_info = self.available_models.get(model_name) or self.model_registry[model_name]
                    reason = f"Optimized for {task_type}"
                    recommendations.append((model_name, model_info, reason))
        else:
            # Get general recommendations based on performance and availability
            for model_name, model_info in self.available_models.items():
                if model_info.performance_rating in ["high", "very_high"]:
                    reason = f"High performance {model_info.specialization} model"
                    recommendations.append((model_name, model_info, reason))
        
        # Sort by performance rating and availability
        performance_order = {"very_high": 4, "high": 3, "medium": 2, "low": 1, "unknown": 0}
        recommendations.sort(key=lambda x: (
            1 if x[1].local else 0,  # Available models first
            performance_order.get(x[1].performance_rating, 0)  # Then by performance
        ), reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def get_model_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive model status summary"""
        summary = {
            "current_model": self.current_model,
            "current_model_info": asdict(self.get_current_model_info()) if self.get_current_model_info() else None,
            "total_models_available": len(self.available_models),
            "total_models_registry": len(self.model_registry),
            "models_by_type": {},
            "models_by_specialization": {},
            "usage_statistics": self.model_usage_stats,
            "ollama_status": "unknown",
            "last_updated": datetime.now().isoformat()
        }
        
        # Check Ollama status
        try:
            response = requests.get(f"{self.config_data['ollama']['host']}/api/tags", timeout=2)
            summary["ollama_status"] = "running" if response.status_code == 200 else "error"
        except:
            summary["ollama_status"] = "not_running"
        
        # Categorize models
        for model_info in self.available_models.values():
            # By type
            if model_info.type not in summary["models_by_type"]:
                summary["models_by_type"][model_info.type] = 0
            summary["models_by_type"][model_info.type] += 1
            
            # By specialization
            if model_info.specialization not in summary["models_by_specialization"]:
                summary["models_by_specialization"][model_info.specialization] = 0
            summary["models_by_specialization"][model_info.specialization] += 1
        
        return summary
    
    def export_model_config(self, file_path: str = None) -> str:
        """Export current model configuration to JSON"""
        if file_path is None:
            file_path = f"model_config_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "current_model": self.current_model,
            "available_models": {name: asdict(info) for name, info in self.available_models.items()},
            "model_registry": {name: asdict(info) for name, info in self.model_registry.items()},
            "usage_statistics": self.model_usage_stats,
            "configuration": self.config_data["ollama"],
            "export_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Model configuration exported to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to export model configuration: {e}")
            raise
    
    def get_optimal_model_for_task(self, task_type: str, constraints: Dict[str, Any] = None) -> Optional[str]:
        """Get the optimal model for a specific task with optional constraints"""
        constraints = constraints or {}
        max_size_gb = constraints.get("max_size_gb")
        min_context_length = constraints.get("min_context_length", 0)
        required_capabilities = constraints.get("required_capabilities", [])
        
        recommendations = self.get_model_recommendations(task_type)
        
        for model_name, model_info, reason in recommendations:
            # Check constraints
            if max_size_gb and model_info.size_gb > max_size_gb:
                continue
            
            if model_info.context_length < min_context_length:
                continue
            
            if required_capabilities:
                if not all(cap in model_info.capabilities for cap in required_capabilities):
                    continue
            
            # Check if model is available locally
            if model_info.local:
                return model_name
        
        return None
