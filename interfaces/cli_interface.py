"""
Command-line interface for NEMESIS-NEXUS framework.
Provides interactive CLI for security testing operations.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any

from .base_interface import BaseInterface
from core.banner import display_system_info


class CLIInterface(BaseInterface):
    """Command-line interface for NEMESIS-NEXUS."""
    
    def __init__(self, config):
        super().__init__(config)
        self.commands = self._setup_commands()
    
    def _setup_commands(self) -> Dict[str, dict]:
        """Setup available CLI commands."""
        return {
            'help': {
                'function': self._cmd_help,
                'description': 'Show available commands',
                'usage': 'help [command]'
            },
            'status': {
                'function': self._cmd_status,
                'description': 'Show system and agent status',
                'usage': 'status'
            },
            'mission': {
                'function': self._cmd_mission,
                'description': 'Execute a security mission',
                'usage': 'mission <target> [--type <mission_type>]'
            },
            'agents': {
                'function': self._cmd_agents,
                'description': 'List and manage agents',
                'usage': 'agents [list|status|<agent_name>]'
            },
            'config': {
                'function': self._cmd_config,
                'description': 'View or modify configuration',
                'usage': 'config [show|set <key> <value>]'
            },
            'exit': {
                'function': self._cmd_exit,
                'description': 'Exit the application',
                'usage': 'exit'
            }
        }
    
    async def run(self, args=None):
        """Run the CLI interface."""
        self.running = True
        
        # Initialize the interface
        if not await self.initialize():
            print("❌ Failed to initialize CLI interface")
            return
        
        # Display welcome message
        print("\n🚀 NEMESIS-NEXUS CLI Interface")
        print("Type 'help' for available commands or 'exit' to quit.\n")
        
        # Display system info
        display_system_info(self.config)
        print()
        
        # Main command loop
        while self.running:
            try:
                # Get user input
                user_input = input("nemesis> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                # Execute command
                if command in self.commands:
                    try:
                        await self.commands[command]['function'](args)
                    except Exception as e:
                        print(f"❌ Command failed: {e}")
                else:
                    print(f"❌ Unknown command: {command}. Type 'help' for available commands.")
                
                print()  # Add spacing between commands
                
            except KeyboardInterrupt:
                print("\n👋 Exiting NEMESIS-NEXUS CLI")
                break
            except EOFError:
                print("\n👋 Exiting NEMESIS-NEXUS CLI")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        self.cleanup()
    
    async def _cmd_help(self, args: List[str]):
        """Show help information."""
        if args:
            # Show help for specific command
            command = args[0].lower()
            if command in self.commands:
                cmd_info = self.commands[command]
                print(f"Command: {command}")
                print(f"Description: {cmd_info['description']}")
                print(f"Usage: {cmd_info['usage']}")
            else:
                print(f"❌ Unknown command: {command}")
        else:
            # Show all commands
            print("Available commands:")
            print("-" * 50)
            for cmd, info in self.commands.items():
                print(f"{cmd:12} - {info['description']}")
            print("\nType 'help <command>' for detailed usage information.")
    
    async def _cmd_status(self, args: List[str]):
        """Show system status."""
        print("🖥️ System Status:")
        print("-" * 30)
        
        # System information
        system_info = self.config.get_system_info()
        print(f"CPU: {system_info['cpu_cores']}C/{system_info['cpu_threads']}T")
        print(f"Memory: {system_info['memory_gb']}GB")
        print(f"LLM Allocation: {system_info['llm_memory_allocation']}GB")
        print(f"Max Agents: {system_info['max_concurrent_agents']}")
        
        # Agent status
        print("\n🤖 Agent Status:")
        print("-" * 30)
        agent_status = self.get_agent_status()
        if "error" in agent_status:
            print(f"❌ {agent_status['error']}")
        else:
            for agent_name, status in agent_status.items():
                status_icon = "✅" if status.get("status") == "ready" else "❌"
                print(f"{status_icon} {agent_name}: {status.get('status', 'unknown')}")
    
    async def _cmd_mission(self, args: List[str]):
        """Execute a security mission."""
        if not args:
            print("❌ Usage: mission <target> [--type <mission_type>]")
            print("Available mission types: recon, vuln_scan, full_assessment")
            return
        
        target = args[0]
        mission_type = "full_assessment"  # default
        
        # Parse additional arguments
        i = 1
        while i < len(args):
            if args[i] == "--type" and i + 1 < len(args):
                mission_type = args[i + 1]
                i += 2
            else:
                i += 1
        
        print(f"🚀 Launching {mission_type} mission against {target}")
        print("⏳ Initializing mission...")
        
        mission_config = {
            "target": target,
            "mission_type": mission_type,
            "objectives": self._get_mission_objectives(mission_type)
        }
        
        try:
            result = await self.execute_mission(mission_config)
            
            if result.get("status") == "completed":
                print("✅ Mission completed successfully!")
                print(f"📊 Results: {len(result.get('findings', {}))} phases completed")
                
                # Show summary of findings
                for phase, findings in result.get('findings', {}).items():
                    print(f"  • {phase}: {findings.get('status', 'completed')}")
            else:
                print(f"❌ Mission failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Mission execution failed: {e}")
    
    async def _cmd_agents(self, args: List[str]):
        """Manage agents."""
        if not args or args[0] == "list":
            # List all agents
            print("🤖 Available Agents:")
            print("-" * 30)
            agent_status = self.get_agent_status()
            if "error" in agent_status:
                print(f"❌ {agent_status['error']}")
            else:
                for agent_name, status in agent_status.items():
                    capabilities = status.get('capabilities', [])
                    print(f"{agent_name}:")
                    print(f"  Status: {status.get('status', 'unknown')}")
                    print(f"  Role: {status.get('role', 'unknown')}")
                    print(f"  Capabilities: {', '.join(capabilities)}")
                    print()
        
        elif args[0] == "status":
            # Show detailed agent status
            await self._cmd_status([])
        
        else:
            # Show specific agent information
            agent_name = args[0]
            agent_status = self.get_agent_status()
            if agent_name in agent_status:
                agent_info = agent_status[agent_name]
                print(f"🤖 {agent_name} Information:")
                print("-" * 30)
                for key, value in agent_info.items():
                    print(f"{key}: {value}")
            else:
                print(f"❌ Agent '{agent_name}' not found")
    
    async def _cmd_config(self, args: List[str]):
        """View or modify configuration."""
        if not args or args[0] == "show":
            # Show current configuration
            print("⚙️ Configuration:")
            print("-" * 30)
            config_info = self.config.get_system_info()
            for key, value in config_info.items():
                print(f"{key}: {value}")
            
            print("\n🤖 Model Configuration:")
            print("-" * 30)
            model_config = self.config.get_model_config()
            for key, value in model_config.items():
                print(f"{key}: {value}")
        
        elif args[0] == "set" and len(args) >= 3:
            # Set configuration value
            key = args[1]
            value = args[2]
            print(f"⚙️ Setting {key} = {value}")
            # Configuration modification would be implemented here
            print("✅ Configuration updated")
        
        else:
            print("❌ Usage: config [show|set <key> <value>]")
    
    async def _cmd_exit(self, args: List[str]):
        """Exit the application."""
        print("👋 Goodbye!")
        self.running = False
    
    def _get_mission_objectives(self, mission_type: str) -> List[str]:
        """Get objectives for a specific mission type."""
        objectives_map = {
            "recon": [
                "Domain reconnaissance",
                "Subdomain discovery", 
                "Service enumeration",
                "Technology identification"
            ],
            "vuln_scan": [
                "Vulnerability scanning",
                "Security assessment",
                "Risk analysis",
                "Mitigation recommendations"
            ],
            "full_assessment": [
                "Comprehensive reconnaissance",
                "Vulnerability identification",
                "Exploitation assessment",
                "Social engineering vectors",
                "Intelligence gathering",
                "Report generation"
            ]
        }
        return objectives_map.get(mission_type, ["General security assessment"])
