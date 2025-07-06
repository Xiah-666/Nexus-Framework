"""
Startup banner and dependency checking for NEMESIS-NEXUS framework.
"""

import shutil
import sys
from pathlib import Path

try:
    import pyfiglet
    import colorama
    from colorama import Fore, Style
    colorama.init()
    HAS_GRAPHICS = True
except ImportError:
    HAS_GRAPHICS = False


def display_banner():
    """Display the NEMESIS-NEXUS startup banner."""
    if HAS_GRAPHICS:
        print(Fore.RED + Style.BRIGHT)
        try:
            print(pyfiglet.figlet_format("NEMESIS-NEXUS"))
        except:
            print("NEMESIS-NEXUS")
        print(Style.RESET_ALL)
        
        print(Fore.CYAN + "🚀 Advanced Multi-Agent AI Cybersecurity Framework v2.0")
        print("⚡ Powered by Uncensored AI Models & Multi-Agent Orchestration")
        print("🛡️ For Authorized Security Testing & Red Team Operations")
        print("🤖 Full Local AI Processing with Advanced Capabilities")
        print(Style.RESET_ALL)
        
        print(Fore.YELLOW + "\n⚠️ AUTHORIZED SECURITY TESTING ONLY ⚠️")
        print("This framework is designed for professional penetration testing")
        print("and red team operations with proper authorization.")
        print(Style.RESET_ALL)
    else:
        print("=" * 60)
        print("NEMESIS-NEXUS - Advanced Multi-Agent AI Cybersecurity Framework")
        print("=" * 60)
        print("🚀 Powered by Uncensored AI Models")
        print("⚠️ FOR AUTHORIZED SECURITY TESTING ONLY")
        print("=" * 60)


def check_dependencies() -> bool:
    """Check system dependencies and return True if all are satisfied."""
    required_tools = {
        'python3': 'Python 3.8+ interpreter',
        'ollama': 'Ollama for local LLM inference',
    }
    
    optional_tools = {
        'nmap': 'Network scanning capabilities',
        'git': 'Version control operations',
    }
    
    missing_required = []
    missing_optional = []
    
    # Check required tools
    for tool, description in required_tools.items():
        if shutil.which(tool) is None:
            missing_required.append(f"{tool} - {description}")
    
    # Check optional tools
    for tool, description in optional_tools.items():
        if shutil.which(tool) is None:
            missing_optional.append(f"{tool} - {description}")
    
    # Report results
    if HAS_GRAPHICS:
        if missing_required:
            print(f"{Fore.RED}❌ Missing required dependencies:")
            for dep in missing_required:
                print(f"   • {dep}")
            print(Style.RESET_ALL)
        
        if missing_optional:
            print(f"{Fore.YELLOW}⚠️ Missing optional dependencies:")
            for dep in missing_optional:
                print(f"   • {dep}")
            print(Style.RESET_ALL)
        
        if not missing_required:
            print(f"{Fore.GREEN}✅ All required dependencies satisfied")
            if not missing_optional:
                print("✅ All optional dependencies satisfied")
            print(Style.RESET_ALL)
    else:
        if missing_required:
            print("❌ Missing required dependencies:")
            for dep in missing_required:
                print(f"   • {dep}")
        
        if missing_optional:
            print("⚠️ Missing optional dependencies:")
            for dep in missing_optional:
                print(f"   • {dep}")
        
        if not missing_required:
            print("✅ All required dependencies satisfied")
    
    return len(missing_required) == 0


def display_system_info(config):
    """Display system configuration information."""
    if HAS_GRAPHICS:
        print(f"\n{Fore.GREEN}🖥️ System Configuration:")
        print(f" Hardware: {config.cpu_cores}C/{config.cpu_threads}T, {config.memory_gb}GB RAM")
        print(f" AI Models: {config.default_model}")
        print(f" Max Agents: {config.max_concurrent_agents}")
        print(f" Data Directory: {config.data_dir}")
        print(Style.RESET_ALL)
    else:
        print(f"\n🖥️ System Configuration:")
        print(f" Hardware: {config.cpu_cores}C/{config.cpu_threads}T, {config.memory_gb}GB RAM")
        print(f" AI Models: {config.default_model}")
        print(f" Max Agents: {config.max_concurrent_agents}")
        print(f" Data Directory: {config.data_dir}")


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        if HAS_GRAPHICS:
            print(f"{Fore.RED}❌ Python 3.8+ required, found {sys.version}")
            print(Style.RESET_ALL)
        else:
            print(f"❌ Python 3.8+ required, found {sys.version}")
        return False
    return True
