"""
cli_interface.py - Intuitive Command-Line Interface for Nemesis-Nexus Framework
Routes all operations through the unified core APIs (agents  plugin orchestrator)
"""
import sys
import argparse
import json
from pray import NemesisConfig, NemesisOrchestrator, PluginRegistry, execute_plugin
from model_switcher import ModelSwitcher
import asyncio
import colorama
from colorama import Fore, Style
from datetime import datetime
import tabulate

colorama.init()

def print_banner():
    print(Fore.RED + Style.BRIGHT)
    print("NEMESIS-NEXUS CLI - Advanced Multi-Agent AI Cybersecurity Framework v1.0")
    print(Style.RESET_ALL)
    print(Fore.YELLOW + '\u26a0\ufe0f AUTHORIZED SECURITY TESTING ONLY!')
    print(Style.RESET_ALL)

async def exec_plugin(args):
    try:
        result = execute_plugin(args.name, **dict(kv.split('=') for kv in args.options or []))
        print(Fore.GREEN + f"[PLUGIN: {args.name}] Result:\n{result}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Plugin execution failed: {e}" + Style.RESET_ALL)

async def run_agent_task(args):
    config = NemesisConfig()
    orchestrator = NemesisOrchestrator(config)
    await orchestrator.initialize()
    agent = orchestrator.agents.get(args.agent)
    if not agent:
        print(Fore.RED + f"Agent '{args.agent}' not found." + Style.RESET_ALL)
        return
    result = await agent.execute_task(args.task, context={})
    print(f"Agent '{args.agent}' result: {result}")

async def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Nemesis-Nexus CLI")
    
    # Global model selection flags
    parser.add_argument("--model", help="Override the current model for this session")
    parser.add_argument("--set-model", help="Set and persist the active model")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--model-info", help="Show detailed information about a specific model")
    
    subparsers = parser.add_subparsers(dest="command")

    sp_plugin = subparsers.add_parser("plugin", help="Run a plugin by name")
    sp_plugin.add_argument("name", help="Plugin name")
    sp_plugin.add_argument("--options", nargs=argparse.REMAINDER, help="Options as key=value pairs")
    sp_plugin.set_defaults(func=exec_plugin)

    sp_agent = subparsers.add_parser("agent", help="Run an agent task")
    sp_agent.add_argument("agent", help="Agent name (ReconAgent/ExploitAgent/SocialAgent/NetworkAgent/IntelAgent)")
    sp_agent.add_argument("task", help="Task description")
    sp_agent.set_defaults(func=run_agent_task)

    # Alias subsystem commands:
    from alias_manager import AliasManager
    alias_mgr = AliasManager("config.yaml")
    sp_alias = subparsers.add_parser("alias", help="Manage aliasing for payloads, environments, servers, targets")
    sp_alias.add_argument("action", choices=["create", "view", "delete", "import", "export"], help="Action for alias management")
    sp_alias.add_argument("type", nargs="?", choices=["payloads", "environments", "servers", "targets"])
    sp_alias.add_argument("name", nargs="?")
    sp_alias.add_argument("value", nargs="?")
    sp_alias.add_argument("path", nargs="?")
    def handle_alias(args):
        if args.action == "create":
            if not (args.type and args.name and args.value):
                print("Must provide type, name, and value for alias creation.")
                return
            alias_mgr.create_alias(args.type, args.name, args.value)
            print(f"Alias '{args.name}' created for {args.type}.")
        elif args.action == "view":
            aliases = alias_mgr.view_aliases(args.type) if args.type else alias_mgr.view_aliases()
            print(json.dumps(aliases, indent=2))
        elif args.action == "delete":
            if not (args.type and args.name):
                print("Must provide type and name for alias deletion.")
                return
            if alias_mgr.delete_alias(args.type, args.name):
                print(f"Alias '{args.name}' deleted from {args.type}.")
            else:
                print(f"Alias '{args.name}' not found in {args.type}.")
        elif args.action == "import":
            if not args.path:
                print("Must provide path to import aliases from.")
                return
            if alias_mgr.import_aliases(args.path):
                print(f"Aliases imported from {args.path}.")
            else:
                print(f"Import failed for {args.path}.")
        elif args.action == "export":
            if not args.path:
                print("Must provide path to export aliases to.")
                return
            if alias_mgr.export_aliases(args.path):
                print(f"Aliases exported to {args.path}.")
            else:
                print(f"Export failed for {args.path}.")
    sp_alias.set_defaults(func=handle_alias)

    # Model management commands
    sp_model = subparsers.add_parser("model", help="Model management and switching")
    model_subparsers = sp_model.add_subparsers(dest="model_command")
    
    # List models command
    sp_model_list = model_subparsers.add_parser("list", help="List available models")
    sp_model_list.add_argument("--all", action="store_true", help="Show all models in registry (not just available)")
    sp_model_list.add_argument("--format", choices=["table", "json", "simple"], default="table", help="Output format")
    sp_model_list.add_argument("--task-type", help="Filter models by task type")
    sp_model_list.set_defaults(func=handle_model_list)
    
    # Current model command
    sp_model_current = model_subparsers.add_parser("current", help="Show current model")
    sp_model_current.add_argument("--info", action="store_true", help="Show detailed model information")
    sp_model_current.set_defaults(func=handle_model_current)
    
    # Switch model command
    sp_model_switch = model_subparsers.add_parser("switch", help="Switch to a different model")
    sp_model_switch.add_argument("model_name", help="Name of the model to switch to")
    sp_model_switch.set_defaults(func=handle_model_switch)
    
    # Set model command (alias for switch)
    sp_model_set = model_subparsers.add_parser("set", help="Set active model (alias for switch)")
    sp_model_set.add_argument("model_name", help="Name of the model to set as active")
    sp_model_set.set_defaults(func=handle_model_switch)
    
    # Download model command
    sp_model_download = model_subparsers.add_parser("download", help="Download a model from registry")
    sp_model_download.add_argument("model_name", help="Name of the model to download")
    sp_model_download.add_argument("--force", action="store_true", help="Force re-download if already exists")
    sp_model_download.set_defaults(func=handle_model_download)
    
    # Model status command
    sp_model_status = model_subparsers.add_parser("status", help="Show comprehensive model status")
    sp_model_status.add_argument("--json", action="store_true", help="Output in JSON format")
    sp_model_status.set_defaults(func=handle_model_status)
    
    # Model recommendations command
    sp_model_recommend = model_subparsers.add_parser("recommend", help="Get model recommendations")
    sp_model_recommend.add_argument("--task-type", help="Task type for recommendations")
    sp_model_recommend.add_argument("--max-size", type=float, help="Maximum model size in GB")
    sp_model_recommend.add_argument("--min-context", type=int, help="Minimum context length")
    sp_model_recommend.add_argument("--capabilities", nargs="+", help="Required capabilities")
    sp_model_recommend.set_defaults(func=handle_model_recommend)
    
    # Model export command
    sp_model_export = model_subparsers.add_parser("export", help="Export model configuration")
    sp_model_export.add_argument("--output", "-o", help="Output file path")
    sp_model_export.set_defaults(func=handle_model_export)

    args = parser.parse_args()
    
    # Handle global model flags first
    if args.list_models:
        await handle_global_list_models()
        return
    
    if args.model_info:
        await handle_global_model_info(args.model_info)
        return
    
    if args.set_model:
        await handle_global_set_model(args.set_model)
        return
    
    # Initialize model switcher for session override
    if args.model:
        await handle_global_model_override(args.model)
    
    if hasattr(args, 'func'):
        await args.func(args)
    else:
        parser.print_help()

# Model management command handlers

async def handle_model_list(args):
    """Handle model list command"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        if args.all:
            models = model_switcher.list_all_models()
        else:
            models = model_switcher.list_available_models()
        
        # Filter by task type if specified
        if args.task_type:
            filtered_models = {}
            for name, info in models.items():
                if args.task_type.lower() in info.specialization.lower() or \
                   any(args.task_type.lower() in use.lower() for use in info.intended_use):
                    filtered_models[name] = info
            models = filtered_models
        
        if args.format == "json":
            model_data = {name: {
                "type": info.type,
                "size_gb": info.size_gb,
                "context_length": info.context_length,
                "capabilities": info.capabilities,
                "specialization": info.specialization,
                "description": info.description,
                "status": info.status,
                "local": info.local,
                "performance_rating": info.performance_rating,
                "intended_use": info.intended_use
            } for name, info in models.items()}
            print(json.dumps(model_data, indent=2))
        elif args.format == "simple":
            for name, info in models.items():
                status_indicator = "🟢" if info.local else "🔴"
                print(f"{status_indicator} {name} ({info.size_gb:.1f}GB) - {info.specialization}")
        else:  # table format
            table_data = []
            for name, info in models.items():
                status_indicator = "🟢 Available" if info.local else "🔴 Not Local"
                table_data.append([
                    name[:40] + "..." if len(name) > 40 else name,
                    info.type,
                    f"{info.size_gb:.1f}GB",
                    f"{info.context_length:,}",
                    info.specialization,
                    info.performance_rating,
                    status_indicator
                ])
            
            headers = ["Model Name", "Type", "Size", "Context", "Specialization", "Performance", "Status"]
            print("\n" + Fore.CYAN + "Available Models:" + Style.RESET_ALL)
            print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
            
    except Exception as e:
        print(Fore.RED + f"Error listing models: {e}" + Style.RESET_ALL)

async def handle_model_current(args):
    """Handle current model command"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        current_model = model_switcher.get_current_model()
        if not current_model:
            print(Fore.YELLOW + "No model currently selected" + Style.RESET_ALL)
            return
        
        print(Fore.GREEN + f"Current model: {current_model}" + Style.RESET_ALL)
        
        if args.info:
            model_info = model_switcher.get_current_model_info()
            if model_info:
                print(f"\n{Fore.CYAN}Model Information:{Style.RESET_ALL}")
                print(f"  Type: {model_info.type}")
                print(f"  Size: {model_info.size_gb:.1f} GB")
                print(f"  Context Length: {model_info.context_length:,} tokens")
                print(f"  Specialization: {model_info.specialization}")
                print(f"  Performance: {model_info.performance_rating}")
                print(f"  Description: {model_info.description}")
                print(f"  Capabilities: {', '.join(model_info.capabilities)}")
                print(f"  Intended Use: {', '.join(model_info.intended_use)}")
                if model_info.usage_count > 0:
                    print(f"  Usage Count: {model_info.usage_count}")
                    print(f"  Last Used: {model_info.last_used}")
                    
    except Exception as e:
        print(Fore.RED + f"Error getting current model: {e}" + Style.RESET_ALL)

async def handle_model_switch(args):
    """Handle model switch command"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        model_name = args.model_name
        
        print(f"Switching to model: {model_name}...")
        success = await model_switcher.switch_model(model_name)
        
        if success:
            print(Fore.GREEN + f"Successfully switched to {model_name}" + Style.RESET_ALL)
            
            # Show model info
            model_info = model_switcher.get_current_model_info()
            if model_info:
                print(f"  Size: {model_info.size_gb:.1f} GB")
                print(f"  Specialization: {model_info.specialization}")
                print(f"  Context Length: {model_info.context_length:,} tokens")
        else:
            print(Fore.RED + f"Failed to switch to {model_name}" + Style.RESET_ALL)
            print("Available models:")
            available = model_switcher.list_available_models()
            for name in available.keys():
                print(f"  - {name}")
                
    except Exception as e:
        print(Fore.RED + f"Error switching model: {e}" + Style.RESET_ALL)

async def handle_model_download(args):
    """Handle model download command"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        model_name = args.model_name
        
        # Check if model is already available
        available_models = model_switcher.list_available_models()
        if model_name in available_models and not args.force:
            print(Fore.YELLOW + f"Model {model_name} is already available. Use --force to re-download." + Style.RESET_ALL)
            return
        
        print(f"Downloading model: {model_name}...")
        print("This may take several minutes depending on model size.")
        
        success = await model_switcher.download_model(model_name)
        
        if success:
            print(Fore.GREEN + f"Successfully downloaded {model_name}" + Style.RESET_ALL)
            
            # Show model info if available in registry
            all_models = model_switcher.list_all_models()
            if model_name in all_models:
                model_info = all_models[model_name]
                print(f"  Size: {model_info.size_gb:.1f} GB")
                print(f"  Specialization: {model_info.specialization}")
                print(f"  Description: {model_info.description}")
        else:
            print(Fore.RED + f"Failed to download {model_name}" + Style.RESET_ALL)
            print("Please check the model name and try again.")
                
    except Exception as e:
        print(Fore.RED + f"Error downloading model: {e}" + Style.RESET_ALL)

async def handle_model_status(args):
    """Handle model status command"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        status = model_switcher.get_model_status_summary()
        
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"\n{Fore.CYAN}NEMESIS-NEXUS Model Status Summary{Style.RESET_ALL}")
            print("=" * 50)
            
            # Current model
            current = status["current_model"]
            if current:
                print(f"\n{Fore.GREEN}Current Model: {current}{Style.RESET_ALL}")
                if status["current_model_info"]:
                    info = status["current_model_info"]
                    print(f"  Size: {info['size_gb']:.1f} GB")
                    print(f"  Type: {info['type']}")
                    print(f"  Specialization: {info['specialization']}")
                    print(f"  Performance: {info['performance_rating']}")
            else:
                print(f"\n{Fore.YELLOW}No model currently selected{Style.RESET_ALL}")
            
            # Model counts
            print(f"\n{Fore.CYAN}Model Statistics:{Style.RESET_ALL}")
            print(f"  Available Models: {status['total_models_available']}")
            print(f"  Registry Models: {status['total_models_registry']}")
            
            # Ollama status
            ollama_status = status["ollama_status"]
            status_color = Fore.GREEN if ollama_status == "running" else Fore.RED
            print(f"  Ollama Status: {status_color}{ollama_status.title()}{Style.RESET_ALL}")
            
            # Models by type
            if status["models_by_type"]:
                print(f"\n{Fore.CYAN}Models by Type:{Style.RESET_ALL}")
                for model_type, count in status["models_by_type"].items():
                    print(f"  {model_type}: {count}")
            
            # Models by specialization
            if status["models_by_specialization"]:
                print(f"\n{Fore.CYAN}Models by Specialization:{Style.RESET_ALL}")
                for spec, count in status["models_by_specialization"].items():
                    print(f"  {spec}: {count}")
            
            # Usage statistics
            if status["usage_statistics"]:
                print(f"\n{Fore.CYAN}Most Used Models:{Style.RESET_ALL}")
                sorted_usage = sorted(status["usage_statistics"].items(), 
                                     key=lambda x: x[1]["usage_count"], reverse=True)
                for model_name, stats in sorted_usage[:5]:
                    print(f"  {model_name}: {stats['usage_count']} times")
            
            print(f"\nLast Updated: {status['last_updated']}")
            
    except Exception as e:
        print(Fore.RED + f"Error getting model status: {e}" + Style.RESET_ALL)

async def handle_model_recommend(args):
    """Handle model recommend command"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        # Build constraints
        constraints = {}
        if args.max_size:
            constraints["max_size_gb"] = args.max_size
        if args.min_context:
            constraints["min_context_length"] = args.min_context
        if args.capabilities:
            constraints["required_capabilities"] = args.capabilities
        
        recommendations = model_switcher.get_model_recommendations(args.task_type)
        
        if not recommendations:
            print(Fore.YELLOW + "No recommendations found for the specified criteria." + Style.RESET_ALL)
            return
        
        print(f"\n{Fore.CYAN}Model Recommendations{Style.RESET_ALL}")
        if args.task_type:
            print(f"Task Type: {args.task_type}")
        if constraints:
            print(f"Constraints: {constraints}")
        print("=" * 50)
        
        for i, (model_name, model_info, reason) in enumerate(recommendations, 1):
            status_indicator = "🟢" if model_info.local else "🔴"
            print(f"\n{i}. {status_indicator} {model_name}")
            print(f"   Size: {model_info.size_gb:.1f} GB")
            print(f"   Context: {model_info.context_length:,} tokens")
            print(f"   Performance: {model_info.performance_rating}")
            print(f"   Reason: {reason}")
            print(f"   Description: {model_info.description}")
            
            if not model_info.local:
                print(f"   {Fore.YELLOW}⚠️  Model not downloaded. Use 'model download {model_name}' to download.{Style.RESET_ALL}")
        
        # Show optimal model if constraints provided
        if constraints or args.task_type:
            optimal = model_switcher.get_optimal_model_for_task(args.task_type, constraints)
            if optimal:
                print(f"\n{Fore.GREEN}Optimal Model: {optimal}{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.YELLOW}No optimal model found matching all constraints.{Style.RESET_ALL}")
                
    except Exception as e:
        print(Fore.RED + f"Error getting model recommendations: {e}" + Style.RESET_ALL)

async def handle_model_export(args):
    """Handle model export command"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        output_file = model_switcher.export_model_config(args.output)
        print(Fore.GREEN + f"Model configuration exported to: {output_file}" + Style.RESET_ALL)
        
    except Exception as e:
        print(Fore.RED + f"Error exporting model configuration: {e}" + Style.RESET_ALL)

# Global model management handlers

async def handle_global_list_models():
    """Handle global --list-models flag"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        models = model_switcher.list_available_models()
        
        print(f"\n{Fore.CYAN}Available Models:{Style.RESET_ALL}")
        print("=" * 50)
        
        for name, info in models.items():
            status_indicator = "🟢" if info.local else "🔴"
            current_indicator = "⭐" if name == model_switcher.get_current_model() else "  "
            print(f"{current_indicator}{status_indicator} {name}")
            print(f"    Size: {info.size_gb:.1f} GB | Context: {info.context_length:,} | {info.specialization}")
            print(f"    {info.description}")
            print()
        
        current = model_switcher.get_current_model()
        if current:
            print(f"{Fore.GREEN}Current active model: {current}{Style.RESET_ALL}")
        
    except Exception as e:
        print(Fore.RED + f"Error listing models: {e}" + Style.RESET_ALL)

async def handle_global_model_info(model_name: str):
    """Handle global --model-info flag"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        all_models = model_switcher.list_all_models()
        if model_name not in all_models:
            print(Fore.RED + f"Model '{model_name}' not found." + Style.RESET_ALL)
            return
        
        model_info = all_models[model_name]
        
        print(f"\n{Fore.CYAN}Model Information: {model_name}{Style.RESET_ALL}")
        print("=" * 60)
        print(f"Type: {model_info.type}")
        print(f"Size: {model_info.size_gb:.1f} GB")
        print(f"Context Length: {model_info.context_length:,} tokens")
        print(f"Specialization: {model_info.specialization}")
        print(f"Performance Rating: {model_info.performance_rating}")
        print(f"Status: {'🟢 Available' if model_info.local else '🔴 Not Downloaded'}")
        print(f"\nDescription:\n{model_info.description}")
        print(f"\nCapabilities: {', '.join(model_info.capabilities)}")
        print(f"Intended Use: {', '.join(model_info.intended_use)}")
        
        if model_info.usage_count > 0:
            print(f"\nUsage Statistics:")
            print(f"  Times Used: {model_info.usage_count}")
            print(f"  Last Used: {model_info.last_used}")
        
        if not model_info.local:
            print(f"\n{Fore.YELLOW}💡 To download this model, run:{Style.RESET_ALL}")
            print(f"   python -m cli_interface model download {model_name}")
        
    except Exception as e:
        print(Fore.RED + f"Error getting model info: {e}" + Style.RESET_ALL)

async def handle_global_set_model(model_name: str):
    """Handle global --set-model flag"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        print(f"Setting active model to: {model_name}...")
        success = await model_switcher.switch_model(model_name)
        
        if success:
            print(Fore.GREEN + f"✅ Active model set to: {model_name}" + Style.RESET_ALL)
            model_info = model_switcher.get_current_model_info()
            if model_info:
                print(f"   Size: {model_info.size_gb:.1f} GB")
                print(f"   Specialization: {model_info.specialization}")
                print(f"   Context: {model_info.context_length:,} tokens")
        else:
            print(Fore.RED + f"❌ Failed to set model to: {model_name}" + Style.RESET_ALL)
            
    except Exception as e:
        print(Fore.RED + f"Error setting model: {e}" + Style.RESET_ALL)

async def handle_global_model_override(model_name: str):
    """Handle global --model flag (session override)"""
    try:
        model_switcher = ModelSwitcher()
        await model_switcher.initialize()
        
        available_models = model_switcher.list_available_models()
        if model_name not in available_models:
            print(Fore.YELLOW + f"⚠️  Model '{model_name}' not available locally for session override." + Style.RESET_ALL)
            return
        
        # Store original model for restoration
        original_model = model_switcher.get_current_model()
        
        # Temporarily switch to the override model
        success = await model_switcher.switch_model(model_name)
        if success:
            print(Fore.BLUE + f"🔄 Session model override: {model_name}" + Style.RESET_ALL)
            print(f"   Original model: {original_model}")
            print(f"   Override model: {model_name}")
            print(f"   Note: This override is temporary for this CLI session.")
        
    except Exception as e:
        print(Fore.RED + f"Error with model override: {e}" + Style.RESET_ALL)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting Nemesis-Nexus CLI.")

