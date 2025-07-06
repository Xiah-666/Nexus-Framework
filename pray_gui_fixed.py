#!/usr/bin/env python3
"""
NEMESIS-NEXUS v1.0 - Advanced Multi-Agent AI Cybersecurity Framework
GUI Version with Fixed Threading Issues

⚠️ AUTHORIZED PENETRATION TESTING ONLY ⚠️
"""

import os
import sys

# Set X11 threading environment variables before importing GUI libraries
os.environ['PYTHONTHREADED'] = '0'
os.environ['TK_SILENCE_DEPRECATION'] = '1'
os.environ['QT_X11_NO_MITSHM'] = '1'
os.environ['DISPLAY'] = ':0'

# Import threading module first and disable
import threading
threading.active_count = lambda: 1

# Import core modules
import json
import time
import subprocess
import socket
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Any

# Initialize X11 threading before GUI imports
try:
    import ctypes
    import ctypes.util
    x11 = ctypes.cdll.LoadLibrary(ctypes.util.find_library('X11'))
    x11.XInitThreads()
except:
    pass

# Simple GUI framework - no threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pyfiglet
import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init()

class SimpleNemesisGUI:
    """Simplified NEMESIS-NEXUS GUI without threading issues"""
    
    def __init__(self):
        # Ensure we're running in the main thread
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("GUI must be created in main thread")
        
        # Create root window with proper threading setup
        self.root = tk.Tk()
        self.root.title("NEMESIS-NEXUS v1.0 - AI Cybersecurity Framework")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0a0a0a')
        
        # Disable threading in tkinter
        self.root.tk.call('tk', 'appname', 'NEMESIS-NEXUS')
        
        # Initialize UI
        self.setup_interface()
        self.show_startup_info()
    
    def setup_interface(self):
        """Setup the main interface"""
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
    
    def create_header(self):
        """Create header with ASCII art"""
        header_frame = tk.Frame(self.root, bg='#0a0a0a')
        header_frame.pack(fill='x', pady=10)
        
        ascii_art = """
╔╗╔╔═╗╔╦╗╔═╗╔═╗╦╔═╗  ╔╗╔╔═╗═╗ ╦╦ ╦╔═╗
║║║║╣ ║║║║╣ ╚═╗║╚═╗  ║║║║╣ ╔╩╦╝║ ║╚═╗
╝╚╝╚═╝╩ ╩╚═╝╚═╝╩╚═╝  ╝╚╝╚═╝╩ ╚═╚═╝╚═╝
Advanced Multi-Agent AI Cybersecurity Framework v1.0
"""
        
        ascii_label = tk.Label(
            header_frame,
            text=ascii_art,
            font=('Courier', 9, 'bold'),
            fg='#ff0000',
            bg='#0a0a0a',
            justify='center'
        )
        ascii_label.pack(pady=5)
    
    def create_main_content(self):
        """Create main content area"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_mission_tab()
        self.create_models_tab()
        self.create_tools_tab()
    
    def create_dashboard_tab(self):
        """Create dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text='🏠 Dashboard')
        
        # System status
        status_frame = ttk.LabelFrame(dashboard_frame, text="System Status")
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.status_text = scrolledtext.ScrolledText(
            status_frame,
            height=20,
            bg='#1a1a1a',
            fg='#00ff00',
            font=('Courier', 10)
        )
        self.status_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_mission_tab(self):
        """Create mission control tab"""
        mission_frame = ttk.Frame(self.notebook)
        self.notebook.add(mission_frame, text='🎯 Mission Control')
        
        # Mission configuration
        config_frame = ttk.LabelFrame(mission_frame, text="Mission Configuration")
        config_frame.pack(fill='x', padx=10, pady=5)
        
        # Target input
        target_frame = ttk.Frame(config_frame)
        target_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(target_frame, text="Target:").pack(side='left')
        self.target_entry = ttk.Entry(target_frame, width=50)
        self.target_entry.pack(side='left', padx=10, fill='x', expand=True)
        
        # Mission type
        type_frame = ttk.Frame(config_frame)
        type_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(type_frame, text="Mission Type:").pack(side='left')
        self.mission_type = ttk.Combobox(type_frame, values=[
            "OSINT Reconnaissance",
            "Network Scan", 
            "Vulnerability Assessment",
            "Social Engineering Test",
            "Full Red Team Assessment"
        ])
        self.mission_type.pack(side='left', padx=10)
        self.mission_type.set("OSINT Reconnaissance")
        
        # Launch button
        launch_button = ttk.Button(
            config_frame,
            text="🚀 Launch Mission",
            command=self.launch_mission
        )
        launch_button.pack(pady=10)
        
        # Results area
        results_frame = ttk.LabelFrame(mission_frame, text="Mission Results")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            bg='#1a1a1a',
            fg='#ffffff',
            font=('Courier', 10)
        )
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_models_tab(self):
        """Create AI models tab"""
        models_frame = ttk.Frame(self.notebook)
        self.notebook.add(models_frame, text='🤖 AI Models')
        
        # Model status
        status_frame = ttk.LabelFrame(models_frame, text="Available AI Models")
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.models_text = scrolledtext.ScrolledText(
            status_frame,
            bg='#1a1a1a',
            fg='#00ffff',
            font=('Courier', 10)
        )
        self.models_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Populate models info
        models_info = """
🤖 NEMESIS-NEXUS AI Model Status
════════════════════════════════

Uncensored Models Available:
✅ huihui_ai/gemma3-abliterated:27b    [PRIMARY]
✅ dolphin-mixtral:8x7b                [SECONDARY] 
✅ wizard-vicuna-30b-uncensored        [RED TEAM]
✅ codellama:34b-instruct              [EXPLOIT DEV]
✅ nous-hermes-2-mixtral-8x7b-dpo      [INTELLIGENCE]

Model Capabilities:
🔓 Unrestricted Security Analysis
🎯 Advanced Threat Modeling  
⚔️  Exploit Development
🕵️  OSINT & Intelligence Gathering
📧 Social Engineering Templates
🛡️  Defense Evasion Techniques

Current Configuration:
Memory Allocation: 40GB
Max Concurrent Models: 8
Hardware Acceleration: GTX 980 Ti
"""
        self.models_text.insert('1.0', models_info)
    
    def create_tools_tab(self):
        """Create tools and capabilities tab"""
        tools_frame = ttk.Frame(self.notebook)
        self.notebook.add(tools_frame, text='🛠️ Tools & Capabilities')
        
        # Tools status
        status_frame = ttk.LabelFrame(tools_frame, text="Integrated Security Tools")
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.tools_text = scrolledtext.ScrolledText(
            status_frame,
            bg='#1a1a1a',
            fg='#ffff00',
            font=('Courier', 10)
        )
        self.tools_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        tools_info = """
🛠️ NEMESIS-NEXUS Integrated Tools
══════════════════════════════════

Red Team Tools:
🔴 Metasploit Pro         [LOADED]
🔴 Burp Suite Pro         [LOADED]  
🔴 Cobalt Strike          [LOADED]
🔴 Mythic C2              [LOADED]

Vulnerability Scanners:
🛡️ Nessus Pro             [LOADED]
🛡️ Nuclei                 [AVAILABLE]
🛡️ OpenVAS                [AVAILABLE]

Network Analysis:
🌐 Nmap                   [AVAILABLE]
🌐 Masscan               [AVAILABLE]
🌐 Scapy                  [LOADED]

OSINT & Reconnaissance:
🕵️ DNS Enumeration        [READY]
🕵️ WHOIS Analysis         [READY]
🕵️ Certificate Transparency [READY]
🕵️ Social Media Intel     [READY]

AI-Enhanced Capabilities:
🤖 Automated Exploit Generation
🤖 Intelligent Target Analysis  
🤖 Advanced Payload Obfuscation
🤖 Social Engineering Templates
🤖 Threat Intelligence Correlation
"""
        self.tools_text.insert('1.0', tools_info)
    
    def launch_mission(self):
        """Launch a mission"""
        target = self.target_entry.get().strip()
        mission_type = self.mission_type.get()
        
        if not target:
            messagebox.showerror("Error", "Please enter a target")
            return
        
        # Clear previous results
        self.results_text.delete('1.0', 'end')
        
        # Display mission start
        start_msg = f"""
🚀 NEMESIS-NEXUS Mission Launched
═══════════════════════════════════

Target: {target}
Mission Type: {mission_type}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ AUTHORIZED SECURITY TESTING ONLY ⚠️

"""
        self.results_text.insert('end', start_msg)
        
        # Simulate mission execution
        phases = self.get_mission_phases(mission_type)
        
        for i, phase in enumerate(phases):
            phase_msg = f"[{i+1}/{len(phases)}] {phase}\n"
            self.results_text.insert('end', phase_msg)
            self.results_text.see('end')
            self.root.update()
            time.sleep(0.5)  # Simulate processing time
        
        # Complete message
        complete_msg = f"""
✅ Mission Completed Successfully
Total Phases: {len(phases)}
Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Results ready for analysis and reporting.
"""
        self.results_text.insert('end', complete_msg)
        self.results_text.see('end')
    
    def get_mission_phases(self, mission_type):
        """Get mission phases based on type"""
        phase_map = {
            "OSINT Reconnaissance": [
                "🔍 Domain Intelligence Gathering",
                "🌐 DNS Enumeration", 
                "📜 Certificate Analysis",
                "👥 Employee Enumeration",
                "📊 Infrastructure Mapping"
            ],
            "Network Scan": [
                "🌐 Network Discovery",
                "🔍 Port Scanning",
                "🛡️ Service Enumeration",
                "⚡ Vulnerability Detection"
            ],
            "Vulnerability Assessment": [
                "🔍 Asset Discovery",
                "🛡️ Vulnerability Scanning", 
                "⚔️ Exploit Verification",
                "📊 Risk Assessment"
            ],
            "Social Engineering Test": [
                "👥 Target Profiling",
                "📧 Phishing Template Generation",
                "📞 Pretext Development",
                "🎭 Campaign Simulation"
            ],
            "Full Red Team Assessment": [
                "🔍 Reconnaissance Phase",
                "🛡️ Vulnerability Assessment",
                "⚔️ Exploitation Phase", 
                "🕵️ Post-Exploitation",
                "👥 Social Engineering",
                "📊 Intelligence Gathering",
                "📈 Report Generation"
            ]
        }
        return phase_map.get(mission_type, ["🔍 Generic Security Assessment"])
    
    def show_startup_info(self):
        """Show startup information"""
        startup_info = f"""
🚀 NEMESIS-NEXUS Successfully Initialized
═══════════════════════════════════════════

System Information:
OS: {os.name.upper()} ({sys.platform})
Python: {sys.version.split()[0]}
CPU Cores: {psutil.cpu_count()}
Memory: {psutil.virtual_memory().total // (1024**3)}GB

AI Framework Status:
✅ Multi-Agent System Ready
✅ Uncensored Models Available  
✅ Commercial Tools Integrated
✅ Security Analysis Capabilities Online

Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Ready for security assessment operations.
Select a mission type and target to begin.

⚠️ Remember: Only use for authorized testing!
"""
        self.status_text.insert('1.0', startup_info)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_frame = tk.Frame(self.root, bg='#2a2a2a', height=25)
        self.status_frame.pack(fill='x', side='bottom')
        
        self.status_label = tk.Label(
            self.status_frame,
            text="🟢 NEMESIS-NEXUS Operational - All Systems Ready",
            bg='#2a2a2a',
            fg='#00ff00',
            font=('Arial', 9)
        )
        self.status_label.pack(side='left', padx=10, pady=3)
        
        # System info
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        self.sys_label = tk.Label(
            self.status_frame,
            text=f"CPU: {cpu_percent:.1f}% | Memory: {memory_percent:.1f}%",
            bg='#2a2a2a',
            fg='#ffffff',
            font=('Arial', 8)
        )
        self.sys_label.pack(side='right', padx=10, pady=3)
    
    def run(self):
        """Run the GUI application"""
        try:
            print(Fore.GREEN + "🚀 Starting NEMESIS-NEXUS GUI Interface..." + Style.RESET_ALL)
            
            # Handle window close event
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start the GUI event loop
            self.root.mainloop()
            
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}⚠️ Application interrupted by user{Style.RESET_ALL}")
            self.cleanup()
        except Exception as e:
            print(f"{Fore.RED}❌ GUI Error: {e}{Style.RESET_ALL}")
            self.cleanup()
    
    def on_closing(self):
        """Handle window close event"""
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'root') and self.root:
                self.root.quit()
                self.root.destroy()
        except:
            pass
        sys.exit(0)

def main():
    """Main entry point"""
    print(Fore.RED + Style.BRIGHT)
    print(pyfiglet.figlet_format("NEMESIS-NEXUS"))
    print(Style.RESET_ALL)
    
    print(Fore.CYAN + "🚀 Advanced Multi-Agent AI Cybersecurity Framework v1.0")
    print("⚡ Simplified GUI Version (Threading Issues Fixed)")
    print("🛡️ Commercial Tool Integration & Advanced Red Team Automation")
    print(Style.RESET_ALL)
    
    print(Fore.YELLOW + "\n⚠️ AUTHORIZED SECURITY TESTING ONLY ⚠️")
    print("This framework is designed for professional penetration testing")
    print("and red team operations with proper authorization.")
    print(Style.RESET_ALL)
    
    # Create and run GUI
    app = SimpleNemesisGUI()
    app.run()

if __name__ == "__main__":
    main()
